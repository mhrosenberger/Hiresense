import re
import time
from collections import Counter

import docx
import pdfplumber
import spacy
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# =========================================================
# CONFIG
# =========================================================
# For local GPU / stronger machine, use:
# LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
#
# For Streamlit Cloud / lighter deployment, use:
LLM_MODEL_NAME = "google/flan-t5-base"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

# Small dictionary for structured skill comparison
KNOWN_SKILLS = [
    "python", "r", "sql", "excel", "tableau", "power bi",
    "machine learning", "deep learning", "data analysis", "data analytics",
    "data science", "predictive modeling", "statistics", "statistical analysis",
    "database management", "spacy", "pytorch", "transformers",
    "big data", "hadoop", "spark", "anomaly detection",
    "correlation studies", "data modeling", "microsoft office", "office 365"
]


# =========================================================
# STREAMLIT PAGE
# =========================================================
st.set_page_config(page_title="HireSense", layout="wide")

st.title("HireSense")
st.subheader("AI Resume & Job Description Matching Web Application")

st.write(
    "Upload your resume and paste a job description to receive a match score, "
    "missing skills, extracted keywords, and AI-generated recommendations."
)


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_spacy():
    return spacy.load("en_core_web_sm")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_generator():
    return pipeline(
        "text2text-generation",
        model=LLM_MODEL_NAME,
        device_map="auto" if torch.cuda.is_available() else None
    )


nlp = load_spacy()
embed_model = load_embedding_model()
generator = load_generator()


# =========================================================
# FILE READING
# =========================================================
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


def extract_uploaded_text(uploaded_file):
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".pdf"):
            return extract_text_from_pdf(uploaded_file)
        elif filename.endswith(".docx"):
            return extract_text_from_docx(uploaded_file)
        elif filename.endswith(".txt"):
            return extract_text_from_txt(uploaded_file)
        else:
            return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


# =========================================================
# PREPROCESSING
# =========================================================
def clean_text(text):
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[•▪◦]", "•", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_phrase(text):
    text = text.lower().strip()
    text = re.sub(r"^[\-\•\*\d\.\)\(]+", "", text)
    text = re.sub(r"[^a-zA-Z0-9\+\#\/\-\.\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# =========================================================
# SPACY-BASED EXTRACTION
# =========================================================
def extract_keywords_spacy(text, top_n=25):
    doc = nlp(text)
    keywords = []

    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.text.strip()) > 2
            and token.pos_ in ["NOUN", "PROPN", "ADJ"]
        ):
            keywords.append(token.lemma_.lower())

    freq = Counter(keywords)
    return [word for word, _ in freq.most_common(top_n)]


def extract_entities_spacy(text):
    doc = nlp(text)
    entities = set()

    for ent in doc.ents:
        cleaned = normalize_phrase(ent.text)
        if cleaned and len(cleaned.split()) <= 4:
            entities.add(cleaned)

    return sorted(entities)


def extract_known_skills(text):
    lower_text = text.lower()
    found = []

    for skill in KNOWN_SKILLS:
        pattern = r"\b" + re.escape(skill.lower()) + r"\b"
        if re.search(pattern, lower_text):
            found.append(skill)

    return sorted(list(set(found)))


# =========================================================
# EMBEDDING MATCH SCORE
# =========================================================
def compute_match_score(resume_text, job_text):
    resume_emb = embed_model.encode([resume_text])
    job_emb = embed_model.encode([job_text])
    score = cosine_similarity(resume_emb, job_emb)[0][0]
    return round(float(score) * 100, 2)


# =========================================================
# SKILL COMPARISON
# =========================================================
def compare_skills(resume_text, job_text):
    resume_skills = set(extract_known_skills(resume_text))
    job_skills = set(extract_known_skills(job_text))

    matched_skills = sorted(job_skills.intersection(resume_skills))
    missing_skills = sorted(job_skills - resume_skills)
    extra_resume_skills = sorted(resume_skills - job_skills)

    skill_match_percent = 0.0
    if len(job_skills) > 0:
        skill_match_percent = round((len(matched_skills) / len(job_skills)) * 100, 2)

    return {
        "resume_skills": sorted(resume_skills),
        "job_skills": sorted(job_skills),
        "matched_skills": matched_skills,
        "missing_skills": missing_skills,
        "extra_resume_skills": extra_resume_skills,
        "skill_match_percent": skill_match_percent,
    }


def find_missing_keywords(resume_text, job_text):
    resume_keywords = set(extract_keywords_spacy(resume_text, 40))
    job_keywords = set(extract_keywords_spacy(job_text, 40))
    missing = sorted(list(job_keywords - resume_keywords))
    return missing[:20]


# =========================================================
# LLM-BASED SUGGESTIONS
# =========================================================
def generate_llm_suggestions(resume_text, job_text, matched_skills, missing_skills, missing_keywords):
    prompt = f"""
You are a resume assistant.

Compare the resume and job description below.

Resume:
{resume_text[:2500]}

Job Description:
{job_text[:2500]}

Matched Skills:
{", ".join(matched_skills)}

Missing Skills:
{", ".join(missing_skills)}

Missing Keywords:
{", ".join(missing_keywords)}

Please provide:
1. Three specific resume improvement suggestions
2. Two bullet point rewrite ideas
3. A short professional summary tailored to the job

Keep the response concise, practical, and grounded in the provided texts.
"""

    try:
        result = generator(prompt, max_length=300, do_sample=False)
        if result and len(result) > 0:
            return result[0]["generated_text"]
    except Exception as e:
        return f"LLM generation failed: {e}"

    return "No AI suggestions available."


# =========================================================
# SIMPLE RULE-BASED BULLET SUPPORT
# =========================================================
def simple_bullet_rewrites(resume_text, missing_skills):
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    bullets = [line for line in lines if line.startswith("-") or line.startswith("•")]

    rewrites = []

    for bullet in bullets[:3]:
        cleaned = bullet.lstrip("-• ").strip()
        if missing_skills:
            rewrites.append(
                f"Improved: {cleaned} while emphasizing experience related to {', '.join(missing_skills[:2])}."
            )
        else:
            rewrites.append(
                f"Improved: {cleaned} with stronger action verbs and measurable outcomes."
            )

    if not rewrites:
        rewrites.append(
            "Add bullet points that start with action verbs and include results, tools, or measurable outcomes."
        )

    return rewrites


# =========================================================
# SIDEBAR
# =========================================================
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload your resume as PDF, DOCX, or TXT.")
st.sidebar.write("2. Paste the target job description.")
st.sidebar.write("3. Click Analyze.")
st.sidebar.write("4. Review the score, missing skills, and AI suggestions.")


# =========================================================
# INPUTS
# =========================================================
uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx", "txt"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=300,
    placeholder="Paste the full job description here..."
)

analyze_button = st.button("Analyze Resume Match")


# =========================================================
# MAIN APP
# =========================================================
if analyze_button:
    if uploaded_resume is None:
        st.warning("Please upload a resume.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Analyzing..."):
            start_time = time.time()

            resume_text = extract_uploaded_text(uploaded_resume)
            if not resume_text.strip():
                st.error("Could not extract text from the uploaded resume.")
                st.stop()

            clean_resume = clean_text(resume_text)
            clean_job = clean_text(job_description)

            match_score = compute_match_score(clean_resume, clean_job)

            skill_results = compare_skills(clean_resume, clean_job)
            missing_keywords = find_missing_keywords(clean_resume, clean_job)
            resume_entities = extract_entities_spacy(clean_resume)
            job_entities = extract_entities_spacy(clean_job)

            llm_output = generate_llm_suggestions(
                clean_resume,
                clean_job,
                skill_results["matched_skills"],
                skill_results["missing_skills"],
                missing_keywords
            )

            bullet_rewrites = simple_bullet_rewrites(
                resume_text,
                skill_results["missing_skills"]
            )

            end_time = time.time()
            latency = round(end_time - start_time, 2)

        st.success("Analysis complete.")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Overall Match Score", f"{match_score}%")
            st.metric("Inference Time", f"{latency} sec")

        with col2:
            st.metric("Skill Match %", f'{skill_results["skill_match_percent"]}%')
            st.metric("Matched Skills", len(skill_results["matched_skills"]))

        with col3:
            st.metric("Missing Skills", len(skill_results["missing_skills"]))
            st.metric("Missing Keywords", len(missing_keywords))

        st.markdown("## Skills Found in Job Description")
        st.write(", ".join(skill_results["job_skills"]) if skill_results["job_skills"] else "No known job skills found.")

        st.markdown("## Skills Found in Resume")
        st.write(", ".join(skill_results["resume_skills"]) if skill_results["resume_skills"] else "No known resume skills found.")

        st.markdown("## Matched Skills")
        st.write(", ".join(skill_results["matched_skills"]) if skill_results["matched_skills"] else "No matched skills found.")

        st.markdown("## Missing Skills")
        st.write(", ".join(skill_results["missing_skills"]) if skill_results["missing_skills"] else "No missing skills found.")

        st.markdown("## Extra Skills on Resume")
        st.write(", ".join(skill_results["extra_resume_skills"]) if skill_results["extra_resume_skills"] else "No extra resume skills found.")

        st.markdown("## Missing Keywords")
        st.write(", ".join(missing_keywords) if missing_keywords else "No major missing keywords found.")

        st.markdown("## spaCy Entities from Job Description")
        st.write(", ".join(job_entities[:30]) if job_entities else "No entities found.")

        st.markdown("## spaCy Entities from Resume")
        st.write(", ".join(resume_entities[:30]) if resume_entities else "No entities found.")

        st.markdown("## Bullet Point Rewrite Ideas")
        for rewrite in bullet_rewrites:
            st.write(f"- {rewrite}")

        st.markdown("## AI-Generated Recommendations")
        st.write(llm_output)

        with st.expander("View Extracted Resume Text"):
            st.text(resume_text[:5000])

        with st.expander("View Job Description"):
            st.text(job_description[:5000])
