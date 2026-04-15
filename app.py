import re
import time
from collections import Counter

import docx
import pdfplumber
import spacy
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="HireSense", layout="wide")

st.title("HireSense")
st.subheader("AI Resume & Job Description Matching Tool")

st.write(
    "Upload your resume and paste a job description to receive a match score, "
    "missing skills, keyword suggestions, and rewrite recommendations."
)

# -----------------------------
# Load Models
# -----------------------------
@st.cache_resource
def load_spacy():
    return spacy.blank("en")


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


nlp = load_spacy()
embed_model = load_embedding_model()

# -----------------------------
# Text Extraction Helpers
# -----------------------------
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


def extract_resume_text(uploaded_file):
    if uploaded_file is None:
        return ""

    filename = uploaded_file.name.lower()

    try:
        if filename.endswith(".pdf"):
            return extract_text_from_pdf(uploaded_file)
        if filename.endswith(".docx"):
            return extract_text_from_docx(uploaded_file)
        if filename.endswith(".txt"):
            return extract_text_from_txt(uploaded_file)
        return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


# -----------------------------
# Cleaning and NLP Helpers
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s\-\+#\.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_keywords(text):
    doc = nlp(text)
    keywords = []

    for token in doc:
        if (
            not token.is_stop
            and not token.is_punct
            and not token.is_space
            and len(token.text.strip()) > 2
        ):
            keywords.append(token.text.lower())

    return keywords


def extract_named_skills(text):
    common_skills = [
        "python", "java", "sql", "machine learning", "deep learning",
        "data analysis", "data science", "tensorflow", "pytorch",
        "streamlit", "fastapi", "docker", "aws", "google cloud",
        "nlp", "llm", "rag", "transformers", "pandas", "numpy",
        "scikit-learn", "excel", "power bi", "tableau", "communication",
        "leadership", "project management", "statistics", "api",
        "linux", "git", "hugging face", "spacy", "artificial intelligence",
        "data visualization", "matplotlib", "seaborn"
    ]

    lower_text = text.lower()
    found = []

    for skill in common_skills:
        if skill in lower_text:
            found.append(skill)

    return sorted(list(set(found)))


def get_top_keywords(text, top_n=20):
    keywords = extract_keywords(text)
    freq = Counter(keywords)
    return [word for word, count in freq.most_common(top_n)]


# -----------------------------
# Similarity and Scoring
# -----------------------------
def compute_match_score(resume_text, job_text):
    resume_emb = embed_model.encode([resume_text])
    job_emb = embed_model.encode([job_text])
    score = cosine_similarity(resume_emb, job_emb)[0][0]
    return round(float(score) * 100, 2)


def find_missing_keywords(resume_text, job_text):
    resume_keywords = set(get_top_keywords(resume_text, 40))
    job_keywords = set(get_top_keywords(job_text, 40))
    missing = sorted(list(job_keywords - resume_keywords))
    return missing[:20]


def find_missing_skills(resume_text, job_text):
    resume_skills = set(extract_named_skills(resume_text))
    job_skills = set(extract_named_skills(job_text))
    missing_skills = sorted(list(job_skills - resume_skills))
    matched_skills = sorted(list(job_skills.intersection(resume_skills)))
    return matched_skills, missing_skills


# -----------------------------
# Suggestion Generation
# -----------------------------
def rule_based_suggestions(missing_skills, missing_keywords):
    suggestions = []

    if missing_skills:
        suggestions.append(
            "Add or emphasize these missing skills if you truly have them: "
            + ", ".join(missing_skills[:10]) + "."
        )

    if missing_keywords:
        suggestions.append(
            "Try including these important keywords from the job description: "
            + ", ".join(missing_keywords[:10]) + "."
        )

    suggestions.append(
        "Rewrite bullet points so they begin with action verbs and show measurable impact."
    )
    suggestions.append(
        "Align your summary section more closely with the job title and required qualifications."
    )
    suggestions.append(
        "Use wording from the job description when it accurately reflects your experience."
    )

    return suggestions


# -----------------------------
# Bullet Point Rewriter
# -----------------------------
def simple_bullet_rewrites(resume_text, missing_skills):
    lines = [line.strip() for line in resume_text.split("\n") if line.strip()]
    bullets = [line for line in lines if line.startswith("-") or line.startswith("•")]

    rewrites = []

    for bullet in bullets[:3]:
        cleaned = bullet.lstrip("-• ").strip()
        if missing_skills:
            skill_text = ", ".join(missing_skills[:2])
            rewrite = f"Improved: {cleaned} while highlighting experience relevant to {skill_text}."
        else:
            rewrite = f"Improved: {cleaned} with stronger action verbs and measurable outcomes."
        rewrites.append(rewrite)

    if not rewrites:
        rewrites.append(
            "Add bullet points that start with action verbs and include results, metrics, or technologies used."
        )

    return rewrites


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload your resume as DOCX or TXT.")
st.sidebar.write("2. Paste the target job description.")
st.sidebar.write("3. Click Analyze.")
st.sidebar.write("4. Review the score and suggestions.")

# -----------------------------
# Main Inputs
# -----------------------------
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

# -----------------------------
# Main Logic
# -----------------------------
if analyze_button:
    if uploaded_resume is None:
        st.warning("Please upload a resume.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Analyzing..."):
            start_time = time.time()

            resume_text = extract_resume_text(uploaded_resume)
            if not resume_text.strip():
                st.error("Could not extract text from the uploaded resume.")
                st.stop()

            clean_resume = clean_text(resume_text)
            clean_job = clean_text(job_description)

            match_score = compute_match_score(clean_resume, clean_job)
            missing_keywords = find_missing_keywords(clean_resume, clean_job)
            matched_skills, missing_skills = find_missing_skills(clean_resume, clean_job)
            bullet_rewrites = simple_bullet_rewrites(resume_text, missing_skills)

            end_time = time.time()
            latency = round(end_time - start_time, 2)

        st.success("Analysis complete.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Match Score", f"{match_score}%")
            st.metric("Inference Time", f"{latency} sec")

        with col2:
            st.metric("Matched Skills", len(matched_skills))
            st.metric("Missing Skills", len(missing_skills))

        st.markdown("## Matched Skills")
        st.write(", ".join(matched_skills) if matched_skills else "No clearly matched skills found.")

        st.markdown("## Missing Skills")
        st.write(", ".join(missing_skills) if missing_skills else "No obvious missing skills detected.")

        st.markdown("## Missing Keywords")
        st.write(", ".join(missing_keywords) if missing_keywords else "No major missing keywords found.")

        st.markdown("## Resume Improvement Suggestions")
        for suggestion in rule_based_suggestions(missing_skills, missing_keywords):
            st.write(f"- {suggestion}")

        st.markdown("## Bullet Point Rewrite Ideas")
        for rewrite in bullet_rewrites:
            st.write(f"- {rewrite}")

        with st.expander("View Extracted Resume Text"):
            st.text(resume_text[:5000])

        with st.expander("View Job Description"):
            st.text(job_description[:5000])
