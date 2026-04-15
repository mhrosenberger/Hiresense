import re
import time

import docx
import pdfplumber
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

LLM_MODEL_NAME = "google/flan-t5-base"
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title="HireSense", layout="wide")

st.title("HireSense")
st.subheader("AI Resume & Job Description Matching Web Application")

st.write(
    "Upload your resume and paste a job description to receive a match score, "
    "AI-extracted skills, and AI-generated recommendations."
)


# =========================================================
# LOAD MODELS
# =========================================================
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_generator():
    return pipeline("text-generation", model=LLM_MODEL_NAME)


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
        return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


# =========================================================
# CLEANING
# =========================================================
def clean_text(text):
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[•▪◦]", "•", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# AI SKILL EXTRACTION
# =========================================================
def ai_extract_skills(text):
    prompt = f"""
Extract ONLY technical skills, tools, programming languages, libraries, software, and platforms.

STRICT RULES:
- NO locations
- NO benefits
- NO company names
- NO universities
- NO people's names
- NO dates
- NO soft skills
- ONLY real technical skills

Return ONLY a clean comma-separated list.
Do not explain anything.

Text:
{text[:2000]}
"""

    try:
        result = generator(prompt, max_length=150, do_sample=False)
        output = result[0]["generated_text"]

        skills = [s.strip().lower() for s in output.split(",") if s.strip()]

        blacklist = {
            "colorado", "company", "benefits", "environment",
            "fort collins", "denver", "university",
            "communication", "leadership", "teamwork"
        }

        cleaned = []
        for skill in skills:
            skill = re.sub(r"[^a-zA-Z0-9\+\#\.\-/ ]", "", skill).strip()
            if skill and skill not in blacklist and len(skill) < 40:
                cleaned.append(skill)

        return sorted(list(set(cleaned)))

    except:
        return []


# =========================================================
# MATCHING
# =========================================================
def compute_match_score(resume_text, job_text):
    resume_emb = embed_model.encode([resume_text])
    job_emb = embed_model.encode([job_text])
    score = cosine_similarity(resume_emb, job_emb)[0][0]
    return round(float(score) * 100, 2)


def compare_skills(resume_text, job_text):
    resume_skills = set(ai_extract_skills(resume_text))
    job_skills = set(ai_extract_skills(job_text))

    matched = sorted(job_skills.intersection(resume_skills))
    missing = sorted(job_skills - resume_skills)
    extra = sorted(resume_skills - job_skills)

    percent = 0
    if len(job_skills) > 0:
        percent = round((len(matched) / len(job_skills)) * 100, 2)

    return job_skills, resume_skills, matched, missing, extra, percent


# =========================================================
# LLM SUGGESTIONS
# =========================================================
def generate_llm_suggestions(resume_text, job_text, missing_skills):
    prompt = f"""
You are a resume assistant.

Resume:
{resume_text[:2000]}

Job Description:
{job_text[:2000]}

Missing Skills:
{", ".join(missing_skills)}

Give:
1. 3 improvements
2. 2 bullet rewrites
3. short summary

Keep it concise.
"""

    try:
        result = generator(prompt, max_length=200, do_sample=False)
        return result[0]["generated_text"]
    except:
        return "LLM unavailable."


# =========================================================
# UI
# =========================================================
uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

job_description = st.text_area("Paste Job Description", height=300)

analyze = st.button("Analyze")


# =========================================================
# MAIN
# =========================================================
if analyze:
    if uploaded_resume is None:
        st.warning("Upload resume.")
    elif not job_description.strip():
        st.warning("Paste job description.")
    else:
        with st.spinner("Analyzing..."):
            start = time.time()

            resume_text = extract_uploaded_text(uploaded_resume)
            job_text = job_description

            clean_resume = clean_text(resume_text)
            clean_job = clean_text(job_text)

            score = compute_match_score(clean_resume, clean_job)

            job_skills, resume_skills, matched, missing, extra, percent = compare_skills(
                clean_resume,
                clean_job
            )

            llm_output = generate_llm_suggestions(clean_resume, clean_job, missing)

            latency = round(time.time() - start, 2)

        st.success("Done")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Match Score", f"{score}%")
            st.metric("Skill Match %", f"{percent}%")

        with col2:
            st.metric("Matched Skills", len(matched))
            st.metric("Missing Skills", len(missing))

        st.markdown("## Skills from Job")
        st.write(", ".join(job_skills))

        st.markdown("## Skills from Resume")
        st.write(", ".join(resume_skills))

        st.markdown("## Matched Skills")
        st.write(", ".join(matched))

        st.markdown("## Missing Skills")
        st.write(", ".join(missing))

        st.markdown("## Extra Resume Skills")
        st.write(", ".join(extra))

        st.markdown("## AI Suggestions")
        st.write(llm_output)

        st.markdown("## Runtime")
        st.write(f"{latency} sec")
