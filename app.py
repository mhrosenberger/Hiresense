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
    "Upload a resume and paste a job description to receive an overall match score, "
    "AI-extracted skills, and AI-generated improvement suggestions."
)


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


def normalize_skill_text(skill):
    skill = skill.lower().strip()
    skill = re.sub(r"^[\-\*\•\d\.\)\(]+", "", skill)
    skill = re.sub(r"[^a-zA-Z0-9\+\#\.\-/& ]", "", skill)
    skill = re.sub(r"\s+", " ", skill).strip()
    return skill


def is_valid_skill(skill):
    if not skill:
        return False

    if len(skill) < 2 or len(skill) > 50:
        return False

    bad_exact = {
        "none", "n/a", "na", "resume", "job", "description", "candidate",
        "employee", "applicant", "role", "position", "company", "organization"
    }

    if skill in bad_exact:
        return False

    bad_contains = [
        "@", "http", "www", ".com", ".org", ".edu",
        "benefits", "salary", "compensation", "location",
        "equal opportunity", "full time", "part time"
    ]

    if any(bad in skill for bad in bad_contains):
        return False

    if skill.isdigit():
        return False

    return True


# =========================================================
# AI SKILL EXTRACTION
# =========================================================
def ai_extract_skills(text):
    prompt = f"""
Extract only SKILLS from the text.

Skills include:
- tools
- technologies
- methods
- software
- programming languages
- technical processes

Return ONLY comma-separated items.
No sentences. No explanations.

Text:
{text[:2000]}
"""

    try:
        result = generator(prompt, max_length=150, do_sample=False)
        output = result[0]["generated_text"]

        raw = [s.strip().lower() for s in output.split(",") if s.strip()]

        cleaned = []

        for skill in raw:
            # remove punctuation
            skill = re.sub(r"[^a-zA-Z0-9\+\#\.\-/ ]", "", skill).strip()

            # ❌ REMOVE BAD SKILLS
            if (
                len(skill.split()) > 3  # too long = sentence
                or any(x in skill for x in [
                    "company", "global", "team", "working",
                    "building", "together", "future",
                    "customer", "employees", "environment"
                ])
                or re.search(r"\b(inc|llc|corp|ltd)\b", skill)
                or re.search(r"\b\d{4}\b", skill)  # years
                or skill in ["clean", "collect", "production"]
            ):
                continue

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
def generate_llm_suggestions(resume_text, job_text, matched_skills, missing_skills):
    prompt = f"""
You are a professional resume assistant.

Compare the resume and the job description below.

Resume:
{resume_text[:2200]}

Job Description:
{job_text[:2200]}

Matched Skills:
{", ".join(matched_skills)}

Missing Skills:
{", ".join(missing_skills)}

Provide:
1. Three practical improvements to make the resume better fit the job
2. Two stronger bullet point rewrite ideas
3. One short professional summary tailored to the job

Rules:
- Work for any profession or industry
- Be specific and practical
- Do not invent fake experience
- Only recommend adding skills if the person actually has them
- Keep the response concise
"""

    try:
        result = generator(prompt, max_length=260, do_sample=False)
        return result[0]["generated_text"]
    except Exception:
        return "AI suggestions are unavailable right now."


# =========================================================
# UI
# =========================================================
uploaded_resume = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx", "txt"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=300
)

analyze = st.button("Analyze")


# =========================================================
# MAIN
# =========================================================
if analyze:
    if uploaded_resume is None:
        st.warning("Please upload a resume.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
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

            llm_output = generate_llm_suggestions(
                clean_resume,
                clean_job,
                matched,
                missing
            )

            latency = round(time.time() - start, 2)

        st.success("Analysis complete.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Overall Match Score", f"{score}%")
            st.metric("Skill Match %", f"{percent}%")

        with col2:
            st.metric("Matched Skills", len(matched))
            st.metric("Missing Skills", len(missing))

        st.markdown("## Skills Extracted from Job Description")
        st.write(", ".join(sorted(job_skills)) if job_skills else "No skills found.")

        st.markdown("## Skills Extracted from Resume")
        st.write(", ".join(sorted(resume_skills)) if resume_skills else "No skills found.")

        st.markdown("## Matched Skills")
        st.write(", ".join(matched) if matched else "No matched skills found.")

        st.markdown("## Missing Skills")
        st.write(", ".join(missing) if missing else "No missing skills found.")

        st.markdown("## Extra Skills from Resume")
        st.write(", ".join(extra) if extra else "No extra skills found.")

        st.markdown("## AI Suggestions")
        st.write(llm_output)

        st.markdown("## Runtime")
        st.write(f"{latency} sec")

        with st.expander("View Resume Text"):
            st.text(resume_text[:5000])

        with st.expander("View Job Description"):
            st.text(job_text[:5000])
