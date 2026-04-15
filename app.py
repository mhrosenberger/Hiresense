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
    "AI-identified skills, and AI-generated improvement suggestions."
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
    text = re.sub(r"\n+", "\n", text)
    text = re.sub(r"[ ]+", " ", text)
    return text.strip()


def normalize_skill(skill):
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

    if len(skill.split()) > 5:
        return False

    bad_exact = {
        "resume", "job", "description", "candidate", "employee", "applicant",
        "skills", "experience", "projects", "education", "activities",
        "company", "organization", "none", "n/a"
    }

    if skill in bad_exact:
        return False

    bad_contains = [
        "@", "http", "www", ".com", ".org", ".edu",
        "benefits", "salary", "compensation", "location",
        "equal opportunity", "full time", "part time",
        "expected graduation", "gpa"
    ]

    if any(bad in skill for bad in bad_contains):
        return False

    # remove obvious dates / citation fragments
    if re.search(r"\b(19|20)\d{2}\b", skill):
        return False

    if re.search(r"\b[a-z]\.\b", skill):
        return False

    return True


def dedupe_skills(skills):
    cleaned = []
    seen = set()

    replacements = {
        "microsoft office suite": "microsoft office",
        "basic rstudio": "rstudio",
        "basic matlab": "matlab",
    }

    for skill in skills:
        s = normalize_skill(skill)
        s = replacements.get(s, s)

        if not is_valid_skill(s):
            continue

        if s not in seen:
            seen.add(s)
            cleaned.append(s)

    return sorted(cleaned)


def parse_llm_list(output_text):
    parts = re.split(r",|\n|;", output_text)
    return dedupe_skills([p.strip() for p in parts if p.strip()])


def run_llm(prompt, max_length=220):
    try:
        result = generator(prompt, max_length=max_length, do_sample=False)
        return result[0]["generated_text"]
    except Exception:
        return ""


# =========================================================
# SECTION HELPERS
# =========================================================
def extract_resume_skills_section(resume_text):
    text = clean_text(resume_text)
    match = re.search(
        r"skills\s*:(.*?)(activities\s*:|publications\s*:|projects\s*:|education\s*:|experience\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""


def extract_resume_project_experience_section(resume_text):
    text = clean_text(resume_text)
    chunks = []

    for section_name in ["experience", "projects"]:
        match = re.search(
            rf"{section_name}\s*:(.*?)(education\s*:|skills\s*:|activities\s*:|publications\s*:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            chunks.append(match.group(1).strip())

    return "\n".join(chunks).strip()


def extract_job_relevant_section(job_text):
    text = clean_text(job_text)
    chunks = []

    for section_name in ["qualifications", "requirements", "experience", "responsibilities", "position summary"]:
        match = re.search(
            rf"{section_name}\s*:(.*?)(benefits\s*:|compensation\s*:|education\s*:|work environment\s*:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            chunks.append(match.group(1).strip())

    if chunks:
        return "\n".join(chunks).strip()

    return text[:2500]


# =========================================================
# AI SKILL IDENTIFICATION
# =========================================================
def ai_identify_resume_skills(resume_text):
    skills_section = extract_resume_skills_section(resume_text)
    exp_proj_section = extract_resume_project_experience_section(resume_text)

    prompt = f"""
Identify the applicant's real skills from this resume.

Instructions:
- First prioritize the explicit Skills section.
- Then identify additional skills shown in projects and experience.
- Include tools, software, programming languages, platforms, methods, certifications,
  and job-relevant professional skills.
- Do NOT include names, schools, locations, GPA, dates, employers, publication citations,
  or generic filler words.

Return only a comma-separated list.
Do not explain anything.

Resume Skills Section:
{skills_section[:1200]}

Resume Projects and Experience:
{exp_proj_section[:1500]}
"""

    output = run_llm(prompt, max_length=220)
    return parse_llm_list(output)


def ai_identify_job_skills(job_text):
    relevant_job_text = extract_job_relevant_section(job_text)

    prompt = f"""
Identify the skills required or preferred for this job.

Include:
- tools
- software
- programming languages
- platforms
- methods
- certifications
- domain knowledge
- technical and professional skills

Do NOT include:
- company descriptions
- benefits
- locations
- marketing text
- sentence fragments
- generic filler language

Return only a comma-separated list.
Do not explain anything.

Job Text:
{relevant_job_text[:2200]}
"""

    output = run_llm(prompt, max_length=220)
    return parse_llm_list(output)


# =========================================================
# MATCHING
# =========================================================
def compute_match_score(resume_text, job_text):
    resume_emb = embed_model.encode([resume_text])
    job_emb = embed_model.encode([job_text])
    score = cosine_similarity(resume_emb, job_emb)[0][0]
    return round(float(score) * 100, 2)


def compare_skills(resume_text, job_text):
    resume_skills = set(ai_identify_resume_skills(resume_text))
    job_skills = set(ai_identify_job_skills(job_text))

    matched = sorted(job_skills.intersection(resume_skills))
    missing = sorted(job_skills - resume_skills)
    extra = sorted(resume_skills - job_skills)

    percent = 0
    if len(job_skills) > 0:
        percent = round((len(matched) / len(job_skills)) * 100, 2)

    return job_skills, resume_skills, matched, missing, extra, percent


# =========================================================
# SUGGESTIONS
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

    output = run_llm(prompt, max_length=260)
    if output.strip():
        return output
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

        st.markdown("## Skills Identified in Job Description")
        st.write(", ".join(sorted(job_skills)) if job_skills else "No skills found.")

        st.markdown("## Skills Identified in Resume")
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
