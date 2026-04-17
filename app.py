import re
import time

import docx
import pdfplumber
import streamlit as st
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

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
def load_llm():
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
    return tokenizer, model


embed_model = load_embedding_model()
tokenizer, llm_model = load_llm()


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
# TEXT PREP
# =========================================================
def prepare_text(text):
    text = text.replace("\r", "\n")
    text = text.replace("\t", " ")
    text = re.sub(r"[•▪◦]", "•", text)
    text = re.sub(r"\n+", "\n", text)
    return text.strip()


def parse_skill_lines(output_text):
    lines = [line.strip() for line in output_text.split("\n") if line.strip()]

    cleaned = []
    for line in lines:
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        line = re.sub(r"^[-•*]\s*", "", line)
        line = line.strip()
        if line:
            cleaned.append(line)

    return sorted(list(set(cleaned)))


# =========================================================
# LLM HELPER
# =========================================================
def run_llm(prompt, max_new_tokens=120):
    try:
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=768
        )

        with torch.no_grad():
            outputs = llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=4,
                early_stopping=True
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return decoded
    except Exception:
        return ""


# =========================================================
# SECTION EXTRACTION
# =========================================================
def extract_resume_skills_section(resume_text):
    text = prepare_text(resume_text)
    match = re.search(
        r"skills\s*:(.*?)(activities\s*:|publications\s*:|projects\s*:|education\s*:|experience\s*:|$)",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return ""


def extract_resume_experience_projects(resume_text):
    text = prepare_text(resume_text)
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
    text = prepare_text(job_text)
    chunks = []

    for section_name in ["qualifications", "experience", "responsibilities", "position summary"]:
        match = re.search(
            rf"{section_name}\s*:(.*?)(benefits\s*:|compensation\s*:|education\s*:|work environment\s*:|co only\s*:|$)",
            text,
            flags=re.IGNORECASE | re.DOTALL
        )
        if match:
            chunks.append(match.group(1).strip())

    if chunks:
        return "\n".join(chunks).strip()

    return text[:2000]


# =========================================================
# AI SKILL IDENTIFICATION
# =========================================================
def ai_identify_resume_skills(resume_text):
    skills_section = extract_resume_skills_section(resume_text)
    exp_proj_section = extract_resume_experience_projects(resume_text)

    prompt = f"""
Extract the candidate's actual skills from this resume.

Rules:
- Output one skill per line
- Focus first on the Skills section
- Then include additional tools, software, programming languages, platforms,
  methods, and technical skills shown in projects and experience
- Do not include names, schools, locations, dates, GPA, employers, or publication citations
- Do not write sentences

Resume Skills Section:
{skills_section[:1200]}

Resume Experience and Projects:
{exp_proj_section[:1200]}
"""

    output = run_llm(prompt, max_new_tokens=120)
    return parse_skill_lines(output)


def ai_identify_job_skills(job_text):
    relevant_job_text = extract_job_relevant_section(job_text)

    prompt = f"""
Extract the required or preferred job skills from this job description.

Rules:
- Output one skill per line
- Include tools, software, programming languages, platforms, methods,
  technical skills, domain knowledge, and certifications
- Do not include company descriptions, benefits, locations, or sentence fragments
- Do not write sentences

Job Description:
{relevant_job_text[:1800]}
"""

    output = run_llm(prompt, max_new_tokens=120)
    return parse_skill_lines(output)


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

    return sorted(job_skills), sorted(resume_skills), matched, missing, extra, percent


# =========================================================
# SUGGESTIONS
# =========================================================
def generate_llm_suggestions(resume_text, job_text, matched_skills, missing_skills):
    prompt = f"""
Compare this resume and job description.

Matched skills:
{", ".join(matched_skills)}

Missing skills:
{", ".join(missing_skills)}

Write exactly:
1. Three practical resume improvements
2. Two bullet point rewrite ideas
3. One short professional summary

Resume:
{resume_text[:1400]}

Job Description:
{job_text[:1400]}
"""

    output = run_llm(prompt, max_new_tokens=180)
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

            prepared_resume = prepare_text(resume_text)
            prepared_job = prepare_text(job_text)

            score = compute_match_score(prepared_resume, prepared_job)

            job_skills, resume_skills, matched, missing, extra, percent = compare_skills(
                prepared_resume,
                prepared_job
            )

            llm_output = generate_llm_suggestions(
                prepared_resume,
                prepared_job,
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
        st.write(", ".join(job_skills) if job_skills else "No skills found.")

        st.markdown("## Skills Identified in Resume")
        st.write(", ".join(resume_skills) if resume_skills else "No skills found.")

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
