import json
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


# =========================================================
# LOAD MODELS
# =========================================================
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


# =========================================================
# SECTION EXTRACTION
# =========================================================
def extract_section(text, start_labels, stop_labels):
    """
    Extracts text between a start label and the next stop label.
    """
    prepared = prepare_text(text)
    lower = prepared.lower()

    starts = []
    for label in start_labels:
        idx = lower.find(label.lower() + ":")
        if idx != -1:
            starts.append(idx)

    if not starts:
        return ""

    start_idx = min(starts)
    tail = prepared[start_idx:]

    stop_positions = []
    tail_lower = tail.lower()
    for label in stop_labels:
        idx = tail_lower.find(label.lower() + ":")
        if idx > 0:
            stop_positions.append(idx)

    if stop_positions:
        end_idx = min(stop_positions)
        return tail[:end_idx].strip()

    return tail.strip()


def extract_resume_relevant_text(resume_text):
    skills = extract_section(
        resume_text,
        start_labels=["skills"],
        stop_labels=["activities", "publications", "projects", "education", "experience"]
    )

    experience = extract_section(
        resume_text,
        start_labels=["experience"],
        stop_labels=["education", "skills", "activities", "publications", "projects"]
    )

    projects = extract_section(
        resume_text,
        start_labels=["projects"],
        stop_labels=["education", "skills", "activities", "publications", "experience"]
    )

    relevant_parts = []
    if skills:
        relevant_parts.append(skills)
    if experience:
        relevant_parts.append(experience)
    if projects:
        relevant_parts.append(projects)

    return "\n\n".join(relevant_parts).strip()


def extract_job_relevant_text(job_text):
    qualifications = extract_section(
        job_text,
        start_labels=["qualifications"],
        stop_labels=["experience", "education", "compensation", "benefits", "work environment", "co only"]
    )

    experience = extract_section(
        job_text,
        start_labels=["experience"],
        stop_labels=["education", "compensation", "benefits", "work environment", "co only"]
    )

    responsibilities = extract_section(
        job_text,
        start_labels=["responsibilities"],
        stop_labels=["work environment", "qualifications", "experience", "education", "compensation", "benefits", "co only"]
    )

    summary = extract_section(
        job_text,
        start_labels=["position summary"],
        stop_labels=["responsibilities", "work environment", "qualifications", "experience", "education", "compensation", "benefits", "co only"]
    )

    relevant_parts = []
    if qualifications:
        relevant_parts.append(qualifications)
    if experience:
        relevant_parts.append(experience)
    if responsibilities:
        relevant_parts.append(responsibilities)
    if summary:
        relevant_parts.append(summary)

    return "\n\n".join(relevant_parts).strip()


# =========================================================
# LLM HELPER
# =========================================================
def run_llm(prompt, max_new_tokens=180):
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


def parse_json_skills(output_text):
    """
    Expect output like:
    ["python", "sql", "tableau"]
    """
    try:
        start = output_text.find("[")
        end = output_text.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_text = output_text[start:end + 1]
            data = json.loads(json_text)
            if isinstance(data, list):
                return sorted(list(set([str(x).strip() for x in data if str(x).strip()])))
    except Exception:
        pass

    # fallback: line/comma split if model doesn't return valid JSON
    parts = re.split(r",|\n|;", output_text)
    skills = [p.strip() for p in parts if p.strip()]
    return sorted(list(set(skills)))


# =========================================================
# AI SKILL IDENTIFICATION
# =========================================================
def ai_identify_resume_skills(resume_text):
    relevant_text = extract_resume_relevant_text(resume_text)

    prompt = f"""
Extract the candidate's actual skills from the resume content below.

Return ONLY a JSON array of strings.

Include:
- software
- tools
- programming languages
- platforms
- technical methods
- certifications
- professional skills clearly shown in the resume

Do NOT include:
- names
- schools
- locations
- dates
- GPA
- employers
- publication titles
- sentence fragments

Example output:
["python", "sql", "tableau"]

Resume content:
{relevant_text[:2200]}
"""

    output = run_llm(prompt, max_new_tokens=160)
    return parse_json_skills(output)


def ai_identify_job_skills(job_text):
    relevant_text = extract_job_relevant_text(job_text)

    prompt = f"""
Extract the required or preferred skills from the job content below.

Return ONLY a JSON array of strings.

Include:
- software
- tools
- programming languages
- platforms
- technical methods
- certifications
- domain knowledge
- professional skills required for the role

Do NOT include:
- company descriptions
- benefits
- locations
- marketing language
- sentence fragments

Example output:
["python", "sql", "power bi"]

Job content:
{relevant_text[:2200]}
"""

    output = run_llm(prompt, max_new_tokens=160)
    return parse_json_skills(output)


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
Write exactly:
1. Three practical resume improvements
2. Two bullet point rewrite ideas
3. One short professional summary

Use the information below.

Matched skills:
{", ".join(matched_skills)}

Missing skills:
{", ".join(missing_skills)}

Resume:
{resume_text[:1400]}

Job Description:
{job_text[:1400]}
"""

    output = run_llm(prompt, max_new_tokens=180)
    return output if output.strip() else "AI suggestions are unavailable right now."


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
