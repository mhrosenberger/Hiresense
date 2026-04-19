import json
import os
import time

import docx
import pdfplumber
import streamlit as st
from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Set these in Streamlit secrets or environment variables
HF_TOKEN = st.secrets.get("HF_TOKEN", os.getenv("HF_TOKEN", ""))
HF_MODEL = st.secrets.get(
    "HF_MODEL",
    os.getenv("HF_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
)

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

st.set_page_config(page_title="HireSense", layout="wide")

st.title("HireSense")
st.subheader("AI Resume & Job Description Matching Web Application")

st.write(
    "Upload a resume (PDF or Word file) and paste a job description to receive an overall match score, "
    "AI-extracted skills, and AI-generated improvement suggestions."
)



# LOAD MODELS

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@st.cache_resource
def load_llm_client():
    return InferenceClient(api_key=HF_TOKEN)


embed_model = load_embedding_model()
hf_client = load_llm_client()



# FILE READING

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
    return "\n".join(para.text for para in doc.paragraphs)


def extract_text_from_txt(uploaded_file):
    return uploaded_file.read().decode("utf-8", errors="ignore")


def extract_uploaded_text(uploaded_file):
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



# BASIC TEXT PREP

def prepare_text(text):
    return text.replace("\r", "\n").replace("\t", " ").strip()



# LLM HELPERS

def run_chat(messages, max_tokens=500):
    try:
        completion = hf_client.chat_completion(
            model=HF_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def parse_json_array(text):
    try:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            arr = json.loads(text[start:end + 1])
            if isinstance(arr, list):
                return sorted(list({str(x).strip() for x in arr if str(x).strip()}))
    except Exception:
        pass
    return []



# AI EXTRACTION

def ai_identify_resume_skills(resume_text):
    messages = [
        {
            "role": "system",
            "content": (
                "You extract skills from resumes. "
                "Return only valid JSON. "
                "Output must be a JSON array of strings and nothing else."
            ),
        },
        {
            "role": "user",
            "content": f"""
Read this resume and identify the actual skills the person has.

Include only real skills such as:
- programming languages
- software
- tools
- platforms
- technical methods
- certifications
- professional skills clearly demonstrated or explicitly listed

Do not include:
- names
- schools
- locations
- dates
- GPA
- employers
- paper titles
- publication citations
- section headers
- job titles by themselves

Return only a JSON array of strings.

Resume:
{resume_text[:5000]}
""",
        },
    ]

    output = run_chat(messages, max_tokens=400)
    return parse_json_array(output)


def ai_identify_job_skills(job_text):
    messages = [
        {
            "role": "system",
            "content": (
                "You extract required skills from job descriptions. "
                "Return only valid JSON. "
                "Output must be a JSON array of strings and nothing else."
            ),
        },
        {
            "role": "user",
            "content": f"""
Read this job description and identify the actual required or preferred skills.

Include only:
- programming languages
- software
- tools
- platforms
- technical methods
- certifications
- domain knowledge
- required professional skills

Do not include:
- company descriptions
- benefits
- locations
- marketing language
- role titles by themselves
- section headers
- sentence fragments

Return only a JSON array of strings.

Job Description:
{job_text[:5000]}
""",
        },
    ]

    output = run_chat(messages, max_tokens=400)
    return parse_json_array(output)


def generate_llm_suggestions(resume_text, job_text, matched_skills, missing_skills):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a professional resume assistant. "
                "Be concise, practical, and do not invent experience."
            ),
        },
        {
            "role": "user",
            "content": f"""
Compare this resume with this job description.

Matched skills:
{", ".join(matched_skills)}

Missing skills:
{", ".join(missing_skills)}

Write exactly:
1. Three practical resume improvements
2. Two stronger bullet point rewrite ideas
3. One short professional summary tailored to the job

Resume:
{resume_text[:3000]}

Job Description:
{job_text[:3000]}
""",
        },
    ]

    output = run_chat(messages, max_tokens=500)
    if output.startswith("ERROR:"):
        return "AI suggestions are unavailable right now."
    return output



# MATCHING

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



# UI

uploaded_resume = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
job_description = st.text_area("Paste Job Description", height=300)
analyze = st.button("Analyze")



# MAIN

if analyze:
    if not HF_TOKEN:
        st.error("Missing HF_TOKEN. Add it to Streamlit secrets or environment variables.")
    elif uploaded_resume is None:
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
