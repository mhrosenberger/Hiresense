import re
import time

import docx
import pdfplumber
import streamlit as st


# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="HireSense", layout="wide")

st.title("HireSense")
st.subheader("AI Resume & Job Description Skill Matcher")

st.write(
    "Upload your resume and paste a job description to extract skill phrases "
    "from both documents and compare them."
)

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
        elif filename.endswith(".docx"):
            return extract_text_from_docx(uploaded_file)
        elif filename.endswith(".txt"):
            return extract_text_from_txt(uploaded_file)
        else:
            return ""
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


# -----------------------------
# Cleaning
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = text.replace("\t", " ")
    text = re.sub(r"[•▪◦]", "\n", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_skill(skill):
    skill = skill.lower().strip()
    skill = re.sub(r"^[\-\•\*\d\.\)\(]+", "", skill)
    skill = re.sub(r"[^a-zA-Z0-9\+\#\/\-\.\s]", "", skill)
    skill = re.sub(r"\s+", " ", skill).strip()

    stop_phrases = {
        "", "and", "or", "with", "of", "in", "to", "for", "the", "a", "an",
        "preferred", "required", "plus", "etc", "including"
    }

    if skill in stop_phrases:
        return ""

    if len(skill) < 2:
        return ""

    return skill


# -----------------------------
# Dynamic Skill Extraction
# -----------------------------
def split_skill_candidates(text):
    pieces = []

    # split by lines first
    lines = text.split("\n")
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # split on commas / semicolons
        comma_parts = re.split(r"[,;]", line)
        for part in comma_parts:
            part = part.strip()
            if not part:
                continue

            # split on slash only if it looks like grouped skills
            slash_parts = re.split(r"\s*/\s*", part)
            for s in slash_parts:
                s = s.strip()
                if s:
                    pieces.append(s)

    return pieces


def extract_skills_from_text(text):
    original_text = text
    text = text.lower()

    skills = set()

    # patterns that often introduce skills
    patterns = [
        r"(?:skills|requirements|qualifications|experience with|proficient in|proficiency in|knowledge of|familiarity with|expertise in|tools|technologies|stack)\s*:\s*([^\n]+)",
        r"(?:experience with|proficient in|knowledge of|familiarity with|expertise in)\s+([^\.;\n]+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, original_text, flags=re.IGNORECASE)
        for match in matches:
            for piece in split_skill_candidates(match):
                cleaned = normalize_skill(piece)
                if cleaned:
                    skills.add(cleaned)

    # extract from bullet-like or list-like lines
    for line in original_text.split("\n"):
        line_clean = line.strip()
        if not line_clean:
            continue

        # likely skill lines
        if (
            "," in line_clean
            or "/" in line_clean
            or line_clean.startswith("-")
            or line_clean.startswith("•")
            or line_clean.startswith("*")
        ):
            for piece in split_skill_candidates(line_clean):
                cleaned = normalize_skill(piece)
                if cleaned:
                    # keep short phrase-sized items
                    if len(cleaned.split()) <= 4:
                        skills.add(cleaned)

    # extract capitalized / technical token patterns directly from text
    tech_patterns = [
        r"\b[a-zA-Z]+\+\+\b",
        r"\bc#\b",
        r"\b[a-zA-Z]+\.[a-zA-Z]+\b",
        r"\b(?:aws|azure|gcp|sql|python|java|javascript|typescript|react|node\.js|pytorch|tensorflow|tableau|excel|power bi|docker|kubernetes|linux|git)\b",
    ]

    lowered = original_text.lower()
    for pattern in tech_patterns:
        matches = re.findall(pattern, lowered, flags=re.IGNORECASE)
        for match in matches:
            cleaned = normalize_skill(match)
            if cleaned:
                skills.add(cleaned)

    # remove phrases that are too long to be skills
    final_skills = set()
    for skill in skills:
        word_count = len(skill.split())
        if 1 <= word_count <= 4:
            final_skills.add(skill)

    return sorted(final_skills)


def compare_skills(resume_text, job_text):
    resume_skills = set(extract_skills_from_text(resume_text))
    job_skills = set(extract_skills_from_text(job_text))

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


# -----------------------------
# Suggestions
# -----------------------------
def generate_skill_suggestions(missing_skills):
    suggestions = []

    if missing_skills:
        suggestions.append(
            "Add these job-required skills only if you genuinely have experience with them: "
            + ", ".join(missing_skills[:10]) + "."
        )
        suggestions.append(
            "Use the same wording from the job description when describing relevant experience."
        )
        suggestions.append(
            "Add missing skills into your summary, projects, and bullet points where appropriate."
        )
    else:
        suggestions.append("Your resume appears to cover the extracted job skills well.")
        suggestions.append("Focus on improving bullet points with stronger outcomes and metrics.")

    return suggestions


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
            "Add bullet points that begin with action verbs and include tools, results, or measurable outcomes."
        )

    return rewrites


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Instructions")
st.sidebar.write("1. Upload your resume as PDF, DOCX, or TXT.")
st.sidebar.write("2. Paste the job description.")
st.sidebar.write("3. Click Analyze.")
st.sidebar.write("4. Review extracted skills and compare them.")

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

analyze_button = st.button("Analyze Skills Match")

# -----------------------------
# Main Logic
# -----------------------------
if analyze_button:
    if uploaded_resume is None:
        st.warning("Please upload a resume.")
    elif not job_description.strip():
        st.warning("Please paste a job description.")
    else:
        with st.spinner("Analyzing skills..."):
            start_time = time.time()

            resume_text = extract_resume_text(uploaded_resume)
            if not resume_text.strip():
                st.error("Could not extract text from the uploaded resume.")
                st.stop()

            results = compare_skills(resume_text, job_description)
            suggestions = generate_skill_suggestions(results["missing_skills"])
            bullet_rewrites = simple_bullet_rewrites(resume_text, results["missing_skills"])

            end_time = time.time()
            latency = round(end_time - start_time, 2)

        st.success("Skill analysis complete.")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Skill Match %", f'{results["skill_match_percent"]}%')
            st.metric("Inference Time", f"{latency} sec")

        with col2:
            st.metric("Matched Skills", len(results["matched_skills"]))
            st.metric("Missing Skills", len(results["missing_skills"]))

        st.markdown("## Skills Extracted from Job Description")
        st.write(", ".join(results["job_skills"]) if results["job_skills"] else "No job skills found.")

        st.markdown("## Skills Extracted from Resume")
        st.write(", ".join(results["resume_skills"]) if results["resume_skills"] else "No resume skills found.")

        st.markdown("## Matched Skills")
        st.write(", ".join(results["matched_skills"]) if results["matched_skills"] else "No matched skills found.")

        st.markdown("## Missing Skills")
        st.write(", ".join(results["missing_skills"]) if results["missing_skills"] else "No missing skills found.")

        st.markdown("## Extra Skills on Resume")
        st.write(", ".join(results["extra_resume_skills"]) if results["extra_resume_skills"] else "No extra resume skills found.")

        st.markdown("## Suggestions")
        for suggestion in suggestions:
            st.write(f"- {suggestion}")

        st.markdown("## Bullet Point Rewrite Ideas")
        for rewrite in bullet_rewrites:
            st.write(f"- {rewrite}")

        with st.expander("View Extracted Resume Text"):
            st.text(resume_text[:5000])

        with st.expander("View Job Description"):
            st.text(job_description[:5000])
