import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
MAX_JD_LENGTH = 5000

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide")
st.sidebar.title("Smart Resume Analyzer")
st.sidebar.write("Upload a resume and compare against a job description.")


def tag_list(items, color):
    chips = " ".join(
        [
            f"<span style='background:{color}; padding:4px 8px; border-radius:12px; color:white;'>{item}</span>"
            for item in items
        ]
    )
    st.markdown(chips, unsafe_allow_html=True)


def _error_detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:  # noqa: BLE001
        return resp.text


st.title("Resume vs Job Description")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX, max 2MB)", type=["pdf", "docx"])
with col2:
    jd_text = st.text_area("Job Description (max 5000 chars)", height=300, max_chars=MAX_JD_LENGTH)

analyze_clicked = st.button("Analyze Resume", type="primary")

if analyze_clicked:
    if not uploaded_file:
        st.error("Please upload a resume file.")
    elif uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error("File exceeds 2MB limit.")
    elif not jd_text or len(jd_text) < 10:
        st.error("Job description must be at least 10 characters.")
    else:
        with st.spinner("Parsing resume..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            resp = requests.post(f"{API_URL}/upload_resume", files=files, timeout=60)
        if resp.status_code != 200:
            st.error(_error_detail(resp) or "Failed to parse resume")
        else:
            resume_text = resp.json()["parsed"]["resume_text"]
            with st.spinner("Analyzing..."):
                payload = {"resume_text": resume_text, "job_description_text": jd_text}
                analysis = requests.post(f"{API_URL}/analyze", json=payload, timeout=60)
            if analysis.status_code != 200:
                st.error(_error_detail(analysis) or "Analysis failed")
            else:
                data = analysis.json()
                score = data["compatibility_score"]
                st.subheader("Results")
                st.metric("Compatibility Score", f"{score} %", delta=None)
                st.progress(min(score / 100, 1.0))
                st.write(f"Match Level: **{data['match_level']}**")
                st.write("Matched Skills")
                tag_list(data["matched_skills"] or ["None"], "#16a34a")
                st.write("Missing Skills")
                tag_list(data["missing_skills"] or ["None"], "#dc2626")

                st.write("Role Recommendations (top 10 seed roles)")
                for rec in data["recommendations"][:10]:
                    st.write(f"- {rec['role']}: {rec['score']}%")
