import os
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("API_URL", "http://localhost:8000")
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
MAX_JD_LENGTH = 5000

st.set_page_config(page_title="Smart Resume Analyzer", layout="wide", page_icon="📄")

st.sidebar.title("Smart Resume Analyzer")
st.sidebar.info("Upload a resume and compare against a job description.")

def tag_list(items, color):
    if not items:
        return
    chips = " ".join(
        [
            f"<span style='background:{color}; padding:4px 8px; border-radius:12px; color:white; display:inline-block; margin:2px;'>{item}</span>"
            for item in items
        ]
    )
    st.markdown(chips, unsafe_allow_html=True)

def _error_detail(resp: requests.Response) -> str:
    try:
        return resp.json().get("detail", resp.text)
    except Exception:
        return resp.text

st.title("🚀 Resume Analyzer and Recommender")

col1, col2 = st.columns(2)
with col1:
    uploaded_file = st.file_uploader("Upload Resume (PDF/DOCX)", type=["pdf", "docx"])
with col2:
    jd_text = st.text_area("Job Description", height=300, max_chars=MAX_JD_LENGTH)

analyze_clicked = st.button("Analyze Resume", type="primary", use_container_width=True)

if analyze_clicked:
    if not uploaded_file:
        st.error("Please upload a resume file.")
    elif uploaded_file.size > MAX_FILE_SIZE_BYTES:
        st.error(f"File exceeds 2MB limit (Size: {uploaded_file.size/1024/1024:.2f} MB).")
    elif not jd_text or len(jd_text) < 10:
        st.error("Job description must be at least 10 characters.")
    else:
        # 1. Upload/Parse
        with st.spinner("Parsing resume..."):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            try:
                resp = requests.post(f"{API_URL}/upload_resume", files=files, timeout=60)
            except requests.exceptions.ConnectionError:
                st.error("Backend API is unreachable. Is uvicorn running?")
                st.stop()
                
        if resp.status_code != 200:
            st.error(_error_detail(resp))
        else:
            resume_data = resp.json().get("parsed", {})
            resume_text = resume_data.get("resume_text", "")
            
            # 2. Analyze
            with st.spinner("Analyzing match..."):
                payload = {"resume_text": resume_text, "job_description_text": jd_text}
                # Use V2 endpoint
                analysis = requests.post(f"{API_URL}/analyze_v2", json=payload, timeout=60)
            
            if analysis.status_code != 200:
                st.error(_error_detail(analysis))
            else:
                data = analysis.json()
                score = data["compatibility_score"]
                
                st.divider()
                st.header("Analysis Results")
                
                # Main Stats Row
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Compatibility Score", f"{score:.1f}%")
                    st.progress(min(score / 100, 1.0))
                
                with col_b:
                    st.metric("Match Level", data["match_level"])
                    exp_years = data.get("total_experience_years", 0)
                    if exp_years > 0:
                        st.caption(f"Detected Experience: **{exp_years:.1f} years**")

                # Skills Display
                st.subheader("Skills Analysis")
                s_col1, s_col2 = st.columns(2)
                
                with s_col1:
                    st.write("**✅ Matched Skills**")
                    if data["matched_skills"]:
                        tag_list(data["matched_skills"], "#16a34a")
                    else:
                        st.info("No direct skill matches found.")
                
                with s_col2:
                    st.write("**⚠️ Missing Skills**")
                    if data["missing_skills"]:
                        tag_list(data["missing_skills"], "#dc2626")
                    else:
                        st.success("You have all the required skills!")
                
                # Recommendations
                if data.get("recommendations"):
                    st.divider()
                    st.subheader("Role Recommendations (Top 3)")
                    for rec in data["recommendations"][:3]:
                        st.markdown(f"#### 🎯 {rec['role']}: **{rec['score']:.1f}%**")
                
                st.divider()
                
                # Experience Section
                st.subheader("Experience Detection")
                
                skill_years = data.get("skill_experience", {})
                if skill_years:
                    # Create readable table for skill years
                    exp_data = [{"Skill": k.title(), "Years": v} for k, v in skill_years.items() if v > 0]
                    if exp_data:
                        exp_df = pd.DataFrame(exp_data).sort_values("Years", ascending=False)
                        # Bulletproof method: set index to Skill so we don't have a 0,1,2 column
                        exp_df.set_index("Skill", inplace=True)
                        st.table(exp_df.style.format({"Years": "{:.1f} yrs"}))
                    else:
                        st.caption("No specific years of experience detected for skills.")
