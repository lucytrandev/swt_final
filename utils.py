import io
import json
import os
import re
from typing import Dict, List, Sequence, Set, Tuple

import spacy
from PyPDF2 import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
MAX_JD_LENGTH = 5000
MATCH_THRESHOLD = 0.7

_common_hard_skills = {
    "python",
    "java",
    "c++",
    "c#",
    "javascript",
    "typescript",
    "sql",
    "nosql",
    "mongodb",
    "postgresql",
    "mysql",
    "fastapi",
    "django",
    "flask",
    "react",
    "vue",
    "docker",
    "kubernetes",
    "aws",
    "gcp",
    "azure",
    "terraform",
    "ansible",
    "ci/cd",
    "jenkins",
    "gitlab",
    "spark",
    "pandas",
    "numpy",
    "scikit-learn",
    "tensorflow",
    "pytorch",
    "airflow",
    "kafka",
    "hadoop",
    "linux",
    "bash",
}


def _load_nlp():
    try:
        return spacy.load(SPACY_MODEL)
    except OSError as exc:
        raise RuntimeError(f"Install spaCy model: python -m spacy download {SPACY_MODEL}") from exc


nlp = _load_nlp()


def load_seed_jobs(seed_path: str = "seed_jobs.json") -> List[Dict]:
    if not os.path.exists(seed_path):
        return []
    with open(seed_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def extract_text_from_docx(file_bytes: bytes) -> str:
    document = Document(io.BytesIO(file_bytes))
    return "\n".join([para.text for para in document.paragraphs])


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_contacts(text: str) -> Dict[str, str]:
    email_match = re.search(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
    phone_match = re.search(r"\+?\d[\d\-\s]{7,}\d", text)
    return {
        "email": email_match.group(0) if email_match else "",
        "phone": phone_match.group(0) if phone_match else "",
    }


def extract_entities(text: str) -> Dict[str, object]:
    doc = nlp(text)
    names = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    skills = find_skills(text, _common_hard_skills)
    contacts = extract_contacts(text)
    return {
        "name": names[0] if names else "",
        "email": contacts["email"],
        "skills": sorted(skills),
        "resume_text": text,
    }


def find_skills(text: str, known_skills: Set[str]) -> Set[str]:
    lowered = clean_text(text)
    found = set()
    for skill in known_skills:
        if skill in lowered:
            found.add(skill)
    return found


def _weighted_text(text: str, skills: Set[str]) -> str:
    if not skills:
        return text
    boost = " ".join(skills)
    return f"{text} {boost} {boost}"


def compute_similarity(
    resume_text: str, jd_text: str, hard_skills: Set[str]
) -> Tuple[float, List[str], List[str]]:
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    resume_skills = find_skills(resume_text, hard_skills)
    jd_skills = find_skills(jd_text, hard_skills)

    resume_weighted = _weighted_text(resume_text, resume_skills)
    jd_weighted = _weighted_text(jd_text, jd_skills)

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform([resume_weighted, jd_weighted])
    score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])

    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)
    return score, missing_skills, matched_skills


def recommend_roles(resume_text: str, jobs: Sequence[Dict]) -> List[Dict[str, object]]:
    results = []
    for job in jobs:
        skills = set(job.get("ideal_skills", []))
        jd_text = " ".join(skills)
        score, _, _ = compute_similarity(resume_text, jd_text, skills or _common_hard_skills)
        results.append({"role": job.get("role"), "score": round(score * 100, 2)})
    return sorted(results, key=lambda x: x["score"], reverse=True)


def validate_file_size(file_bytes: bytes) -> None:
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File exceeds 2MB limit")


def validate_jd_length(text: str) -> None:
    if len(text) > MAX_JD_LENGTH:
        raise ValueError("Job description exceeds 5000 characters")
