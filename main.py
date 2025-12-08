import logging
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pymongo.errors import PyMongoError
from pydantic import BaseModel, Field

from database import ensure_seed_jobs, get_collections
from utils import (
    MATCH_THRESHOLD,
    compute_similarity,
    extract_entities,
    extract_text_from_docx,
    extract_text_from_pdf,
    load_seed_jobs,
    recommend_roles,
    validate_file_size,
    validate_jd_length,
)

logger = logging.getLogger(__name__)
app = FastAPI(title="Smart Resume Analyzer", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=10)
    job_description_text: str = Field(..., min_length=10, max_length=5000)


class AnalyzeResponse(BaseModel):
    compatibility_score: float
    match_level: str
    missing_skills: List[str]
    matched_skills: List[str]
    recommendations: List[Dict[str, Any]]


def get_db():
    return get_collections()


@app.on_event("startup")
async def seed_jobs_if_needed() -> None:
    try:
        ensure_seed_jobs(load_seed_jobs())
    except Exception as exc:  # noqa: BLE001
        logger.warning("Skipping job seeding; database not reachable: %s", exc)


@app.post("/upload_resume")
async def upload_resume(file: UploadFile = File(...), db=Depends(get_db)):
    candidates, _ = db
    file_bytes = await file.read()
    try:
        validate_file_size(file_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=413, detail=str(exc))

    content_type = file.content_type or ""
    suffix = (file.filename or "").lower()

    try:
        if "pdf" in content_type or suffix.endswith(".pdf"):
            text = extract_text_from_pdf(file_bytes)
        elif "word" in content_type or suffix.endswith(".docx"):
            text = extract_text_from_docx(file_bytes)
        else:
            raise HTTPException(status_code=400, detail="Only PDF or DOCX files are supported")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {exc}")

    parsed = extract_entities(text)
    record = {
        "name": parsed.get("name"),
        "email": parsed.get("email"),
        "skills": parsed.get("skills", []),
        "resume_text": text,
    }
    try:
        inserted = candidates.insert_one(record)
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable; start MongoDB") from exc
    sanitized = dict(record)
    sanitized.pop("_id", None)
    return {"id": str(inserted.inserted_id), "parsed": sanitized, "message": "Resume parsed and stored"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest, db=Depends(get_db)):
    _candidates, jobs_col = db
    try:
        validate_jd_length(payload.job_description_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    hard_skills = set()
    try:
        jobs = list(jobs_col.find({}, {"_id": 0}))
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable; start MongoDB") from exc
    for job in jobs:
        hard_skills.update(job.get("ideal_skills", []))

    score, missing_skills, matched_skills = compute_similarity(
        payload.resume_text, payload.job_description_text, hard_skills
    )
    match_level = "High Match" if score >= MATCH_THRESHOLD else "Needs Improvement"
    recommendations = recommend_roles(payload.resume_text, jobs) if jobs else []

    return AnalyzeResponse(
        compatibility_score=round(score * 100, 2),
        match_level=match_level,
        missing_skills=missing_skills,
        matched_skills=matched_skills,
        recommendations=recommendations,
    )
