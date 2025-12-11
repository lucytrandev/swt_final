import logging
from typing import Any, Dict, List

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from pymongo.errors import PyMongoError
from pydantic import BaseModel, Field

from database import ensure_seed_jobs, get_collections
from utils import (
    MATCH_THRESHOLD,
    _common_hard_skills,
    compute_similarity,
    extract_entities,
    extract_text_from_docx,
    extract_text_from_pdf,
    load_seed_jobs,
    recommend_roles,
    validate_file_size,
    validate_jd_length,
    # V2 Imports
    find_skills_with_synonyms,
    compute_similarity_section_aware,
    extract_experience_years,
    extract_total_experience,
    calculate_experience_bonus,
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

    hard_skills = set(_common_hard_skills)
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

class AnalyzeResponseV2(BaseModel):
    compatibility_score: float = Field(..., description="Final score (0-100)")
    match_level: str = Field(..., description="High Match / Needs Improvement")
    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)
    recommendations: List[Dict[str, Any]] = Field(default_factory=list)
    
    # New fields
    total_experience_years: float = Field(default=0.0)
    skill_experience: Dict[str, float] = Field(default_factory=dict)
    sections_detected: List[str] = Field(default_factory=list)
    repetition_penalty: float = Field(default=1.0)
    score_breakdown: Dict[str, float] = Field(default_factory=dict)


@app.post("/analyze_v2", response_model=AnalyzeResponseV2)
async def analyze_v2(payload: AnalyzeRequest, db=Depends(get_db)):
    """
    Enhanced analysis with section weighting, repetition penalty, and experience extraction.
    """
    _candidates, jobs_col = db
    
    try:
        validate_jd_length(payload.job_description_text)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    
    # Build skill set from seed jobs
    hard_skills = set(_common_hard_skills)
    try:
        jobs = list(jobs_col.find({}, {"_id": 0}))
    except PyMongoError as exc:
        raise HTTPException(status_code=503, detail="Database unavailable") from exc
    
    for job in jobs:
        hard_skills.update(job.get("ideal_skills", []))
    
    # Extract skills from both texts
    resume_skills = find_skills_with_synonyms(payload.resume_text, hard_skills)
    jd_skills = find_skills_with_synonyms(payload.job_description_text, hard_skills)
    
    # Section-aware similarity calculation
    score, missing_skills, matched_skills, breakdown = compute_similarity_section_aware(
        payload.resume_text,
        payload.job_description_text,
        resume_skills,
        jd_skills
    )
    
    # Extract experience
    # Note: Using resume text for skill extraction years
    skill_years = extract_experience_years(payload.resume_text)
    total_years = extract_total_experience(payload.resume_text)
    
    # Apply experience bonus
    experience_bonus = calculate_experience_bonus(total_years, skill_years)
    final_score = max(0.0, min(score + experience_bonus, 1.0))
    
    # Determine match level
    if final_score >= 0.70:
        match_level = "High Match"
    elif final_score >= 0.50:
        match_level = "Good Match"
    elif final_score >= 0.30:
        match_level = "Partial Match"
    else:
        match_level = "Needs Improvement"
    
    # Generate recommendations
    recommendations = recommend_roles(payload.resume_text, jobs) if jobs else []
    
    # --- LOGGING FOR DEMO ---
    report_text = f"""
============================================================
           NEW RESUME ANALYSIS RECEIVED
============================================================
TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

MATCH SUMMARY
-------------
Final Compatibility Score: {round(final_score * 100, 2)}%
Match Level: {match_level}
Detected Experience: {total_years} years

SCORE BREAKDOWN (INTERNAL METRICS)
----------------------------------
Base Hybrid Score: {round(breakdown.get('hybrid_raw', 0) * 100, 2)}%

1. TF-IDF (Context):   {round(breakdown.get('tfidf', 0) * 100, 2)}%
2. BM25 (Keywords):    {round(breakdown.get('bm25', 0) * 100, 2)}%
3. Jaccard (Skills):   {round(breakdown.get('jaccard', 0) * 100, 2)}%

ADJUSTMENTS
-----------
Experience Bonus:      +{round(experience_bonus * 100, 2)}%
Repetition Penalty:    -{round((1 - breakdown.get('repetition_penalty', 1.0)) * 100, 2)}%
(Quality Score: {round(breakdown.get('repetition_penalty', 1.0) * 100, 2)}%)

SKILLS & INSIGHTS
-----------------
Matched: {", ".join(matched_skills)}
Missing: {", ".join(missing_skills)}
============================================================
"""
    print(report_text)  # Prints to terminal
    # ------------------------
    
    return AnalyzeResponseV2(
        compatibility_score=round(final_score * 100, 2),
        match_level=match_level,
        matched_skills=matched_skills,
        missing_skills=missing_skills,
        recommendations=recommendations,
        total_experience_years=total_years,
        skill_experience=skill_years,
        sections_detected=breakdown.get('sections_detected', []),
        repetition_penalty=breakdown.get('repetition_penalty', 1.0),
        score_breakdown={
            'tfidf': breakdown.get('tfidf', 0) * 100,
            'bm25': breakdown.get('bm25', 0) * 100,
            'jaccard': breakdown.get('jaccard', 0) * 100,
            'hybrid_raw': breakdown.get('hybrid_raw', 0) * 100,
            'experience_bonus': experience_bonus * 100,
            'final': final_score * 100
        }
    )
