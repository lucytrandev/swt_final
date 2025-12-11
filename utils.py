import io
import json
import os
import re
from typing import Dict, List, Sequence, Set, Tuple
from collections import Counter
from datetime import datetime

import spacy
import pdfplumber
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi

SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
MAX_FILE_SIZE_BYTES = 2 * 1024 * 1024
MAX_JD_LENGTH = 5000
MATCH_THRESHOLD = 0.5

SKILL_SYNONYMS = {
    # Programming Languages
    "javascript": ["js", "javascript", "ecmascript"],
    "typescript": ["ts", "typescript"],
    "python": ["python", "py"],
    "c++": ["cpp", "c++", "cplusplus"],
    "c#": ["csharp", "c#"],
    
    # Frameworks
    "react": ["react", "reactjs", "react.js"],
    "vue": ["vue", "vuejs", "vue.js"],
    "node.js": ["nodejs", "node.js", "node"],
    
    # ML/AI
    "machine learning": ["ml", "machine learning", "machinelearning"],
    "deep learning": ["dl", "deep learning", "deeplearning"],
    "artificial intelligence": ["ai", "artificial intelligence"],
    "natural language processing": ["nlp", "natural language processing"],
    "computer vision": ["cv", "computer vision"],
    
    # Data Science
    "scikit-learn": ["sklearn", "scikit-learn", "scikit learn"],
    "tensorflow": ["tf", "tensorflow"],
    "pytorch": ["torch", "pytorch"],
    
    # Databases
    "postgresql": ["postgres", "postgresql", "psql"],
    "mongodb": ["mongo", "mongodb"],
    
    # Cloud
    "amazon web services": ["aws", "amazon web services"],
    "google cloud platform": ["gcp", "google cloud platform", "google cloud"],
    
    # DevOps
    "continuous integration": ["ci", "continuous integration", "ci/cd"],
    "continuous deployment": ["cd", "continuous deployment"],
    
    # Methodologies
    "object oriented programming": ["oop", "object oriented programming"],
    "test driven development": ["tdd", "test driven development"],

    # Data Analytics
    "data analytics": ["data analyst", "data analytics"],
}

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
    "communication",
    "teamwork",
    "problem solving",
    "critical thinking",
    "adaptability",
    "time management",
    "leadership",
    "creativity",
    "attention to detail",
    "emotional intelligence",
    "decision making",
    "project management",
    "organization",
    "self-motivation",
    "accountability",
    "reliability",
    "collaboration",
    "stakeholder management",
    "agile",
    "scrum",
    "kanban",
    "customer service",
    "presentation skills",
    "writing skills",
    "public speaking",
    "analytical thinking",
    "figma",
    "excel",
    "notion",
    "jira",
    "trello",
    "visio",
    "power bi",
    "matplotlib",
    "seaborn",
    # Extended list to support experience extraction for roles
    "data science",
    "data scientist",
    "data analyst",
    "data engineering",
    "data engineer",
    "software engineering",
    "software engineer",
    "web development",
    "machine learning",
    "artificial intelligence",
    "deep learning",
    "cloud computing",
    "devops",
    "project management",
    "product management",
    "business intelligence",
    "quality assurance",
    "testing",
    "maintenance", # careful, but often a skill "System Maintenance"
    "deployment",
    "security",
    "network engineering",
    "mobile development",
    # Telecommunications Specific
    "telecommunications",
    "network infrastructure",
    "capacity planning",
    "performance optimization",
    "network engineering",
    "5g",
    "lte",
    "voip",
    "manager",
}


_SEED_CORPUS_TOKENS = []

def _get_expanded_corpus(resume_tokens: List[str]) -> List[List[str]]:
    """
    Returns a corpus containing the resume + seed jobs.
    This provides a background frequency distribution for BM25.
    """
    global _SEED_CORPUS_TOKENS
    
    if not _SEED_CORPUS_TOKENS:
        try:
            jobs = load_seed_jobs()
            for job in jobs:
                # Combine role + description + skills for a rich document
                text = f"{job.get('role', '')} {job.get('description', '')} {' '.join(job.get('ideal_skills', []))}"
                tokens = clean_text(text).split()
                if tokens:
                    _SEED_CORPUS_TOKENS.append(tokens)
        except Exception:
            pass
            
        # Fallback if no seed jobs found (add some dummy generic business/tech words)
        if not _SEED_CORPUS_TOKENS:
             _SEED_CORPUS_TOKENS = [
                 ["manager", "business", "communication", "leadership"],
                 ["developer", "software", "code", "engineering"],
                 ["analyst", "data", "report", "excel"]
             ] * 5

    return [resume_tokens] + _SEED_CORPUS_TOKENS

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
    """
    Extracts text from PDF using pdfplumber (better layout handling).
    Also extracts tables if present.
    """
    text_parts = []
    
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # Extract text
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            
            # Extract tables (many resumes have skills in tables)
            tables = page.extract_tables()
            for table in tables:
                for row in table:
                    if row:
                        text_parts.append(" ".join([str(cell) for cell in row if cell]))
    
    return "\n".join(text_parts)


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
    skills = find_skills_with_synonyms(text, _common_hard_skills)
    contacts = extract_contacts(text)
    return {
        "name": names[0] if names else "",
        "email": contacts["email"],
        "skills": sorted(skills),
        "resume_text": text,
    }




def normalize_skill(skill: str) -> str:
    """
    Normalizes a skill to its canonical form.
    Example: 'ML' -> 'machine learning'
    """
    skill_lower = skill.lower().strip()
    
    # Check if it's already a canonical form
    if skill_lower in SKILL_SYNONYMS:
        return skill_lower
    
    # Check if it's a synonym
    for canonical, synonyms in SKILL_SYNONYMS.items():
        if skill_lower in synonyms:
            return canonical
    
    return skill_lower


def find_skills_with_synonyms(text: str, known_skills: Set[str]) -> Set[str]:
    """
    Enhanced skill extraction with synonym support using robust regex boundaries.
    Prevents false positives like 'ai' in 'maintenance' or 'ci' in 'special'.
    """
    lowered = clean_text(text)
    found = set()
    
    # 1. Check known skills
    for skill in known_skills:
        # Create pattern: (start or non-alphanum) + skill + (end or non-alphanum)
        # This handles "c++" or "node.js" correctly unlike \b
        pattern = r'(?:^|[^a-z0-9])' + re.escape(normalize_skill(skill)) + r'(?:$|[^a-z0-9])'
        if re.search(pattern, lowered):
            found.add(normalize_skill(skill))
            
    # 2. Check synonyms
    for canonical, synonyms in SKILL_SYNONYMS.items():
        for synonym in synonyms:
             pattern = r'(?:^|[^a-z0-9])' + re.escape(synonym) + r'(?:$|[^a-z0-9])'
             if re.search(pattern, lowered):
                found.add(canonical)
                break
                
    return found


def compute_similarity(
    resume_text: str, jd_text: str, hard_skills: Set[str]
) -> Tuple[float, List[str], List[str]]:
    """
    Improved similarity calculation with proper skill weighting using Vector Boosting.
    """
    resume_text = clean_text(resume_text)
    jd_text = clean_text(jd_text)

    # Extract skills
    resume_skills = find_skills_with_synonyms(resume_text, hard_skills)
    jd_skills = find_skills_with_synonyms(jd_text, hard_skills)
    
    matched_skills = sorted(resume_skills & jd_skills)
    missing_skills = sorted(jd_skills - resume_skills)

    # Create custom vectorizer with skill boosting
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),  # Include bi-grams (e.g., "machine learning")
        max_features=500,     # Limit vocabulary size
        min_df=1,             # Allow rare terms
        sublinear_tf=True     # Use log scaling for term frequency
    )
    
    # Fit and transform
    try:
        vectors = vectorizer.fit_transform([resume_text, jd_text])
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # BOOST SKILL TERMS in the vector representation
        for i, feature in enumerate(feature_names):
            normalized = normalize_skill(feature)
            # Boost if the feature maps to a known hard skill
            if normalized in hard_skills:
                # Multiply TF-IDF score by 2.0 (boost factor)
                vectors[0, i] *= 2.0  # Resume
                vectors[1, i] *= 2.0  # JD
        
        # Calculate similarity
        score = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
    except ValueError:
        score = 0.0

    return score, missing_skills, matched_skills


def recommend_roles(resume_text: str, jobs: Sequence[Dict]) -> List[Dict[str, object]]:
    results = []
    
    # 1. Pre-calculate resume skills
    resume_skills = find_skills_with_synonyms(resume_text, _common_hard_skills)
    
    # 2. Pre-process resume sections (identical to analyze_v2)
    # This ensures "Skills" and "Experience" sections get higher priority
    sections = parse_resume_sections(resume_text)
    weighted_resume = apply_section_weights(sections)
    
    for job in jobs:
        jd_skills = set(job.get("ideal_skills", []))
        jd_text = f"{job.get('role', '')} {job.get('description', '')} {' '.join(jd_skills)}"
        
        # Use V2 Hybrid Scoring with WEIGHTED resume
        score, _, _, _ = compute_hybrid_similarity_v2(
            weighted_resume, 
            jd_text, 
            resume_skills, 
            jd_skills
        )
        
        # Filter: Only recommend if matches at least 10% (user requested 10% threshold)
        if score >= 0.10:
            results.append({"role": job.get("role"), "score": round(score * 100, 2)})
            
    return sorted(results, key=lambda x: x["score"], reverse=True)


def validate_file_size(file_bytes: bytes) -> None:
    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise ValueError("File exceeds 2MB limit")


def validate_jd_length(text: str) -> None:
    if len(text) > MAX_JD_LENGTH:
        raise ValueError("Job description exceeds 5000 characters")

def calculate_repetition_penalty(text: str, top_n: int = 10) -> float:
    """
    Detects unnatural keyword repetition and returns a penalty factor.
    Returns: float penalty factor from 0.5 (high penalty) to 1.0 (no penalty)
    """
    words = clean_text(text).split()
    
    if len(words) < 30:
        return 1.0  # Too short to judge fairly
    
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 
        'for', 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are',
        'have', 'has', 'had', 'be', 'been', 'being', 'do', 'does', 'did',
        'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can'
    }
    
    word_counts = Counter(words)
    
    common_words = [
        (word, count) for word, count in word_counts.most_common(top_n * 2)
        if word not in stop_words and len(word) > 3
    ][:top_n]
    
    if not common_words:
        return 1.0
    
    total_words = len(words)
    repetition_scores = []
    
    for word, count in common_words:
        frequency = count / total_words
        if frequency > 0.05:  # >5% of text
            repetition_scores.append(frequency * 3.0)
        elif frequency > 0.03:  # 3-5%
            repetition_scores.append(frequency * 1.5)
        elif frequency > 0.02:  # 2-3%
            repetition_scores.append(frequency * 0.5)
    
    if not repetition_scores:
        return 1.0
    
    avg_repetition = sum(repetition_scores) / len(repetition_scores)
    penalty = max(0.5, 1.0 - (avg_repetition * 4))
    
    return round(penalty, 3)


def compute_hybrid_similarity_v2(
    resume_text: str,
    jd_text: str,
    resume_skills: Set[str],
    jd_skills: Set[str]
) -> Tuple[float, List[str], List[str], Dict[str, float]]:
    """
    Revised hybrid similarity with optimized weights and repetition penalty.
    Weights: TF-IDF (30%), BM25 (45%), Jaccard (25%)
    """
    resume_clean = clean_text(resume_text)
    jd_clean = clean_text(jd_text)
    
    # 1. TF-IDF (30%)
    try:
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 1),
            max_features=300,
            min_df=1
        )
        skill_boost_resume = " ".join(resume_skills) if resume_skills else ""
        skill_boost_jd = " ".join(jd_skills) if jd_skills else ""
        
        resume_boosted = f"{resume_clean} {skill_boost_resume} {skill_boost_resume}"
        jd_boosted = f"{jd_clean} {skill_boost_jd} {skill_boost_jd}"
        
        vectors = vectorizer.fit_transform([resume_boosted, jd_boosted])
        # Check if we have valid vectors (2 documents)
        if vectors.shape[0] >= 2:
             raw_similarity = float(cosine_similarity(vectors[0:1], vectors[1:2])[0][0])
             tfidf_score = max(0.0, min(raw_similarity, 1.0))
        else:
             tfidf_score = 0.0
    except Exception as e:
        print(f"TF-IDF error: {e}")
        tfidf_score = 0.0
    
    # 2. BM25 (45%)
    try:
        resume_tokens = resume_clean.split()
        jd_tokens = jd_clean.split()
        
        if resume_tokens and jd_tokens:
            # Fix: Use expanded corpus to avoid negative IDF
            corpus = _get_expanded_corpus(resume_tokens)
            bm25 = BM25Okapi(corpus)
            bm25_scores = bm25.get_scores(jd_tokens)
            # Normalize BM25 dynamically by calculating the max possible score (JD against itself)
            # We add the JD to the corpus temporarily to find its "perfect match" score
            bm25_ideal = BM25Okapi(corpus + [jd_tokens])
            scores_ideal = bm25_ideal.get_scores(jd_tokens)
            max_score = scores_ideal[-1] # The last doc is the JD itself
            
            bm25_raw = bm25_scores[0]
            if max_score > 0:
                bm25_score = max(0.0, min(bm25_raw / max_score, 1.0))
            else:
                bm25_score = 0.0
        else:
            bm25_score = 0.0
    except Exception as e:
        print(f"BM25 error: {e}")
        bm25_score = 0.0
    
    # 3. Jaccard (25%)
    if resume_skills and jd_skills:
        intersection = len(resume_skills & jd_skills)
        union = len(resume_skills | jd_skills)
        jaccard_score = intersection / union if union > 0 else 0.0
    else:
        jaccard_score = 0.0
    
    # Hybrid Score
    hybrid_score = (
        (0.30 * tfidf_score) + 
        (0.45 * bm25_score) + 
        (0.25 * jaccard_score)
    )
    
    # Repetition Penalty
    repetition_penalty = calculate_repetition_penalty(resume_text)
    final_score = hybrid_score * repetition_penalty
    
    matched_skills = sorted(list(resume_skills & jd_skills))
    missing_skills = sorted(list(jd_skills - resume_skills))
    
    breakdown = {
        'tfidf': round(tfidf_score, 3),
        'bm25': round(bm25_score, 3),
        'jaccard': round(jaccard_score, 3),
        'hybrid_raw': round(hybrid_score, 3),
        'repetition_penalty': round(repetition_penalty, 3),
        'final': round(final_score, 3)
    }
    
    return final_score, missing_skills, matched_skills, breakdown


SECTION_PATTERNS = {
    'skills': [
        r'(?i)^(technical\s+)?skills?$',
        r'(?i)^core\s+competenc(y|ies)$',
        r'(?i)^expertise$',
        r'(?i)^technical\s+proficiency$',
    ],
    'experience': [
        r'(?i)^(work\s+|professional\s+)?experience$',
        r'(?i)^employment(\s+history)?$',
        r'(?i)^work\s+history$',
    ],
    'education': [
        r'(?i)^education(al\s+background)?$',
        r'(?i)^academic\s+(background|qualifications?)$',
        r'(?i)^degrees?$',
    ],
    'projects': [
        r'(?i)^projects?$',
        r'(?i)^portfolio$',
        r'(?i)^selected\s+projects?$',
    ],
    'summary': [
        r'(?i)^(professional\s+)?summary$',
        r'(?i)^profile$',
        r'(?i)^(career\s+)?objective$',
        r'(?i)^about(\s+me)?$',
    ],
    'certifications': [
        r'(?i)^certifications?$',
        r'(?i)^licenses?$',
        r'(?i)^credentials?$',
    ]
}

def parse_resume_sections(text: str) -> Dict[str, str]:
    """Parses resume into sections using pattern matching."""
    sections = {'other': []}
    lines = text.split('\n')
    current_section = 'other'
    current_content = []
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
            
        section_detected = None
        for section_name, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, line_stripped):
                    if current_content:
                        if current_section not in sections: sections[current_section] = []
                        sections[current_section].extend(current_content)
                    current_section = section_name
                    current_content = []
                    section_detected = True
                    break
            if section_detected: break
        
        if not section_detected:
            current_content.append(line_stripped)
            
    if current_content:
        if current_section not in sections: sections[current_section] = []
        sections[current_section].extend(current_content)
        
    return {k: '\n'.join(v) for k, v in sections.items() if v}


def apply_section_weights(sections: Dict[str, str]) -> str:
    """Creates weighted text where important sections are repeated."""
    WEIGHTS = {
        'skills': 3.0, 'experience': 2.0, 'projects': 2.0,
        'certifications': 1.5, 'summary': 1.5, 'education': 1.0, 'other': 0.5
    }
    weighted_parts = []
    for section_name, content in sections.items():
        weight = WEIGHTS.get(section_name, 0.5)
        repetitions = int(weight)
        weighted_parts.extend([content] * max(1, repetitions))
    return '\n'.join(weighted_parts)


def compute_similarity_section_aware(
    resume_text: str, jd_text: str, resume_skills: Set[str], jd_skills: Set[str]
) -> Tuple[float, List[str], List[str], Dict[str, float]]:
    sections = parse_resume_sections(resume_text)
    weighted_resume = apply_section_weights(sections)
    score, missing, matched, breakdown = compute_hybrid_similarity_v2(
        weighted_resume, jd_text, resume_skills, jd_skills
    )
    breakdown['sections_detected'] = list(sections.keys())
    return score, missing, matched, breakdown


def extract_experience_years(text: str) -> Dict[str, float]:
    """Extracts years of experience per skill."""
    skill_years = {}
    text_lower = text.lower()
    patterns = [
        r'(\d+)\+?\s*(?:year|yr)s?\s+(?:of\s+)?(?:professional\s+)?(?:domain\s+)?(?:expertise|experience)?\s+(?:in\s+|with\s+)?((?:\w+(?:\s+)?){1,3})',
        r'((?:\w+(?:\s+)?){1,3})\s*\((\d+)\+?\s*(?:year|yr)s?\)',
        r'((?:\w+(?:\s+)?){1,3})\s*[:-]\s*(\d+)\+?\s*(?:year|yr)s?',
        r'(?:experienced\s+in\s+|worked\s+with\s+)((?:\w+(?:\s+)?){1,3})\s+(?:for\s+)?(\d+)\+?\s*(?:year|yr)s?',
        r'((?:\w+(?:\s+)?){1,3})\s+(?:professional|specialist|expert|analyst|developer|engineer|manager|consultant)\s+(?:with\s+)?(?:over\s+)?(\d+)\+?\s*(?:year|yr)s?',
        r'((?:\w+(?:\s+)?){1,3})\s+with\s+(?:over\s+)?(\d+)\+?\s*(?:year|yr)s?',
    ]
    for pattern in patterns:
        matches = re.finditer(pattern, text_lower)
        for match in matches:
            groups = match.groups()
            if len(groups) == 2:
                if groups[0].isdigit():
                    years_str, skill = groups[0], groups[1]
                else:
                    skill, years_str = groups[0], groups[1]
                
                skill = skill.strip()
                
                valid_skill_found = None
                tokens = skill.split()
                
                # Check from longest to shortest 
                for i in range(len(tokens)):
                    sub_skill = " ".join(tokens[i:])
                    
                    # 1. Check strict list
                    if sub_skill in _common_hard_skills:
                        valid_skill_found = normalize_skill(sub_skill)
                        break
                    
                    # 2. Check synonyms
                    for canonical, synonyms in SKILL_SYNONYMS.items():
                        if sub_skill in synonyms:
                            valid_skill_found = canonical
                            break
                    
                    if valid_skill_found:
                        break
                
                if not valid_skill_found:
                    continue

                try:
                    years = float(years_str)
                    if valid_skill_found in skill_years:
                        skill_years[valid_skill_found] = max(skill_years[valid_skill_found], years)
                    else:
                        skill_years[valid_skill_found] = years
                except ValueError: continue
    return skill_years


def extract_total_experience(text: str) -> float:
    """Extracts total professional experience in years."""
    text_lower = text.lower()
    # Method 1: Direct mention
    patterns = [
        # Standard "5 years of experience"
        r'(\d+)\+?\s*(?:year|yr)s?\s+of\s+(?:professional\s+)?(?:domain\s+)?(?:expertise|experience)',
        # "Total experience: 5 years"
        r'(?:total\s+)?experience:\s*(\d+)\+?\s*(?:year|yr)s?',
        # "Role with 5 years..."
        r'with\s+(\d+)\+?\s*(?:year|yr)s?\s+of\s+(?:professional\s+)?(?:domain\s+)?(?:expertise|experience)',
    ]
    for pattern in patterns:
        match = re.search(pattern, text_lower)
        if match:
            try:
                return float(match.group(1))
            except ValueError: continue
            
    # Method 2: Date ranges
    year_pattern = r'\b(19|20)\d{2}\b'
    years = re.findall(year_pattern, text)
    if len(years) >= 2:
        try:
            years_int = sorted([int(f"{y[0]}{y[1:]}") if len(y) == 2 else int(y) for y in years])
            earliest = years_int[0]
            latest = years_int[-1]
            current_year = datetime.now().year
            if earliest >= 1990 and latest <= current_year:
                return float(latest - earliest)
        except ValueError: pass
        
    # Method 3: Fallback to max skill years
    skill_years = extract_experience_years(text)
    if skill_years:
        return max(skill_years.values())
        
    return 0.0


def calculate_experience_bonus(total_years: float, skill_years: Dict[str, float]) -> float:
    if total_years >= 10: return 0.08
    elif total_years >= 6: return 0.05
    elif total_years >= 3: return 0.03
    return 0.0
