# Smart Resume Analyzer & Job Recommendation System

Local-first app for parsing resumes, comparing to job descriptions, and recommending roles.

## Stack
- Backend: FastAPI
- Frontend: Streamlit
- Database: MongoDB
- NLP/ML: spaCy, scikit-learn, TF-IDF + cosine similarity

## Prerequisites
- Python 3.10+
- MongoDB running locally (or a URI you control)
- spaCy model `en_core_web_sm` (install step below)

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Configuration
- Env vars (optional):
  - `MONGODB_URI` (default `mongodb://localhost:27017`)
  - `MONGODB_DB` (default `smart_resume`)
  - `API_URL` (Streamlit uses this; default `http://localhost:8000`)

## Run MongoDB
- Local service: `sudo systemctl start mongod` (Linux) or `brew services start mongodb-community` (macOS)
- Docker: `docker run -d --name resume-mongo -p 27017:27017 mongo:6`

## Start the backend (FastAPI)
```bash
uvicorn main:app --reload
```
The app seeds `jobs` collection from `seed_jobs.json` on startup if empty.

## Start the frontend (Streamlit)
```bash
streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`).

## Usage
1) Upload a resume (PDF/DOCX, <= 2MB).
2) Paste a job description (<= 5000 chars).
3) Click **Analyze Resume** to see compatibility score, matched/missing skills, and role recommendations from seeded roles.

## Screenshots
- Upload flow:

![Screenshot from 2025-12-07 18-52-39.png](docs/screenshots/Screenshot%20from%202025-12-07%2018-52-39.png)

- Resume 1 Checking Result:

![Screenshot from 2025-12-07 19-11-04.png](docs/screenshots/Screenshot%20from%202025-12-07%2019-11-04.png)

- Resume 2 Checking Result:

![Screenshot from 2025-12-07 19-11-13.png](docs/screenshots/Screenshot%20from%202025-12-07%2019-11-13.png)

- Resume 3 Checking Result:

![Screenshot from 2025-12-07 19-11-20.png](docs/screenshots/Screenshot%20from%202025-12-07%2019-11-20.png)

## File overview
- `main.py` — FastAPI API (`/upload_resume`, `/analyze`), seeding.
- `app.py` — Streamlit UI.
- `utils.py` — parsing, TF-IDF/cosine, skill extraction, validations.
- `database.py` — Mongo connection helpers, seeding.
- `seed_jobs.json` — 10 benchmark roles.
- `requirements.txt` — dependencies.
