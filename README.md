# LLM-powered Social Listening System

> **📋 [View Project Timeline & Milestones on Notion](https://www.notion.so/LLM-powered-Social-Listening-System-28abe5c2ca3180e7a35defa1c99e44e8)**

# Walmart Social Media Listener

End‑to‑end pipeline for Twitter/X listening about Walmart: cleaning → sentiment → aspect‑based analysis → unsupervised themes → **LLM executive summaries** → interactive demo.

## 1) Quickstart

```bash
# clone or unzip locally, then:
python3 -m venv .venv       # Windows: python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# set env
cp .env.example .env        # Windows: copy .env.example .env
# open .env and set OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

### Run the frontend
```bash
cd frontend
npm run dev
```

### CLI summarizer
```bash
python scripts/generate_summary.py   --data data/tweets_clean.parquet   --start 2025-07-01 --end 2025-07-31 --keyword "delivery"
```

## 2) Repo layout & where to paste notebook code

```
walmart_social_listener/
├─ frontend/                       # React frontend (Vite + Tailwind CSS)
├─ data/                          # Put your CSV/XLSX/Parquet here (kept out of git)
├─ notebooks/
│  └─ Walmart_Social_Media_Listener_V1.ipynb   # Move your current ipynb here
├─ scripts/
│  ├─ run_pipeline.py             # Orchestrate data→features→topics (paste pipeline steps)
│  └─ generate_summary.py         # Batch LLM summaries from a file
├─ src/
│  ├─ features/
│  │  ├─ clean.py                 # ⬅️ Paste your cleaning/prep utilities
│  │  ├─ sentiment.py             # ⬅️ Paste sentiment functions (CardiffNLP, etc.)
│  │  ├─ absa.py                  # ⬅️ Paste aspect-based code here
│  │  └─ topics.py                # ⬅️ Paste BERTopic/topic modeling code here
│  ├─ llm/
│  │  └─ summary.py               # LLM executive summary logic (already scaffolded)
│  └─ utils/
│     └─ text.py                  # Small helpers (tokenize, normalize, etc.)
├─ .env.example                   # Environment variables template
├─ requirements.txt
└─ README.md
```

### Mapping your notebook sections → files
- **Data loading/cleaning** → `src/features/clean.py`
- **Sentiment** → `src/features/sentiment.py`
- **Aspect‑Based** → `src/features/absa.py`
- **Topics/BERTopic** → `src/features/topics.py`
- **Helper functions** → `src/utils/text.py`
- **LLM Summary** → use/extend `src/llm/summary.py`
- **Pipeline orchestration** (read→process→save) → `scripts/run_pipeline.py`

> Tip: copy code cell‑by‑cell into these files, turn notebook variables into function parameters, and return DataFrames.

## 3) Git workflow (local → GitHub)

```bash
# initialize and first commit
git init
git add .
git commit -m "init: scaffold social listener + LLM summary app"

# add remote (replace with your repo URL)
git branch -M main
git remote add origin https://github.com/<you>/walmart-social-listener.git
git push -u origin main

# day-to-day
git pull            # bring latest from remote
git checkout -b feat/llm-summary
# ... edit, run, test ...
git add -A
git commit -m "feat: add LLM executive summary + React frontend"
git push -u origin feat/llm-summary
# open a PR, review, merge
```

## 4) Deploy a live demo

**Frontend (React):**
1. Build the frontend: `cd frontend && npm run build`
2. Deploy to Netlify/Vercel/Cloudflare Pages
3. Set environment variables for API endpoints

**Backend (Node.js/Express):**
1. Deploy `server/server.js` to services like Railway, Render, or Heroku
2. Ensure Python dependencies are available for theme generation scripts
3. Set environment variables: `OPENAI_API_KEY`, data paths, etc.

## 5) Expected columns

The system expects parquet files (CSV/XLSX/Parquet) with at least:
- `clean_tweet` or `text` (str): cleaned text
- `createdat` or `created_at` or `date` (datetime/ISO or parseable string)
- Optional: `sentiment_label`, `aspect_dominant`, `theme`, etc.

## 6) OpenAI usage
We use the official SDK (`openai`), via the **Responses API** by default. See comments in `src/llm/summary.py` to switch to Chat Completions if you prefer.
