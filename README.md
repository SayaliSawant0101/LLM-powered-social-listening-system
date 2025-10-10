
# Walmart Social Media Listener

End‑to‑end pipeline for Twitter/X listening about Walmart: cleaning → sentiment → aspect‑based analysis → unsupervised themes → **LLM executive summaries** → interactive demo.

## 1) Quickstart

```bash
# clone or unzip locally, then:
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt

# set env
cp .env.example .env
# open .env and set OPENAI_API_KEY=sk-...
```

### Run the demo app
```bash
streamlit run app/streamlit_app.py
```

### CLI summarizer
```bash
python scripts/generate_summary.py   --data data/tweets_clean.parquet   --start 2025-07-01 --end 2025-07-31 --keyword "delivery"
```

## 2) Repo layout & where to paste notebook code

```
walmart_social_listener/
├─ app/
│  └─ streamlit_app.py            # Live demo UI (date range + keyword → LLM summary)
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
git commit -m "feat: add LLM executive summary + streamlit app"
git push -u origin feat/llm-summary
# open a PR, review, merge
```

## 4) Deploy a live demo

**Streamlit Community Cloud (fastest):**
1. Push repo to GitHub (public or private with access).
2. Go to [Streamlit Cloud], "New app" → select this repo, pick `app/streamlit_app.py`.
3. In **Advanced settings → Secrets**, add `OPENAI_API_KEY`.
4. Deploy. Share the URL.

**Hugging Face Spaces:**
1. Create a new Space → Framework: Streamlit.
2. Connect your GitHub or upload files (ensure `requirements.txt`).
3. Add secret `OPENAI_API_KEY`.
4. Spaces builds and serves the app.

> Note: Netlify is for static sites; for Python apps use Streamlit Cloud or HF Spaces and link to it from your portfolio.

## 5) Expected columns

`streamlit_app.py` expects a file (CSV/XLSX/Parquet) with at least:
- `clean_tweet` (str): cleaned text
- `created_at` or `date` (datetime/ISO or parseable string)
- Optional: `sentiment_label`, `aspect`, `topic_id`, `topic_keywords`

You can adapt column names in `app/streamlit_app.py`.

## 6) OpenAI usage
We use the official SDK (`openai`), via the **Responses API** by default. See comments in `src/llm/summary.py` to switch to Chat Completions if you prefer.
