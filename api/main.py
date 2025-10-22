# api/main.py
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import date
from typing import Optional, Tuple
import pandas as pd
import traceback
import os

# LLM summaries (exec summary + structured brief)
from src.llm.summary import build_executive_summary, summarize_tweets

# --- Load the repo-root .env no matter where Uvicorn is started from ---
ROOT_DOTENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_DOTENV)

def _read_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "") or ""
    return key.strip().strip('"').strip("'")

# ---- Theme computation (Stage 3 dynamic) ----
# requires src/features/themes.py with compute_themes_payload()
from src.features.themes import compute_themes_payload

# ------------ Paths ------------
SENTI_PATH   = "data/tweets_stage1_sentiment.parquet"
ASPECT_PATH  = "data/tweets_stage2_aspects.parquet"
STAGE3_PATH  = "data/tweets_stage3_aspect_sentiment.parquet"  # optional cache (no dates)
STAGE3_THEMES_PARQUET = "data/tweets_stage3_themes.parquet"   # written by /themes

app = FastAPI(title="Walmart Social Listener API")

# Allow calls from Vite dev server (add prod origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Helpers ------------
def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["createdat", "created_dt", "created_at", "date"]:
        if c in df.columns:
            return c
    raise KeyError("No createdat/date column found in parquet.")

def _normalize_dates(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).copy()
    out["date_only"] = out[date_col].dt.date
    return out

def _sentiment_summary(sub: pd.DataFrame) -> dict:
    counts = sub["sentiment_label"].value_counts().to_dict()
    total = int(sum(counts.values()) or 1)
    pct = {k: round(v / total * 100, 2) for k, v in counts.items()}
    for k in ["positive", "neutral", "negative"]:
        counts.setdefault(k, 0)
        pct.setdefault(k, 0.0)
    return {"total": total, "counts": counts, "percent": pct}

def _aspect_split_from_subset(sub: pd.DataFrame, aspects: list[str], include_others: bool = False) -> dict:
    if sub.empty:
        zero = [0 for _ in aspects]
        zero_f = [0.0 for _ in aspects]
        labels = aspects.copy()
        if include_others:
            labels.append("others")
            zero.append(0)
            zero_f.append(0.0)
        return {
            "labels": labels,
            "counts": {"positive": zero, "neutral": zero, "negative": zero},
            "percent": {"positive": zero_f, "neutral": zero_f, "negative": zero_f},
        }

    g = (
        sub.groupby(["aspect_dominant", "sentiment_label"])
          .size().reset_index(name="count")
    )
    pivot = (
        g.pivot(index="aspect_dominant", columns="sentiment_label", values="count")
         .fillna(0)
    )
    for col in ["positive", "neutral", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0

    # Handle predefined aspects
    predefined_pivot = pivot.reindex(aspects, fill_value=0)
    
    labels = aspects.copy()
    counts = {
        "positive": [int(x) for x in predefined_pivot["positive"].tolist()],
        "neutral":  [int(x) for x in predefined_pivot["neutral"].tolist()],
        "negative": [int(x) for x in predefined_pivot["negative"].tolist()],
    }
    percent = {
        "positive": [float(x) for x in predefined_pivot["positive"].tolist()],
        "neutral":  [float(x) for x in predefined_pivot["neutral"].tolist()],
        "negative": [float(x) for x in predefined_pivot["negative"].tolist()],
    }

    # Add "Others" category if requested
    if include_others:
        # Find aspects not in predefined list
        all_aspects = sub["aspect_dominant"].unique()
        other_aspects = [a for a in all_aspects if a not in aspects]
        
        if len(other_aspects) > 0:
            others_data = sub[sub["aspect_dominant"].isin(other_aspects)]
            others_counts = others_data.groupby("sentiment_label").size()
            
            labels.append("others")
            counts["positive"].append(int(others_counts.get("positive", 0)))
            counts["neutral"].append(int(others_counts.get("neutral", 0)))
            counts["negative"].append(int(others_counts.get("negative", 0)))
            
            # Calculate percentages for others
            others_total = others_counts.sum()
            if others_total > 0:
                percent["positive"].append(float((others_counts.get("positive", 0) / others_total * 100).round(2)))
                percent["neutral"].append(float((others_counts.get("neutral", 0) / others_total * 100).round(2)))
                percent["negative"].append(float((others_counts.get("negative", 0) / others_total * 100).round(2)))
            else:
                percent["positive"].append(0.0)
                percent["neutral"].append(0.0)
                percent["negative"].append(0.0)
        else:
            labels.append("others")
            counts["positive"].append(0)
            counts["neutral"].append(0)
            counts["negative"].append(0)
            percent["positive"].append(0.0)
            percent["neutral"].append(0.0)
            percent["negative"].append(0.0)

    return {
        "labels": labels,
        "counts": counts,
        "percent": percent,
    }

def _pick_any_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["createdat", "created_dt", "created_at", "tweet_date", "date", "dt"]:
        if c in df.columns:
            return c
    return None

# ------------ Load Sentiment (Stage 1) ------------
if not os.path.exists(SENTI_PATH):
    raise FileNotFoundError(f"Missing parquet: {SENTI_PATH}. Run Stage 1 first.")

df = pd.read_parquet(SENTI_PATH)
_sent_date_col = _detect_date_col(df)
df = _normalize_dates(df, _sent_date_col)
SENT_MIN_DATE = df["date_only"].min()
SENT_MAX_DATE = df["date_only"].max()

# ------------ Load Aspects (Stage 2) ------------
ASPECTS = ["pricing", "delivery", "returns", "staff", "app/ux"]

if os.path.exists(ASPECT_PATH):
    adf = pd.read_parquet(ASPECT_PATH)
    _asp_date_col = _detect_date_col(adf)
    adf = _normalize_dates(adf, _asp_date_col)
    ASPECT_MIN_DATE = adf["date_only"].min()
    ASPECT_MAX_DATE = adf["date_only"].max()
else:
    adf = pd.DataFrame(columns=["date_only", "aspect_dominant", "sentiment_label"])
    ASPECT_MIN_DATE = SENT_MIN_DATE
    ASPECT_MAX_DATE = SENT_MAX_DATE

# Optional cache without dates (Stage 3)
stage3_df = None
if os.path.exists(STAGE3_PATH):
    try:
        stage3_df = pd.read_parquet(STAGE3_PATH)
        for c in ["aspect_dominant", "positive", "neutral", "negative"]:
            assert c in stage3_df.columns
    except Exception:
        stage3_df = None

# ---------- Simple in-process cache for /themes ----------
_THEMES_CACHE: dict[Tuple[Optional[str], Optional[str], int, str], dict] = {}

# ------------ Routes ------------
@app.get("/")
def health():
    return {
        "message": "✅ Walmart Sentiment API is running!",
        "date_range": {"min": str(SENT_MIN_DATE), "max": str(SENT_MAX_DATE)},
        "aspect_date_range": {"min": str(ASPECT_MIN_DATE), "max": str(ASPECT_MAX_DATE)},
        "has_aspects": bool(len(adf) > 0),
        "has_stage3_cache": bool(stage3_df is not None),
        "env_loaded": os.path.exists(ROOT_DOTENV),
        "has_openai_key": bool(_read_openai_key()),
    }

# --- Sentiment ---
@app.get("/sentiment/summary")
def sentiment_summary(
    start: date = Query(default=SENT_MIN_DATE),
    end:   date = Query(default=SENT_MAX_DATE),
):
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    if sub.empty:
        return {
            "start": str(start), "end": str(end),
            "total": 0,
            "counts": {"positive": 0, "neutral": 0, "negative": 0},
            "percent": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
        }
    return {"start": str(start), "end": str(end), **_sentiment_summary(sub)}

@app.get("/sentiment/trend")
def sentiment_trend(
    start: date = Query(default=SENT_MIN_DATE),
    end:   date = Query(default=SENT_MAX_DATE),
):
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    if sub.empty:
        return {"start": str(start), "end": str(end), "trend": []}

    daily = (
        sub.groupby(["date_only", "sentiment_label"])
           .size().reset_index(name="count")
           .pivot(index="date_only", columns="sentiment_label", values="count")
           .fillna(0)
    )
    daily = (daily.div(daily.sum(axis=1), axis=0) * 100).reset_index()

    for c in ["positive", "neutral", "negative"]:
        if c not in daily.columns:
            daily[c] = 0.0

    trend = [
        {"date": str(r["date_only"]),
         "positive": float(r["positive"]),
         "neutral":  float(r["neutral"]),
         "negative": float(r["negative"])}
        for _, r in daily.sort_values("date_only").iterrows()
    ]
    return {"start": str(start), "end": str(end), "trend": trend}

# --- Aspects (simple distribution) ---
@app.get("/aspects/summary")
def aspects_summary(
    start: date = Query(default=None),
    end:   date = Query(default=None),
    as_percent: bool = Query(default=False),
):
    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE

    if adf.empty:
        counts = {a: 0 for a in ASPECTS}
        pct = {a: 0.0 for a in ASPECTS}
        return {
            "start": str(s), "end": str(e),
            "counts": counts, "percent": pct, "total": 0,
            "labels": ASPECTS, "series": (pct if as_percent else counts)
        }

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask]

    if sub.empty:
        counts = {a: 0 for a in ASPECTS}
        pct = {a: 0.0 for a in ASPECTS}
        return {
            "start": str(s), "end": str(e),
            "counts": counts, "percent": pct, "total": 0,
            "labels": ASPECTS, "series": (pct if as_percent else counts)
        }

    dom_counts = sub["aspect_dominant"].value_counts().to_dict()
    counts = {a: int(dom_counts.get(a, 0)) for a in ASPECTS}
    total = int(sum(counts.values()))
    percent = {a: round((counts[a] / total * 100), 2) if total else 0.0 for a in ASPECTS}

    return {
        "start": str(s),
        "end": str(e),
        "counts": counts,
        "percent": percent,
        "total": total,
        "labels": ASPECTS,
        "series": (percent if as_percent else counts),
    }

@app.get("/aspects/avg-scores")
def aspects_avg_scores(
    start: date = Query(default=None),
    end:   date = Query(default=None),
):
    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE

    if adf.empty:
        return {"start": str(s), "end": str(e), "avg_scores": {}}

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask]

    score_cols = [c for c in adf.columns if c.startswith("aspect_") and c != "aspect_dominant"]
    if sub.empty or not score_cols:
        return {"start": str(s), "end": str(e), "avg_scores": {c: 0.0 for c in score_cols}}

    avg = sub[score_cols].mean().to_dict()
    avg = {k: round(float(v), 4) for k, v in avg.items()}
    return {"start": str(s), "end": str(e), "avg_scores": avg}

# --- Aspects (stacked bar: sentiment split per aspect) ---
@app.get("/aspects/sentiment-split")
def aspects_sentiment_split(
    start: date = Query(default=None),
    end:   date = Query(default=None),
    as_percent: bool = Query(default=False),
    include_others: bool = Query(default=False),
):
    if start is None and end is None and stage3_df is not None:
        s3 = stage3_df.set_index("aspect_dominant").reindex(ASPECTS, fill_value=0)
        for c in ["positive", "neutral", "negative"]:
            if c not in s3.columns:
                s3[c] = 0
        totals = s3[["positive", "neutral", "negative"]].sum(axis=1).replace(0, 1)
        pct = (s3[["positive", "neutral", "negative"]].div(totals, axis=0) * 100).round(2)

        labels = ASPECTS.copy()
        counts = {
                "positive": [int(x) for x in s3["positive"].tolist()],
                "neutral":  [int(x) for x in s3["neutral"].tolist()],
                "negative": [int(x) for x in s3["negative"].tolist()],
        }
        percent = {
                "positive": [float(x) for x in pct["positive"].tolist()],
                "neutral":  [float(x) for x in pct["neutral"].tolist()],
                "negative": [float(x) for x in pct["negative"].tolist()],
        }

        # Add "Others" category if requested
        if include_others:
            # Calculate others by finding aspects not in predefined list
            all_aspects = stage3_df["aspect_dominant"].unique()
            other_aspects = [a for a in all_aspects if a not in ASPECTS]
            
            if len(other_aspects) > 0:
                others_data = stage3_df[stage3_df["aspect_dominant"].isin(other_aspects)]
                others_counts = others_data[["positive", "neutral", "negative"]].sum()
                
                labels.append("others")
                counts["positive"].append(int(others_counts["positive"]))
                counts["neutral"].append(int(others_counts["neutral"]))
                counts["negative"].append(int(others_counts["negative"]))
                
                # Calculate percentages for others
                others_total = others_counts.sum()
                if others_total > 0:
                    percent["positive"].append(float((others_counts["positive"] / others_total * 100).round(2)))
                    percent["neutral"].append(float((others_counts["neutral"] / others_total * 100).round(2)))
                    percent["negative"].append(float((others_counts["negative"] / others_total * 100).round(2)))
                else:
                    percent["positive"].append(0.0)
                    percent["neutral"].append(0.0)
                    percent["negative"].append(0.0)

        payload = {
            "labels": labels,
            "counts": counts,
            "percent": percent,
        }
        return payload

    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE
    if adf.empty:
        return _aspect_split_from_subset(pd.DataFrame(), ASPECTS, include_others)

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask, ["aspect_dominant", "sentiment_label"]]
    return _aspect_split_from_subset(sub, ASPECTS, include_others)

# --- Executive summary over a date window (LLM-powered with fallback) ---
@app.get("/executive-summary")
def executive_summary(
    start: str = Query(..., description="YYYY-MM-DD"),
    end:   str = Query(..., description="YYYY-MM-DD"),
    sample_per_sentiment: int = Query(default=250, ge=50, le=500),
):
    """
    Summarizes all tweets in the selected duration.
    Uses OpenAI if OPENAI_API_KEY is configured, otherwise a rule-based fallback.
    Returns: {start, end, used_llm, summary, stats:{sentiment, top_aspects, keywords}}
    """
    try:
        result = build_executive_summary(
            df_senti=df,            # Stage 1 DF (has date_only, sentiment_label, text cols)
            df_aspects=adf,         # Stage 2 DF (for top aspects)
            start=start,
            end=end,
            openai_api_key=_read_openai_key(),
            sample_per_sentiment=sample_per_sentiment,
        )
        return result
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace_tail": tb.splitlines()[-6:]},
        )

# --- Structured brief (bullets/themes/risks/opps) ---
@app.get("/structured-brief")
def structured_brief(
    start: str = Query(..., description="YYYY-MM-DD"),
    end:   str = Query(..., description="YYYY-MM-DD"),
    keyword: Optional[str] = Query(default=None),
    sample_size: int = Query(default=50, ge=20, le=200),
):
    try:
        # Work on a copy and ensure we expose exactly ONE 'date' column
        df_for_llm = df.copy()

        if "date" in df_for_llm.columns:
            pass  # already present
        elif "date_only" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["date_only"]
        elif "created_at" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["created_at"]
        elif "timestamp" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["timestamp"]
        else:
            maybe = [c for c in df_for_llm.columns if "date" in c or "time" in c]
            if maybe:
                df_for_llm["date"] = df_for_llm[maybe[0]]
            else:
                df_for_llm["date"] = pd.NaT

        # If multiple 'date' columns somehow exist, keep the first and drop the rest
        if (df_for_llm.columns == "date").sum() > 1:
            first_idx = [i for i, c in enumerate(df_for_llm.columns) if c == "date"][0]
            keep = list(range(len(df_for_llm.columns)))
            for i, c in enumerate(df_for_llm.columns):
                if c == "date" and i != first_idx:
                    keep.remove(i)
            df_for_llm = df_for_llm.iloc[:, keep]

        res = summarize_tweets(
            df=df_for_llm,
            start_date=start,
            end_date=end,
            keyword=keyword,
            sample_size=sample_size,
        )
        return res
    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "trace_tail": tb.splitlines()[-6:]},
        )

# --- Themes (dynamic clustering + summaries) ---
@app.get("/themes")
def themes(
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end:   Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    n_clusters: int      = Query(default=6, ge=2, le=6),  # Limited to 6 themes max
    emb_model: str       = Query(default="sentence-transformers/all-MiniLM-L6-v2"),
    merge_similar: bool  = Query(default=True, description="Automatically merge similar themes"),
):
    """
    Returns:
      { updated_at, used_llm, themes: [{id,name,summary,tweet_count,positive,negative,neutral}] }
    """
    key = (start, end, int(n_clusters), emb_model)
    if key in _THEMES_CACHE:
        return _THEMES_CACHE[key]

    try:
        payload = compute_themes_payload(
            parquet_stage2=ASPECT_PATH,
            n_clusters=n_clusters,
            emb_model=emb_model,
            start_date=start,
            end_date=end,
            openai_api_key=_read_openai_key(),
            merge_similar=merge_similar,
        )
        _THEMES_CACHE[key] = payload
        return payload
    except Exception as e:
        tb = traceback.format_exc()
        print("[/themes ERROR]", e, "\n", tb)
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "hint": "Check OpenAI key, sklearn/torch availability, date range, and data paths.",
                "trace_tail": tb.splitlines()[-6:],
            },
        )

# --- Tweets drill-down for a theme (reads STAGE3_THEMES_PARQUET) ---
@app.get("/themes/{theme_id}/tweets")
def theme_tweets(
    theme_id: int,
    limit: int = Query(default=10, ge=1, le=200),
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end:   Optional[str] = Query(default=None, description="YYYY-MM-DD"),
):
    if not os.path.exists(STAGE3_THEMES_PARQUET):
        return {"items": [], "note": "Stage-3 parquet not found. Call /themes first to generate."}

    df3 = pd.read_parquet(STAGE3_THEMES_PARQUET)
    if "theme" not in df3.columns:
        return {"items": [], "note": "'theme' column not present in stage-3 parquet."}

    date_col = _pick_any_date_col(df3)
    if date_col:
        df3[date_col] = pd.to_datetime(df3[date_col], errors="coerce")
        if start:
            df3 = df3[df3[date_col] >= pd.to_datetime(start)]
        if end:
            df3 = df3[df3[date_col] <= pd.to_datetime(end)]

    sub = df3[df3["theme"] == int(theme_id)].copy()
    if sub.empty:
        return {"items": []}

    if date_col:
        sub = sub.sort_values(date_col, ascending=False)

    cols_keep = [c for c in [
        date_col, "sentiment_label", "sentiment_score",
        "aspect_dominant", "twitter_url", "tweet_url",
        "text_used", "clean_tweet", "text", "fulltext"
    ] if c and c in sub.columns]

    def _pick_text(row: dict) -> str:
        for c in ["text_used", "clean_tweet", "text", "fulltext"]:
            if c in row and c in row and row[c]:
                return str(row[c])
        return ""

    items = []
    for _, r in sub[cols_keep].head(int(limit)).iterrows():
        d = r.to_dict()
        created = str(d.get(date_col)) if date_col else ""
        url_val = d.get("twitter_url") or d.get("tweet_url") or ""
        items.append({
            "date": created,                 # new
            "createdat": created,            # legacy-friendly
            "sentiment_label": d.get("sentiment_label"),
            "sentiment_score": d.get("sentiment_score"),
            "aspect_dominant": d.get("aspect_dominant"),
            "url": url_val,                  # new
            "twitterurl": url_val,           # legacy-friendly
            "text": _pick_text(d),           # new
            "text_clean": _pick_text(d),     # legacy-friendly
        })

    return {"items": items}

