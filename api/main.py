# api/main.py
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from datetime import date
from typing import Optional, Tuple

import pandas as pd
import traceback
import os
from io import StringIO, BytesIO

# LLM summaries (exec summary + structured brief)
from src.llm.summary import build_executive_summary, summarize_tweets

# Theme computation (Stage 3 dynamic)
from src.features.themes import compute_themes_payload

# --- Load the repo-root .env no matter where Uvicorn is started from ---
ROOT_DOTENV = Path(__file__).resolve().parents[1] / ".env"
load_dotenv(dotenv_path=ROOT_DOTENV)


def _read_openai_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "") or ""
    return key.strip().strip('"').strip("'")


# ------------ Paths ------------
RAW_TWEETS_PATH = "data/tweets_stage0_raw.parquet"

SENTI_PATH = "data/tweets_stage1_sentiment.parquet"
ASPECT_PATH = "data/tweets_stage2_aspects.parquet"

STAGE3_PATH = "data/tweets_stage3_aspect_sentiment.parquet"  # optional cache (no dates)
STAGE3_THEMES_PARQUET = "data/tweets_stage3_themes.parquet"  # written by /themes


app = FastAPI(title="Walmart Social Listener API")


# ------------------------------------------------------------
# IMPORTANT:
# Frontend calls /api/..., but many routes are /sentiment/... etc.
# This helper registers BOTH:
#   /xyz    and   /api/xyz
# ------------------------------------------------------------
def _alias(route_path: str):
    def decorator(func):
        app.get(route_path)(func)
        app.get("/api" + route_path)(func)
        return func
    return decorator


# Allow calls from Vite dev server + Netlify (prod + previews)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------ Helpers ------------
def _detect_date_col(df: pd.DataFrame) -> str:
    for c in ["createdat", "created_dt", "created_at", "tweet_date", "date", "dt", "timestamp"]:
        if c in df.columns:
            return c
    raise KeyError("No created/date column found in parquet.")


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
    return {"total": int(sum(counts.values())), "counts": counts, "percent": pct}


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

    predefined = pivot.reindex(aspects, fill_value=0)

    labels = aspects.copy()
    counts = {
        "positive": [int(x) for x in predefined["positive"].tolist()],
        "neutral":  [int(x) for x in predefined["neutral"].tolist()],
        "negative": [int(x) for x in predefined["negative"].tolist()],
    }

    totals = (predefined[["positive", "neutral", "negative"]].sum(axis=1).replace(0, 1))
    pct_df = (predefined[["positive", "neutral", "negative"]].div(totals, axis=0) * 100).round(2)
    percent = {
        "positive": [float(x) for x in pct_df["positive"].tolist()],
        "neutral":  [float(x) for x in pct_df["neutral"].tolist()],
        "negative": [float(x) for x in pct_df["negative"].tolist()],
    }

    if include_others:
        all_aspects = sub["aspect_dominant"].astype(str).unique()
        other_aspects = [a for a in all_aspects if a not in aspects]

        labels.append("others")
        if other_aspects:
            others = sub[sub["aspect_dominant"].astype(str).isin(other_aspects)]
            oc = others.groupby("sentiment_label").size()
            pos = int(oc.get("positive", 0))
            neu = int(oc.get("neutral", 0))
            neg = int(oc.get("negative", 0))
            counts["positive"].append(pos)
            counts["neutral"].append(neu)
            counts["negative"].append(neg)
            ot_total = pos + neu + neg
            if ot_total > 0:
                percent["positive"].append(round(pos / ot_total * 100, 2))
                percent["neutral"].append(round(neu / ot_total * 100, 2))
                percent["negative"].append(round(neg / ot_total * 100, 2))
            else:
                percent["positive"].append(0.0)
                percent["neutral"].append(0.0)
                percent["negative"].append(0.0)
        else:
            counts["positive"].append(0)
            counts["neutral"].append(0)
            counts["negative"].append(0)
            percent["positive"].append(0.0)
            percent["neutral"].append(0.0)
            percent["negative"].append(0.0)

    return {"labels": labels, "counts": counts, "percent": percent}


def _pick_any_date_col(df: pd.DataFrame) -> Optional[str]:
    for c in ["createdat", "created_dt", "created_at", "tweet_date", "date", "dt", "timestamp"]:
        if c in df.columns:
            return c
    return None


def _ensure_llm_columns(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    summarize_tweets() typically needs:
      - date column named 'date'
      - a text column (we force 'text')
      - sentiment_label (best-effort)
    This makes the endpoint resilient to column-name differences across datasets.
    """
    df_out = df_in.copy()

    # --- Ensure date column named 'date' ---
    if "date" not in df_out.columns:
        # Prefer date_only if present, else try detect any date-ish column
        if "date_only" in df_out.columns:
            df_out["date"] = df_out["date_only"]
        else:
            dcol = _pick_any_date_col(df_out)
            df_out["date"] = df_out[dcol] if dcol else pd.NaT

    df_out["date"] = pd.to_datetime(df_out["date"], errors="coerce")

    # --- Ensure text column named 'text' ---
    text_candidates = ["text", "text_used", "clean_tweet", "text_clean", "fulltext", "tweet_text", "content"]
    if "text" not in df_out.columns:
        picked = None
        for c in text_candidates:
            if c in df_out.columns:
                picked = c
                break
        if picked is not None:
            df_out["text"] = df_out[picked].astype(str)
        else:
            df_out["text"] = ""

    # --- Ensure sentiment_label exists ---
    if "sentiment_label" not in df_out.columns:
        df_out["sentiment_label"] = "neutral"

    # Clean empties
    df_out["text"] = df_out["text"].fillna("").astype(str)

    return df_out


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
    adf["aspect_dominant"] = (
    adf["aspect_dominant"]
      .astype(str).str.strip().str.lower()
      .replace({"app_ux": "app/ux"}))
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

# In-process cache for /themes
_THEMES_CACHE: dict[Tuple[Optional[str], Optional[str], int, str], dict] = {}


# ------------ Routes ------------
@app.get("/")
@app.get("/api/")
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
@_alias("/sentiment/summary")
def sentiment_summary(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
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


@_alias("/sentiment/trend")
def sentiment_trend(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    period: str = Query(default="daily"),  # accepted
    offset: int = Query(default=0),        # accepted
    limit: int = Query(default=0),         # accepted
):
    _ = period, offset, limit  # unused, but prevents 422 if frontend passes them

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
        {
            "date": str(r["date_only"]),
            "positive": float(r["positive"]),
            "neutral": float(r["neutral"]),
            "negative": float(r["negative"]),
        }
        for _, r in daily.sort_values("date_only").iterrows()
    ]
    return {"start": str(start), "end": str(end), "trend": trend}


# --- Aspects (simple distribution) ---
@_alias("/aspects/summary")
def aspects_summary(
    start: date = Query(default=None),
    end: date = Query(default=None),
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
            "labels": ASPECTS, "series": (pct if as_percent else counts),
        }

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask]
    if sub.empty:
        counts = {a: 0 for a in ASPECTS}
        pct = {a: 0.0 for a in ASPECTS}
        return {
            "start": str(s), "end": str(e),
            "counts": counts, "percent": pct, "total": 0,
            "labels": ASPECTS, "series": (pct if as_percent else counts),
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


@_alias("/aspects/avg-scores")
def aspects_avg_scores(
    start: date = Query(default=None),
    end: date = Query(default=None),
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


@_alias("/aspects/sentiment-split")
def aspects_sentiment_split(
    start: date = Query(default=None),
    end: date = Query(default=None),
    as_percent: bool = Query(default=False),     # accepted, payload includes both counts & percent anyway
    include_others: bool = Query(default=False),
):
    _ = as_percent  # not needed (payload already contains percent arrays)

    # Use cached Stage3 if no dates passed
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

        if include_others:
            all_aspects = stage3_df["aspect_dominant"].astype(str).unique()
            other_aspects = [a for a in all_aspects if a not in ASPECTS]
            labels.append("others")
            if other_aspects:
                others_data = stage3_df[stage3_df["aspect_dominant"].astype(str).isin(other_aspects)]
                oc = others_data[["positive", "neutral", "negative"]].sum()
                pos, neu, neg = int(oc.get("positive", 0)), int(oc.get("neutral", 0)), int(oc.get("negative", 0))
                counts["positive"].append(pos)
                counts["neutral"].append(neu)
                counts["negative"].append(neg)
                ot = pos + neu + neg
                if ot > 0:
                    percent["positive"].append(round(pos / ot * 100, 2))
                    percent["neutral"].append(round(neu / ot * 100, 2))
                    percent["negative"].append(round(neg / ot * 100, 2))
                else:
                    percent["positive"].append(0.0)
                    percent["neutral"].append(0.0)
                    percent["negative"].append(0.0)
            else:
                counts["positive"].append(0)
                counts["neutral"].append(0)
                counts["negative"].append(0)
                percent["positive"].append(0.0)
                percent["neutral"].append(0.0)
                percent["negative"].append(0.0)

        return {"labels": labels, "counts": counts, "percent": percent}

    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE
    if adf.empty:
        return _aspect_split_from_subset(pd.DataFrame(), ASPECTS, include_others)

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask, ["aspect_dominant", "sentiment_label"]]
    return _aspect_split_from_subset(sub, ASPECTS, include_others)


# --- Executive summary (LLM-powered with fallback) ---
@_alias("/executive-summary")
def executive_summary(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    sample_per_sentiment: int = Query(default=250, ge=50, le=500),
):
    try:
        result = build_executive_summary(
            df_senti=df,
            df_aspects=adf,
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
            content={"error": str(e), "trace_tail": tb.splitlines()[-10:]},
        )


# --- Structured brief ---
@_alias("/structured-brief")
def structured_brief(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    keyword: Optional[str] = Query(default=None),
    sample_size: int = Query(default=80, ge=20, le=200),
):
    """
    This endpoint was throwing 500 in your UI.
    Most common cause: summarize_tweets() expects certain columns (date/text).
    We enforce them here.
    """
    try:
        df_for_llm = _ensure_llm_columns(df)

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
            content={
                "error": str(e),
                "hint": "Structured brief failed. Check that Stage1 parquet has a usable text column and date column. This endpoint now auto-maps columns.",
                "trace_tail": tb.splitlines()[-10:],
            },
        )


# --- Themes (dynamic clustering + summaries) ---
@_alias("/themes")
def themes(
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    n_clusters: Optional[int] = Query(default=None, ge=1, le=8),
    emb_model: str = Query(default="sentence-transformers/all-MiniLM-L6-v2"),
    merge_similar: bool = Query(default=True),
    parquet: Optional[str] = Query(default=None),   # accepted
    max_rows: Optional[int] = Query(default=None),  # accepted
):
    _ = emb_model, merge_similar  # (your compute_themes_payload currently ignores these)

    key = (start, end, int(n_clusters or 0), "default")
    if key in _THEMES_CACHE:
        del _THEMES_CACHE[key]

    try:
        df_raw = pd.read_parquet(parquet or RAW_TWEETS_PATH)
        if isinstance(max_rows, int) and max_rows > 0:
            df_raw = df_raw.head(max_rows)

        payload = compute_themes_payload(
            df=df_raw,
            start_date=start,
            end_date=end,
            n_clusters=n_clusters,
            openai_api_key=_read_openai_key(),
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
                "trace_tail": tb.splitlines()[-10:],
            },
        )


# --- Tweets drill-down for a theme (reads STAGE3_THEMES_PARQUET) ---
@_alias("/themes/{theme_id}/tweets")
def theme_tweets(
    theme_id: int,
    limit: int = Query(default=10, ge=1, le=200),
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD"),
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
            if c in row and row.get(c):
                return str(row[c])
        return ""

    items = []
    for _, r in sub[cols_keep].head(int(limit)).iterrows():
        d = r.to_dict()
        created = str(d.get(date_col)) if date_col else ""
        url_val = d.get("twitter_url") or d.get("tweet_url") or ""
        items.append({
            "date": created,
            "createdat": created,
            "sentiment_label": d.get("sentiment_label"),
            "sentiment_score": d.get("sentiment_score"),
            "aspect_dominant": d.get("aspect_dominant"),
            "url": url_val,
            "twitterurl": url_val,
            "text": _pick_text(d),
            "text_clean": _pick_text(d),
        })

    return {"items": items}


# --- Sample tweets for specific aspect and sentiment ---
@_alias("/tweets/sample")
def sample_tweets(
    start: date = Query(default=None),
    end: date = Query(default=None),
    aspect: str = Query(..., description="Aspect name (e.g., pricing, delivery)"),
    sentiment: str = Query(..., description="Sentiment (positive, neutral, negative)"),
    limit: int = Query(default=10, ge=1, le=1000),
):
    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE

    if adf.empty:
        return {"tweets": [], "count": 0, "aspect": aspect, "sentiment": sentiment}

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask]

    if sub.empty:
        return {"tweets": [], "count": 0, "aspect": aspect, "sentiment": sentiment}

    aspect_mask = sub["aspect_dominant"].astype(str).str.lower() == aspect.lower()
    sentiment_mask = sub["sentiment_label"].astype(str).str.lower() == sentiment.lower()
    filtered = sub[aspect_mask & sentiment_mask]

    if filtered.empty:
        return {"tweets": [], "count": 0, "aspect": aspect, "sentiment": sentiment}

    text_col = None
    for col in ["text", "clean_tweet", "text_used", "fulltext", "tweet_text", "text_clean"]:
        if col in filtered.columns:
            text_col = col
            break

    if text_col is None:
        return {"tweets": [], "count": 0, "aspect": aspect, "sentiment": sentiment, "error": "No text column found"}

    tweets_out = filtered[text_col].dropna().head(limit).tolist()

    return {
        "tweets": tweets_out,
        "count": len(tweets_out),
        "aspect": aspect,
        "sentiment": sentiment,
        "total_available": int(len(filtered)),
    }


# --- Raw Data Downloads ---
@_alias("/tweets/raw")
def download_raw_tweets(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    format: str = Query(default="csv", regex="^(csv|xlsx)$"),
):
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    if sub.empty:
        return {"error": "No data found for the specified date range"}

    columns = ["date", "createdat", "text", "text_clean", "sentiment_label", "aspect_dominant", "user_id"]
    available_columns = [col for col in columns if col in sub.columns]
    export_data = sub[available_columns].copy()

    if format == "csv":
        csv_buffer = StringIO()
        export_data.to_csv(csv_buffer, index=False)
        return Response(
            content=csv_buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=raw_tweets_{start}_to_{end}.csv"},
        )

    excel_buffer = BytesIO()
    with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
        export_data.to_excel(writer, sheet_name="Raw Tweets", index=False)

    return Response(
        content=excel_buffer.getvalue(),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename=raw_tweets_{start}_to_{end}.xlsx"},
    )
