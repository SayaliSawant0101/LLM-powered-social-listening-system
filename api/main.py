# api/main.py
from __future__ import annotations

from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from datetime import date, datetime
from typing import Optional, Tuple
import pandas as pd
import traceback
import os
import json
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


# Load raw tweets data for theme generation
RAW_TWEETS_PATH = "data/tweets_stage0_raw.parquet"

# ------------ Paths ------------
SENTI_PATH = "data/tweets_stage1_sentiment.parquet"
ASPECT_PATH = "data/tweets_stage2_aspects.parquet"
STAGE3_PATH = "data/tweets_stage3_aspect_sentiment.parquet"  # optional cache (no dates)
STAGE3_THEMES_PARQUET = "data/tweets_stage3_themes.parquet"  # written by /themes

app = FastAPI(title="Walmart Social Listener API")

# ------------------------------------------------------------
# Helper: register BOTH routes:
#   /xyz   and   /api/xyz
# ------------------------------------------------------------
def _alias(route_path: str):
    """
    Decorator to register the same endpoint under two paths:
      /xyz   and   /api/xyz
    """
    def decorator(func):
        app.get(route_path)(func)
        app.get("/api" + route_path)(func)
        return func
    return decorator


# ------------------------------------------------------------
# ✅ CORS FIX (Netlify + local dev)
#   - allow_origin_regex handles Netlify previews + prod
#   - allow_credentials=True prevents browser "Failed to fetch" when credentials/headers are present
# ------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:3001",
    ],
    allow_origin_regex=r"https://.*\.netlify\.app",
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

    predefined_pivot = pivot.reindex(aspects, fill_value=0)

    labels = aspects.copy()
    counts = {
        "positive": [int(x) for x in predefined_pivot["positive"].tolist()],
        "neutral":  [int(x) for x in predefined_pivot["neutral"].tolist()],
        "negative": [int(x) for x in predefined_pivot["negative"].tolist()],
    }

    totals = (predefined_pivot[["positive", "neutral", "negative"]].sum(axis=1).replace(0, 1))
    pct_df = (predefined_pivot[["positive", "neutral", "negative"]].div(totals, axis=0) * 100).round(2)
    percent = {
        "positive": [float(x) for x in pct_df["positive"].tolist()],
        "neutral":  [float(x) for x in pct_df["neutral"].tolist()],
        "negative": [float(x) for x in pct_df["negative"].tolist()],
    }

    if include_others:
        all_aspects = sub["aspect_dominant"].unique()
        other_aspects = [a for a in all_aspects if a not in aspects]
        labels.append("others")
        if other_aspects:
            others_data = sub[sub["aspect_dominant"].isin(other_aspects)]
            others_counts = others_data.groupby("sentiment_label").size()
            pos = int(others_counts.get("positive", 0))
            neu = int(others_counts.get("neutral", 0))
            neg = int(others_counts.get("negative", 0))
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
    for c in ["createdat", "created_dt", "created_at", "tweet_date", "date", "dt"]:
        if c in df.columns:
            return c
    return None


def _parse_date_str(s: str) -> str:
    """
    Accepts:
      - YYYY-MM-DD
      - MM/DD/YYYY
    Returns ISO string YYYY-MM-DD.
    """
    s = (s or "").strip()
    if not s:
        return s
    for fmt in ("%Y-%m-%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            pass
    # if already something else, return as-is (build_executive_summary may handle)
    return s


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
@app.get("/api")
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
    period: str = Query(default="daily"),
    offset: int = Query(default=0),
    limit: int = Query(default=0),
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

        if include_others:
            all_aspects = stage3_df["aspect_dominant"].unique()
            other_aspects = [a for a in all_aspects if a not in ASPECTS]
            labels.append("others")
            if other_aspects:
                others_data = stage3_df[stage3_df["aspect_dominant"].isin(other_aspects)]
                others_counts = others_data[["positive", "neutral", "negative"]].sum()
                pos = int(others_counts.get("positive", 0))
                neu = int(others_counts.get("neutral", 0))
                neg = int(others_counts.get("negative", 0))
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

    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE
    if adf.empty:
        return _aspect_split_from_subset(pd.DataFrame(), ASPECTS, include_others)

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask, ["aspect_dominant", "sentiment_label"]]
    return _aspect_split_from_subset(sub, ASPECTS, include_others)


# --- Executive summary (LLM-powered) ---
@_alias("/executive-summary")
def executive_summary(
    start: str = Query(..., description="YYYY-MM-DD or MM/DD/YYYY"),
    end: str = Query(..., description="YYYY-MM-DD or MM/DD/YYYY"),
    sample_per_sentiment: int = Query(default=250, ge=50, le=500),
):
    try:
        start_iso = _parse_date_str(start)
        end_iso = _parse_date_str(end)

        result = build_executive_summary(
            df_senti=df,
            df_aspects=adf,
            start=start_iso,
            end=end_iso,
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


# --- Structured brief ---
@_alias("/structured-brief")
def structured_brief(
    start: str = Query(..., description="YYYY-MM-DD or MM/DD/YYYY"),
    end: str = Query(..., description="YYYY-MM-DD or MM/DD/YYYY"),
    keyword: Optional[str] = Query(default=None),
    sample_size: int = Query(default=50, ge=20, le=200),
):
    try:
        start_iso = _parse_date_str(start)
        end_iso = _parse_date_str(end)

        df_for_llm = df.copy()

        if "date" in df_for_llm.columns:
            pass
        elif "date_only" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["date_only"]
        elif "created_at" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["created_at"]
        elif "timestamp" in df_for_llm.columns:
            df_for_llm["date"] = df_for_llm["timestamp"]
        else:
            maybe = [c for c in df_for_llm.columns if "date" in c or "time" in c]
            df_for_llm["date"] = df_for_llm[maybe[0]] if maybe else pd.NaT

        if (df_for_llm.columns == "date").sum() > 1:
            first_idx = [i for i, c in enumerate(df_for_llm.columns) if c == "date"][0]
            keep = list(range(len(df_for_llm.columns)))
            for i, c in enumerate(df_for_llm.columns):
                if c == "date" and i != first_idx:
                    keep.remove(i)
            df_for_llm = df_for_llm.iloc[:, keep]

        res = summarize_tweets(
            df=df_for_llm,
            start_date=start_iso,
            end_date=end_iso,
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
@_alias("/themes")
def themes(
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD or MM/DD/YYYY"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD or MM/DD/YYYY"),
    n_clusters: Optional[int] = Query(default=None, ge=1, le=8),
    emb_model: str = Query(default="sentence-transformers/all-MiniLM-L6-v2"),
    merge_similar: bool = Query(default=True),
    parquet: Optional[str] = Query(default=None),
    max_rows: Optional[int] = Query(default=None),
):
    key = (start, end, n_clusters or 0, emb_model)
    if key in _THEMES_CACHE:
        del _THEMES_CACHE[key]

    try:
        start_iso = _parse_date_str(start) if start else None
        end_iso = _parse_date_str(end) if end else None

        df_raw = pd.read_parquet(parquet or RAW_TWEETS_PATH)
        if isinstance(max_rows, int) and max_rows > 0:
            df_raw = df_raw.head(max_rows)

        payload = compute_themes_payload(
            df=df_raw,
            start_date=start_iso,
            end_date=end_iso,
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
                "trace_tail": tb.splitlines()[-6:],
            },
        )


# --- Tweets drill-down for a theme ---
@_alias("/themes/{theme_id}/tweets")
def theme_tweets(
    theme_id: int,
    limit: int = Query(default=10, ge=1, le=200),
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD or MM/DD/YYYY"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD or MM/DD/YYYY"),
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
            df3 = df3[df3[date_col] >= pd.to_datetime(_parse_date_str(start))]
        if end:
            df3 = df3[df3[date_col] <= pd.to_datetime(_parse_date_str(end))]

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
    for col in ["text", "clean_tweet", "text_used", "fulltext", "tweet_text"]:
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


# ---------------- Reports ----------------
@_alias("/reports/sentiment")
def download_sentiment_report(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    format: str = Query(default="pdf", regex="^(pdf|xlsx)$"),
):
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    if sub.empty:
        return {"error": "No data found for the specified date range"}

    sentiment_data = _sentiment_summary(sub)

    if format == "xlsx":
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            summary_df = pd.DataFrame([
                {"Metric": "Total Tweets", "Count": sentiment_data["total"]},
                {"Metric": "Positive", "Count": sentiment_data["counts"]["positive"], "Percentage": sentiment_data["percent"]["positive"]},
                {"Metric": "Neutral", "Count": sentiment_data["counts"]["neutral"], "Percentage": sentiment_data["percent"]["neutral"]},
                {"Metric": "Negative", "Count": sentiment_data["counts"]["negative"], "Percentage": sentiment_data["percent"]["negative"]},
            ])
            summary_df.to_excel(writer, sheet_name="Summary", index=False)

        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=sentiment_report_{start}_to_{end}.xlsx"},
        )

    html_content = f"""
    <html>
    <head><title>Sentiment Analysis Report</title></head>
    <body>
        <h1>Sentiment Analysis Report</h1>
        <p>Date Range: {start} to {end}</p>
        <h2>Summary</h2>
        <p>Total Tweets: {sentiment_data['total']}</p>
        <p>Positive: {sentiment_data['counts']['positive']} ({sentiment_data['percent']['positive']}%)</p>
        <p>Neutral: {sentiment_data['counts']['neutral']} ({sentiment_data['percent']['neutral']}%)</p>
        <p>Negative: {sentiment_data['counts']['negative']} ({sentiment_data['percent']['negative']}%)</p>
    </body>
    </html>
    """
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename=sentiment_report_{start}_to_{end}.html"},
    )


@_alias("/reports/aspects")
def download_aspect_report(
    start: date = Query(default=ASPECT_MIN_DATE),
    end: date = Query(default=ASPECT_MAX_DATE),
    format: str = Query(default="pdf", regex="^(pdf|xlsx)$"),
):
    if adf.empty:
        return {"error": "No aspect data available. Run Stage 2 first."}

    mask = (adf["date_only"] >= start) & (adf["date_only"] <= end)
    sub = adf.loc[mask]
    if sub.empty:
        return {"error": "No data found for the specified date range"}

    dom_counts = sub["aspect_dominant"].value_counts().to_dict()
    labels = ASPECTS.copy()
    counts_list = [int(dom_counts.get(a, 0)) for a in labels]
    total = sum(counts_list) or 1
    perc_list = [round(c / total * 100, 2) for c in counts_list]

    if format == "xlsx":
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            aspect_df = pd.DataFrame({
                "Aspect": labels,
                "Count": counts_list,
                "Percentage": perc_list,
            })
            aspect_df.to_excel(writer, sheet_name="Aspect Summary", index=False)

        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=aspect_report_{start}_to_{end}.xlsx"},
        )

    html = f"""
    <html>
      <head><title>Aspect Analysis Report</title></head>
      <body>
        <h1>Aspect Analysis Report</h1>
        <p>Date Range: {start} to {end}</p>
        <h2>Aspect Breakdown</h2>
        <ul>
          {''.join([f"<li>{labels[i]}: {counts_list[i]} ({perc_list[i]}%)</li>" for i in range(len(labels))])}
        </ul>
      </body>
    </html>
    """
    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename=aspect_report_{start}_to_{end}.html"},
    )


@_alias("/reports/themes")
def download_theme_report(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    format: str = Query(default="pdf", regex="^(pdf|xlsx)$"),
):
    mask = (df["date_only"] >= start) & (df["date_only"] <= end)
    sub = df.loc[mask]
    if sub.empty:
        return {"error": "No data found for the specified date range"}

    html_content = f"""
    <html>
    <head><title>Theme Analysis Report</title></head>
    <body>
        <h1>Theme Analysis Report</h1>
        <p>Date Range: {start} to {end}</p>
        <p>Total Tweets Analyzed: {len(sub)}</p>
        <h2>Note</h2>
        <p>Theme analysis requires AI processing. Please use the Theme Analysis page to generate themes first.</p>
    </body>
    </html>
    """
    return Response(
        content=html_content,
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename=theme_report_{start}_to_{end}.html"},
    )


@_alias("/reports/dashboard")
def download_dashboard_report(
    start: date = Query(default=SENT_MIN_DATE),
    end: date = Query(default=SENT_MAX_DATE),
    format: str = Query(default="pdf", regex="^(pdf|xlsx)$"),
):
    senti = sentiment_summary(start=start, end=end)
    asp = aspects_summary(start=start, end=end, as_percent=False)

    if format == "xlsx":
        excel_buffer = BytesIO()
        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
            pd.DataFrame([
                {"metric": "total", "value": senti.get("total", 0)},
                {"metric": "positive", "value": senti["counts"]["positive"], "percent": senti["percent"]["positive"]},
                {"metric": "neutral", "value": senti["counts"]["neutral"], "percent": senti["percent"]["neutral"]},
                {"metric": "negative", "value": senti["counts"]["negative"], "percent": senti["percent"]["negative"]},
            ]).to_excel(writer, sheet_name="Sentiment", index=False)

            pd.DataFrame([
                {"aspect": a, "count": asp["counts"].get(a, 0), "percent": asp["percent"].get(a, 0.0)}
                for a in asp.get("labels", [])
            ]).to_excel(writer, sheet_name="Aspects", index=False)

        return Response(
            content=excel_buffer.getvalue(),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={"Content-Disposition": f"attachment; filename=dashboard_{start}_to_{end}.xlsx"},
        )

    html = f"""
    <html>
      <head><title>Dashboard Report</title></head>
      <body>
        <h1>Analytics Dashboard Report</h1>
        <p><b>Date Range:</b> {start} to {end}</p>

        <h2>Sentiment Summary</h2>
        <ul>
          <li>Total: {senti.get("total", 0)}</li>
          <li>Positive: {senti["counts"]["positive"]} ({senti["percent"]["positive"]}%)</li>
          <li>Neutral: {senti["counts"]["neutral"]} ({senti["percent"]["neutral"]}%)</li>
          <li>Negative: {senti["counts"]["negative"]} ({senti["percent"]["negative"]}%)</li>
        </ul>

        <h2>Aspect Summary</h2>
        <ul>
          {''.join([f"<li>{a}: {asp['counts'].get(a,0)} ({asp['percent'].get(a,0.0)}%)</li>" for a in asp.get("labels",[])])}
        </ul>
      </body>
    </html>
    """
    return Response(
        content=html,
        media_type="text/html",
        headers={"Content-Disposition": f"attachment; filename=dashboard_{start}_to_{end}.html"},
    )


@_alias("/reports/theme/{theme_id}")
def download_theme_tweets_report(
    theme_id: int,
    start: Optional[str] = Query(default=None, description="YYYY-MM-DD or MM/DD/YYYY"),
    end: Optional[str] = Query(default=None, description="YYYY-MM-DD or MM/DD/YYYY"),
    limit: int = Query(default=200, ge=1, le=1000),
):
    try:
        themes_df = pd.read_parquet(STAGE3_THEMES_PARQUET)
        if "theme" not in themes_df.columns:
            return {"error": "Stage-3 themes parquet missing 'theme' column."}

        theme_tweets = themes_df[themes_df["theme"] == theme_id].copy()
        if theme_tweets.empty:
            return {"error": f"No tweets found for theme {theme_id}"}

        if "createdat" in theme_tweets.columns:
            theme_tweets["createdat"] = pd.to_datetime(theme_tweets["createdat"], errors="coerce")
            if start:
                theme_tweets = theme_tweets[theme_tweets["createdat"] >= pd.to_datetime(_parse_date_str(start))]
            if end:
                theme_tweets = theme_tweets[theme_tweets["createdat"] <= pd.to_datetime(_parse_date_str(end))]

        if theme_tweets.empty:
            return {"error": f"No tweets found for theme {theme_id} in the specified date range"}

        sentiment_counts = theme_tweets["sentiment_label"].value_counts()
        total_tweets = int(len(theme_tweets))

        theme_tweets_display = theme_tweets.head(limit)

        positive_count = int(sentiment_counts.get("positive", 0))
        negative_count = int(sentiment_counts.get("negative", 0))
        neutral_count = int(sentiment_counts.get("neutral", 0))

        positive_pct = round((positive_count / total_tweets) * 100, 1) if total_tweets else 0
        negative_pct = round((negative_count / total_tweets) * 100, 1) if total_tweets else 0
        neutral_pct = round((neutral_count / total_tweets) * 100, 1) if total_tweets else 0

        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Theme Report: Theme {theme_id}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f0f0f0; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .tweet {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 8px; }}
                .tweet.positive {{ border-left: 4px solid #28a745; }}
                .tweet.negative {{ border-left: 4px solid #dc3545; }}
                .tweet.neutral {{ border-left: 4px solid #ffc107; }}
                .sentiment {{ font-weight: bold; padding: 4px 8px; border-radius: 4px; }}
                .positive {{ background: #d4edda; color: #155724; }}
                .negative {{ background: #f8d7da; color: #721c24; }}
                .neutral {{ background: #fff3cd; color: #856404; }}
                .date {{ color: #666; font-size: 0.9em; }}
                .aspect {{ background: #e9ecef; padding: 2px 6px; border-radius: 3px; font-size: 0.8em; }}
                .download-btn {{
                    background: #007bff; color: white; padding: 10px 20px; border: none;
                    border-radius: 5px; cursor: pointer; font-size: 14px; margin-bottom: 20px;
                }}
                .download-btn:hover {{ background: #0056b3; }}
                .sentiment-breakdown {{
                    background: #f8f9fa; padding: 15px; border-radius: 8px; margin-bottom: 20px;
                    border-left: 4px solid #007bff;
                }}
                .sentiment-stats {{
                    display: flex; justify-content: space-around; margin-top: 10px;
                }}
                .sentiment-stat {{
                    text-align: center; padding: 10px; border-radius: 5px;
                }}
                .sentiment-stat.positive {{ background: #d4edda; }}
                .sentiment-stat.negative {{ background: #f8d7da; }}
                .sentiment-stat.neutral {{ background: #fff3cd; }}
                @media print {{ .download-btn {{ display: none; }} }}
            </style>
        </head>
        <body>
            <button class="download-btn" onclick="window.print()">📄 Print/Download PDF</button>

            <div class="header">
                <h1>Theme Report: Theme {theme_id}</h1>
                <p><strong>Total Tweets:</strong> {total_tweets}</p>
                <p><strong>Date Range:</strong> {start or 'All time'} to {end or 'All time'}</p>
                <p><strong>Generated:</strong> {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>

            <div class="sentiment-breakdown">
                <h3>Sentiment Analysis</h3>
                <div class="sentiment-stats">
                    <div class="sentiment-stat positive">
                        <strong>Positive</strong><br>
                        {positive_count} tweets<br>
                        <strong>{positive_pct}%</strong>
                    </div>
                    <div class="sentiment-stat negative">
                        <strong>Negative</strong><br>
                        {negative_count} tweets<br>
                        <strong>{negative_pct}%</strong>
                    </div>
                    <div class="sentiment-stat neutral">
                        <strong>Neutral</strong><br>
                        {neutral_count} tweets<br>
                        <strong>{neutral_pct}%</strong>
                    </div>
                </div>
            </div>

            <h2>Tweets</h2>
            <p><em>Showing first {min(limit, len(theme_tweets_display))} tweets out of {total_tweets}</em></p>
        """

        for _, tweet in theme_tweets_display.iterrows():
            sentiment = str(tweet.get("sentiment_label", "neutral"))
            aspect = str(tweet.get("aspect_dominant", "unknown"))
            tweet_text = tweet.get("text") or tweet.get("text_clean") or tweet.get("clean_tweet") or "No text available"
            tweet_date = tweet.get("createdat") or tweet.get("date") or "Unknown date"

            html_content += f"""
            <div class="tweet {sentiment}">
                <div style="margin-bottom: 10px;">
                    <span class="sentiment {sentiment}">{sentiment.title()}</span>
                    <span class="aspect">{aspect}</span>
                    <span class="date">{tweet_date}</span>
                </div>
                <p>{tweet_text}</p>
            </div>
            """

        html_content += """
        </body>
        </html>
        """

        return Response(
            content=html_content,
            media_type="text/html",
            headers={"Content-Disposition": f"attachment; filename=theme_{theme_id}_report_{start or 'all'}_to_{end or 'all'}.html"},
        )

    except Exception as e:
        tb = traceback.format_exc()
        return JSONResponse(status_code=500, content={"error": str(e), "trace_tail": tb.splitlines()[-6:]})
