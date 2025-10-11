# api/main.py
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from datetime import date
import pandas as pd
import os

# ------------ Paths ------------
SENTI_PATH   = "data/tweets_stage1_sentiment.parquet"
ASPECT_PATH  = "data/tweets_stage2_aspects.parquet"
STAGE3_PATH  = "data/tweets_stage3_aspect_sentiment.parquet"  # optional cache (no dates)

app = FastAPI(title="Walmart Social Listener API")

# Allow calls from Vite dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
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

def _aspect_split_from_subset(sub: pd.DataFrame, aspects: list[str]) -> dict:
    """
    Build stacked-bar friendly payload from a subset that HAS columns:
      - aspect_dominant
      - sentiment_label
    Returns both counts and percents arrays (aligned to 'aspects').
    """
    if sub.empty:
        zero = [0 for _ in aspects]
        zero_f = [0.0 for _ in aspects]
        return {
            "labels": aspects,
            "counts": {
                "positive": zero, "neutral": zero, "negative": zero,
            },
            "percent": {
                "positive": zero_f, "neutral": zero_f, "negative": zero_f,
            },
        }

    # counts by aspect × sentiment
    g = (
        sub.groupby(["aspect_dominant", "sentiment_label"])
          .size().reset_index(name="count")
    )
    pivot = (
        g.pivot(index="aspect_dominant", columns="sentiment_label", values="count")
         .fillna(0)
    )
    # Ensure all sentiment columns exist
    for col in ["positive", "neutral", "negative"]:
        if col not in pivot.columns:
            pivot[col] = 0

    # Reindex to our aspect order; fill missing with 0
    pivot = pivot.reindex(aspects, fill_value=0)

    # Per-aspect totals (row-wise)
    totals = pivot.sum(axis=1).replace(0, 1)  # avoid divide-by-zero
    pct = (pivot.div(totals, axis=0) * 100).round(2)

    return {
        "labels": aspects,
        "counts": {
            "positive": [int(x) for x in pivot["positive"].tolist()],
            "neutral":  [int(x) for x in pivot["neutral"].tolist()],
            "negative": [int(x) for x in pivot["negative"].tolist()],
        },
        "percent": {
            "positive": [float(x) for x in pct["positive"].tolist()],
            "neutral":  [float(x) for x in pct["neutral"].tolist()],
            "negative": [float(x) for x in pct["negative"].tolist()],
        },
    }

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
    # API still runs; aspect endpoints return zeros.
    adf = pd.DataFrame(columns=["date_only", "aspect_dominant", "sentiment_label"])
    ASPECT_MIN_DATE = SENT_MIN_DATE
    ASPECT_MAX_DATE = SENT_MAX_DATE

# Optional cache without dates (Stage 3)
stage3_df = None
if os.path.exists(STAGE3_PATH):
    try:
        stage3_df = pd.read_parquet(STAGE3_PATH)
        # Expect columns: aspect_dominant, positive, neutral, negative, total, *_pct (no dates)
        for c in ["aspect_dominant", "positive", "neutral", "negative"]:
            assert c in stage3_df.columns
    except Exception:
        stage3_df = None

# ------------ Routes ------------
@app.get("/")
def health():
    return {
        "message": "✅ Walmart Sentiment API is running!",
        "date_range": {"min": str(SENT_MIN_DATE), "max": str(SENT_MAX_DATE)},
        "aspect_date_range": {"min": str(ASPECT_MIN_DATE), "max": str(ASPECT_MAX_DATE)},
        "has_aspects": bool(len(adf) > 0),
        "has_stage3_cache": bool(stage3_df is not None),
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
):
    """
    Returns stacked-bar data by aspect:
      - labels: aspect names
      - counts: {positive: [], neutral: [], negative: []}
      - percent: same but in %
    If no start/end provided and stage-3 cache exists (no dates), uses it.
    Otherwise computes from Stage-2 (with filter).
    """
    # Fast path: cache without dates (whole period)
    if start is None and end is None and stage3_df is not None:
        # ensure order and presence
        s3 = stage3_df.set_index("aspect_dominant").reindex(ASPECTS, fill_value=0)
        for c in ["positive", "neutral", "negative"]:
            if c not in s3.columns:
                s3[c] = 0
        totals = s3[["positive", "neutral", "negative"]].sum(axis=1).replace(0, 1)
        pct = (s3[["positive", "neutral", "negative"]].div(totals, axis=0) * 100).round(2)

        payload = {
            "labels": ASPECTS,
            "counts": {
                "positive": [int(x) for x in s3["positive"].tolist()],
                "neutral":  [int(x) for x in s3["neutral"].tolist()],
                "negative": [int(x) for x in s3["negative"].tolist()],
            },
            "percent": {
                "positive": [float(x) for x in pct["positive"].tolist()],
                "neutral":  [float(x) for x in pct["neutral"].tolist()],
                "negative": [float(x) for x in pct["negative"].tolist()],
            },
        }
        return payload if as_percent is False else payload  # UI will choose which series to plot

    # Compute from Stage-2 with date filtering
    s = start or ASPECT_MIN_DATE
    e = end or ASPECT_MAX_DATE
    if adf.empty:
        return _aspect_split_from_subset(pd.DataFrame(), ASPECTS)

    mask = (adf["date_only"] >= s) & (adf["date_only"] <= e)
    sub = adf.loc[mask, ["aspect_dominant", "sentiment_label"]]
    return _aspect_split_from_subset(sub, ASPECTS)
