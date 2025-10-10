
from __future__ import annotations
import os
import pandas as pd
from datetime import datetime, date
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

# Official OpenAI SDK
# Primary: Responses API (recommended)
# Ref: https://github.com/openai/openai-python
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")

def _fmt_date(d: Any) -> str:
    if isinstance(d, (datetime, date)):
        return d.isoformat()
    return str(d)

def _basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    out = {"n_tweets": int(len(df))}
    if "sentiment_label" in df.columns:
        counts = df["sentiment_label"].value_counts(dropna=False).to_dict()
        out["sentiment_breakdown"] = {str(k): int(v) for k, v in counts.items()}
    if "aspect" in df.columns:
        top_aspects = df["aspect"].value_counts().head(10).index.tolist()
        out["top_aspects"] = [str(a) for a in top_aspects]
    if "topic_keywords" in df.columns:
        # pick most frequent topic keywords
        top_kw = (
            df["topic_keywords"]
              .fillna("")
              .replace("", pd.NA)
              .dropna()
              .value_counts()
              .head(10)
              .index
              .tolist()
        )
        out["top_topic_keywords"] = top_kw
    return out

def build_prompt(stats: Dict[str, Any], examples: List[str], start: str, end: str, keyword: Optional[str]) -> str:
    # Keep the prompt compact but structured
    return f"""
You are an analytics copilot generating an **executive brief** of Twitter/X chatter about Walmart.

TIME WINDOW: {start} → {end}
KEYWORD FILTER: {keyword or "None"}

BASIC STATS (from analytics pipeline):
{stats}

SAMPLE TWEETS (cleaned, representative subset):
- """ + "\n- ".join(examples) + """

Write:
1) 5–8 bullet executive brief (impactful, business-ready).
2) A short paragraph on key drivers and pain points.
3) 4–6 concise theme labels (Title Case).
4) A strict JSON object with keys: executive_bullets (list[str]), themes (list[str]), risks (list[str]), opportunities (list[str]).

Rules:
- Be specific but avoid hallucinating numbers not present in data.
- Use neutral tone; no fluff.
- If the data is thin, say so explicitly.
"""

def summarize_tweets(
    df: pd.DataFrame,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    keyword: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    sample_size: int = 50,
) -> Dict[str, Any]:
    """Return a dict with 'executive_text' and 'structured' (parsed JSON-like) fields.
    Expects df with columns: 'clean_tweet' and a date column named 'created_at' or 'date'.
    """
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set. Put it in your .env or environment.")

    # Pick date col
    date_col = None
    for c in ["created_at", "date", "timestamp"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise KeyError("No date-like column. Expected one of: created_at, date, timestamp")

    dfx = df.copy()
    dfx[date_col] = pd.to_datetime(dfx[date_col], errors="coerce")
    if start_date:
        dfx = dfx[dfx[date_col] >= pd.to_datetime(start_date)]
    if end_date:
        dfx = dfx[dfx[date_col] <= pd.to_datetime(end_date)]

    if keyword:
        mask = dfx["clean_tweet"].str.contains(keyword, case=False, na=False)
        dfx = dfx[mask]

    if len(dfx) == 0:
        return {
            "executive_text": f"No tweets found for {start_date} to {end_date} keyword={keyword}",
            "structured": {"executive_bullets": [], "themes": [], "risks": [], "opportunities": []},
            "stats": {"n_tweets": 0},
        }

    # lightweight stats & examples
    stats = _basic_stats(dfx)
    samples = dfx["clean_tweet"].dropna().astype(str).sample(min(sample_size, len(dfx)), random_state=42).tolist()

    prompt = build_prompt(stats=stats, examples=samples, start=_fmt_date(start_date), end=_fmt_date(end_date), keyword=keyword)

    client = OpenAI(api_key=OPENAI_API_KEY)

    # Use Responses API (preferred going forward)
    # See: https://github.com/openai/openai-python
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": "You are a precise analytics summarizer that returns clean, business-ready insights.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_output_tokens=900,
    )

    # Unified text accessor (SDK exposes .output_text convenience attr)
    full_text = getattr(response, "output_text", None) or str(response)

    # Try to parse a JSON block if present (best-effort)
    import json, re
    json_obj = {"executive_bullets": [], "themes": [], "risks": [], "opportunities": []}
    try:
        code_blocks = re.findall(r"```json\s*(\{[\s\S]*?\})\s*```", full_text)
        if code_blocks:
            json_obj = json.loads(code_blocks[-1])
        else:
            # Fallback: try to locate a bare JSON object
            first_brace = full_text.find("{")
            last_brace = full_text.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                json_obj = json.loads(full_text[first_brace:last_brace+1])
    except Exception:
        pass

    return {
        "executive_text": full_text,
        "structured": json_obj,
        "stats": stats,
    }
