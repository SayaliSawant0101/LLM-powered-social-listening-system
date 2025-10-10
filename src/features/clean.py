# src/features/clean.py
from __future__ import annotations
import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

# Optional stage filenames
RAW_PARQUET    = "tweets_stage0_raw.parquet"
SENTI_PARQUET  = "tweets_stage1_sentiment.parquet"
ASPECT_PARQUET = "tweets_stage2_aspects.parquet"
THEME_PARQUET  = "tweets_stage3_themes.parquet"

ASPECTS = ["pricing", "delivery", "returns", "staff", "app/ux"]


# ---------------------- Utility Functions ----------------------
def parse_dates_series(s: pd.Series) -> pd.Series:
    """Convert any date-like column to timezone-aware datetime."""
    s = pd.to_datetime(s, errors="coerce", utc=True)
    try:
        s = s.dt.tz_convert("US/Eastern")
    except Exception:
        s = s.dt.tz_localize(None)
    return s


def basic_date_parse(df: pd.DataFrame, date_col: str | None = None) -> pd.DataFrame:
    """
    Make a canonical datetime column and a 'date' (date-only) column.
    - Tries many common column names if `date_col` isn't provided.
    - Handles ISO strings and epoch (seconds/milliseconds).
    - Result:
        * df[<chosen_col>] -> timezone-naive local datetime
        * df['date']       -> date() extracted from that
    """
    # 1) If caller told us which column to use
    if date_col and date_col in df.columns:
        col = date_col
        s = df[col]
        # numeric epoch?
        if pd.api.types.is_numeric_dtype(s):
            # heuristic: seconds vs milliseconds
            # (timestamps > 10^11 are almost always ms)
            unit = "ms" if s.dropna().astype("float64").median() > 1e11 else "s"
            parsed = pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(s, errors="coerce", utc=True)
        try:
            parsed = parsed.dt.tz_convert("US/Eastern").dt.tz_localize(None)
        except Exception:
            parsed = parsed.dt.tz_localize(None)
        df[col] = parsed
        df["date"] = parsed.dt.date
        return df

    # 2) Try common/likely names
    common_candidates = [
        "created_at", "createdAt", "created_time", "creation_time",
        "timestamp", "time", "post_time", "posted_at",
        "tweet_created_at", "tweet_time",
        "date", "datetime", "Date", "DateTime", "DATE",
        "created_utc", "epoch", "unix", "unix_time",
    ]

    # add any col that *contains* date/time in its name
    lowered = {c.lower(): c for c in df.columns}
    for k in list(lowered):
        if "date" in k or "time" in k or "created" in k:
            if lowered[k] not in common_candidates:
                common_candidates.append(lowered[k])

    # 3) Try to parse each candidate until one works well
    for cand in common_candidates:
        if cand not in df.columns:
            continue
        s = df[cand]

        # numeric epoch?
        if pd.api.types.is_numeric_dtype(s):
            unit = "ms" if s.dropna().astype("float64").median() > 1e11 else "s"
            parsed = pd.to_datetime(s, unit=unit, errors="coerce", utc=True)
        else:
            parsed = pd.to_datetime(s, errors="coerce", utc=True)

        # if too many NaT, skip
        if parsed.notna().sum() < max(10, int(0.5 * len(parsed))):
            continue

        try:
            parsed = parsed.dt.tz_convert("US/Eastern").dt.tz_localize(None)
        except Exception:
            parsed = parsed.dt.tz_localize(None)

        df[cand] = parsed
        df["date"] = parsed.dt.date
        return df

    # 4) Nothing worked → help the user
    raise KeyError(
        "No date-like column found. Pass basic_date_parse(df, date_col='your_column') "
        f"or rename one of these columns to a standard name. Columns present: {list(df.columns)}"
    )


def load_any(path: str) -> pd.DataFrame:
    """Read CSV, Excel, or Parquet into a DataFrame."""
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")


# ---------------------- Athena Loader ----------------------
def load_from_athena(
    sql: str | None = None,
    database: str | None = None,
    workgroup: str | None = None,
    s3_staging_dir: str | None = None,
    aws_region: str | None = None,
) -> pd.DataFrame:
    """
    Query AWS Athena and return a Pandas DataFrame.
    Uses non-CTAS path to avoid Glue create/delete permissions.
    """
    import awswrangler as wr
    import boto3
    import os

    sql = sql or os.getenv("ATHENA_SQL")
    if not sql:
        raise ValueError("Provide SQL or set ATHENA_SQL in .env")

    database = database or os.getenv("ATHENA_SCHEMA")
    workgroup = workgroup or os.getenv("ATHENA_WORKGROUP", "primary")
    s3_output = s3_staging_dir or os.getenv("ATHENA_STAGING_DIR")
    region = aws_region or os.getenv("AWS_REGION", "us-east-1")

    boto3_session = boto3.Session(
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        aws_session_token=os.getenv("AWS_SESSION_TOKEN") or None,
        region_name=region,
    )

    print(f"[Athena] Running (non-CTAS) on db='{database}', wg='{workgroup}'")
    print(f"[Athena] SQL: {sql[:100]}...")

    df = wr.athena.read_sql_query(
        sql=sql,
        database=database,
        boto3_session=boto3_session,
        s3_output=s3_output,
        workgroup=workgroup,
        ctas_approach=False,        # 👈 key line: no Glue temp table
        use_threads=True,           # parallel download
        chunksize=None,             # return a single DataFrame
    )
    print(f"[Athena] Retrieved {len(df):,} rows.")
    return df

def standardize_tweet_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize and clean tweet fields for downstream NLP stages.
    - Picks a consistent text column (prefers text_clean > text > fulltext)
    - Fills NaN text values with ""
    - Ensures presence of basic indicator columns
    """
    # Choose text column
    candidates = ["text_clean", "text", "fulltext", "tweet", "tweet_text", "message"]
    text_col = next((c for c in candidates if c in df.columns), None)
    if text_col:
        df["clean_tweet"] = df[text_col].fillna("").astype(str)
    else:
        raise KeyError(f"No text column found. Looked for {candidates}")

    # Add missing default columns
    defaults = {
        "has_url": False,
        "has_mention": False,
        "has_hashtag": False,
        "lang": "en",
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    # Optional: reorder columns to keep things tidy
    main_cols = ["clean_tweet", "date", "lang", "has_url", "has_mention", "has_hashtag"]
    remaining = [c for c in df.columns if c not in main_cols]
    df = df[main_cols + remaining]

    return df
