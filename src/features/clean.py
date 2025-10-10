
from __future__ import annotations
import pandas as pd

def load_any(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    if path.endswith(".xlsx") or path.endswith(".xls"):
        return pd.read_excel(path)
    if path.endswith(".csv"):
        return pd.read_csv(path)
    raise ValueError(f"Unsupported file type: {path}")

def basic_date_parse(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["created_at", "date", "timestamp"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True).dt.tz_convert(None)
            if "date" not in df.columns:
                df["date"] = df[col].dt.date
            return df
    raise KeyError("No date-like column found. Expected one of: created_at, date, timestamp")
