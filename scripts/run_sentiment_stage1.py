# scripts/run_sentiment_stage1.py
import os, sys, pandas as pd
sys.path.insert(0, os.getcwd())

from src.features.clean import SENTI_PARQUET
from src.features.sentiment import run_sentiment

RAW_PATH = "data/tweets_stage0_raw.parquet"  # created by save_to_parquet.py

def main():
    assert os.path.exists(RAW_PATH), f"Run save_to_parquet.py first. Missing: {RAW_PATH}"
    df = pd.read_parquet(RAW_PATH)
    print(f"[Stage 1] Starting sentiment on {len(df):,} rows")

    df = run_sentiment(df, text_col="clean_tweet")
    os.makedirs("data", exist_ok=True)
    out = f"data/{SENTI_PARQUET}" if not SENTI_PARQUET.startswith("data/") else SENTI_PARQUET
    df.to_parquet(out, index=False)

    print("[Stage 1] Sentiment counts:")
    print(df["sentiment_label"].value_counts())
    print(f"[Stage 1] Saved {out}")

if __name__ == "__main__":
    main()
