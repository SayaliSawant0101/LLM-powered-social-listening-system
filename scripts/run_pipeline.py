
# Example pipeline runner — call your functions here in order
# Load → Clean → Sentiment → ABSA → Topics → Save

from src.features.clean import load_any, basic_date_parse
from src.features.sentiment import ensure_sentiment
from src.features.absa import ensure_absa
from src.features.topics import ensure_topics

from src.utils.text import torch_device_name

print(f"[Setup] device: {torch_device_name()}")

def run(path_in: str, path_out: str):
    import pandas as pd
    df = load_any(path_in)
    df = basic_date_parse(df)
    df = ensure_sentiment(df)
    df = ensure_absa(df)
    df = ensure_topics(df)
    df.to_parquet(path_out, index=False)
    print(f"Pipeline complete → {path_out} ({len(df):,} rows)")

if __name__ == "__main__":
    run("data/tweets_clean.parquet", "data/tweets_enriched.parquet")
