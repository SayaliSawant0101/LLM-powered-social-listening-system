import pandas as pd
import numpy as np
import os

ASPECT_PATH = "/content/walmart_social_listener/data/tweets_stage2_aspects.parquet"
OUT_PATH    = "/content/walmart_social_listener/data/tweets_stage3_aspect_sentiment.parquet"

assert os.path.exists(ASPECT_PATH), f"File not found: {ASPECT_PATH}"

print(f"[Stage 3] Loading aspects file: {ASPECT_PATH}")
df = pd.read_parquet(ASPECT_PATH)

# Ensure needed columns exist
required_cols = {"aspect_dominant", "sentiment_label"}
missing = required_cols - set(df.columns)
if missing:
    raise KeyError(f"Missing required columns: {missing}")

# Clean labels
df["aspect_dominant"] = df["aspect_dominant"].fillna("none").str.lower()
df["sentiment_label"] = df["sentiment_label"].fillna("neutral").str.lower()

# Aggregate counts by aspect × sentiment
grouped = (
    df.groupby(["aspect_dominant", "sentiment_label"])
      .size()
      .reset_index(name="count")
)

# Pivot so each sentiment is a column
pivot = (
    grouped.pivot(index="aspect_dominant", columns="sentiment_label", values="count")
    .fillna(0)
    .reset_index()
)
pivot["total"] = pivot[["positive", "neutral", "negative"]].sum(axis=1)
pivot = pivot.sort_values("total", ascending=False)

# Compute percentages
for s in ["positive", "neutral", "negative"]:
    pivot[f"{s}_pct"] = (pivot[s] / pivot["total"] * 100).round(2)

print("\n[Stage 3] Aspect × Sentiment Split (sample):")
print(pivot.head())

pivot.to_parquet(OUT_PATH, index=False)
print(f"\n[Stage 3] Saved aggregated file: {OUT_PATH}")
