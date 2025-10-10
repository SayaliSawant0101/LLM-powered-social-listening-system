# src/features/sentiment.py
from __future__ import annotations
import time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def run_sentiment(
    df: pd.DataFrame,
    text_col: str = "clean_tweet",
    # switched default to safetensors model (3-class)
    model_name: str = "finiteautomata/bertweet-base-sentiment-analysis",
    batch_size: int | None = None,
    max_length: int = 256,
) -> pd.DataFrame:
    assert text_col in df.columns, f"Text column '{text_col}' not found."

    device_str = pick_device()
    device_flag = 0 if device_str == "cuda" else -1

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)  # safetensors by default
    if device_str == "mps":
        mdl.to("mps")

    pipe = TextClassificationPipeline(
        model=mdl,
        tokenizer=tok,
        device=device_flag,
        return_all_scores=True,
        truncation=True,
    )

    texts = df[text_col].astype(str).tolist()
    if batch_size is None:
        batch_size = 32 if device_str == "cuda" else 8  # smaller on CPU

    labels, scores = [], []
    id2label = mdl.config.id2label
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        if i == 0 or i % (batch_size * 50) == 0:
            print(f"[Stage 1] {i}/{len(texts)} | device={device_str} | elapsed={time.time()-t0:.1f}s")
        batch = texts[i:i+batch_size]
        results = pipe(batch, batch_size=batch_size, max_length=max_length, truncation=True, return_all_scores=True)
        for row in results:
            best = max(row, key=lambda x: x["score"])
            # map index-based labels if needed
            try:
                mapped = id2label[int(best["label"].split("_")[-1])]
            except Exception:
                mapped = best["label"]
            labels.append(mapped.lower())
            scores.append(float(best["score"]))

    out = df.copy()
    out["sentiment_label"] = labels
    out["sentiment_score"] = scores
    return out
