from __future__ import annotations
import os, time
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

def pick_device() -> str:
    if os.getenv("FORCE_CPU", "").lower() == "true":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

def build_pipeline(model_name: str, device_str: str):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        use_safetensors=True,      # avoid .bin path
        torch_dtype=torch.float32, # strict fp32 = stable kernels
        low_cpu_mem_usage=False    # avoid aggressive offload tricks
    ).eval()

    device_flag = -1
    if device_str == "cuda":
        mdl.to("cuda")
        device_flag = 0  # GPU index 0

    return TextClassificationPipeline(
        model=mdl,
        tokenizer=tok,
        device=device_flag,
        return_all_scores=True,
        truncation=True,
    ), mdl.config.id2label

def run_sentiment(
    df: pd.DataFrame,
    text_col: str = "clean_tweet",
    model_name: str = "cardiffnlp/twitter-roberta-base-sentiment-latest",
    batch_size: int | None = None,
    max_length: int = 256,
) -> pd.DataFrame:
    assert text_col in df.columns, f"Text column '{text_col}' not found."
    device_str = pick_device()

    # conservative defaults for GPU stability
    if batch_size is None:
        batch_size = 16 if device_str == "cuda" else 8
    max_length = min(max_length, 256)  # keep it small and consistent

    pipe, id2label = build_pipeline(model_name, device_str)

    texts = df[text_col].astype(str).tolist()
    labels, scores = [], []
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        if i == 0 or i % (batch_size * 50) == 0:
            print(f"[Stage 1] {i}/{len(texts)} | device={device_str} | elapsed={time.time()-t0:.1f}s")
        batch = texts[i:i+batch_size]
        # strict truncation on every call
        results = pipe(
            batch,
            batch_size=batch_size,
            max_length=max_length,
            truncation=True,
            return_all_scores=True,
        )
        for row in results:
            best = max(row, key=lambda x: x["score"])
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
