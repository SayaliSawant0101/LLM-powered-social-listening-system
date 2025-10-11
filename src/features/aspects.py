from __future__ import annotations
import os, time
import numpy as np
import pandas as pd
import torch
from typing import List
from transformers import pipeline

DEFAULT_ASPECTS: List[str] = ["pricing", "delivery", "returns", "staff", "app/ux"]

def pick_device_str() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def run_aspects(
    df: pd.DataFrame,
    text_col: str = "clean_tweet",
    aspects: List[str] = None,
    model_name: str = "facebook/bart-large-mnli",
    batch_size: int | None = None,
    fast_preview: bool = False,
    score_threshold: float = 0.5,
) -> pd.DataFrame:
    if aspects is None:
        aspects = DEFAULT_ASPECTS
    assert text_col in df.columns, f"Text column '{text_col}' not found."

    if fast_preview:
        for a in aspects:
            df[f"aspect_{a}"] = 0.0
        df["aspect_dominant"] = "none"
        return df

    device = pick_device_str()
    device_flag = 0 if device == "cuda" else -1
    model_kwargs = {"torch_dtype": torch.float16} if device == "cuda" else {}

    zpipe = pipeline(
        task="zero-shot-classification",
        model=model_name,
        device=device_flag,
        model_kwargs=model_kwargs,
    )

    texts = df[text_col].astype(str).tolist()
    if batch_size is None:
        batch_size = 24 if device == "cuda" else 8

    scores_per_aspect = {a: [] for a in aspects}
    t0 = time.time()

    for i in range(0, len(texts), batch_size):
        if i == 0 or i % (batch_size * 20) == 0:
            print(f"[Stage 2] Aspect progress: {i}/{len(texts)} | device={device} | elapsed={time.time()-t0:.1f}s")
        batch = texts[i:i+batch_size]
        preds = zpipe(batch, candidate_labels=aspects, multi_label=True, truncation=True)
        if isinstance(preds, dict):
            preds = [preds]
        for p in preds:
            l2s = dict(zip(p["labels"], p["scores"]))
            for a in aspects:
                scores_per_aspect[a].append(float(l2s.get(a, 0.0)))

    for a in aspects:
        df[f"aspect_{a}"] = scores_per_aspect[a]

    arr = df[[f"aspect_{a}" for a in aspects]].values
    idxmax = arr.argmax(axis=1)
    maxval = arr.max(axis=1)
    dom = np.where(maxval >= score_threshold, np.array(aspects)[idxmax], "none")
    df["aspect_dominant"] = dom

    return df
