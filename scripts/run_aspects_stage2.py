import os, sys, argparse
import pandas as pd

sys.path.insert(0, os.getcwd())
from src.features.aspects import run_aspects, DEFAULT_ASPECTS

DATA_DIR        = "data"
SENTI_PARQUET   = os.path.join(DATA_DIR, "tweets_stage1_sentiment.parquet")
ASPECT_PARQUET  = os.path.join(DATA_DIR, "tweets_stage2_aspects.parquet")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="FAST_PREVIEW: skip modeling, fill zeros")
    ap.add_argument("--model", default="facebook/bart-large-mnli", help="Zero-shot model")
    ap.add_argument("--batch", type=int, default=None, help="Batch size override")
    ap.add_argument("--thr", type=float, default=0.5, help="Dominant aspect threshold")
    ap.add_argument("--text-col", default="clean_tweet", help="Input text column")
    ap.add_argument("--out", default=ASPECT_PARQUET, help="Output parquet path")
    args = ap.parse_args()

    assert os.path.exists(SENTI_PARQUET), f"Missing Stage 1 parquet: {SENTI_PARQUET}"
    df = pd.read_parquet(SENTI_PARQUET)
    print(f"[Stage 2] Starting on {len(df):,} rows | fast={args.fast}")

    df = run_aspects(
        df,
        text_col=args.text_col,
        aspects=DEFAULT_ASPECTS,
        model_name=args.model,
        batch_size=args.batch,
        fast_preview=args.fast,
        score_threshold=args.thr,
    )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[Stage 2] Saved → {args.out}")

if __name__ == "__main__":
    main()
