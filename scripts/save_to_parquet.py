# scripts/save_to_parquet.py
import os, sys
sys.path.insert(0, os.getcwd())

from src.features.clean import load_from_athena, basic_date_parse

def main():
    print("[Stage 0] Fetching data from Athena...")
    df = load_from_athena()
    
    print("[Stage 0] Parsing dates...")
    try:
        df = basic_date_parse(df)
    except Exception as e:
        print(f"[Warning] Could not parse dates automatically: {e}")
    
    from src.features.clean import standardize_tweet_columns
    df = standardize_tweet_columns(df)
    
    # Save to Parquet
    os.makedirs("data", exist_ok=True)
    out_path = "data/tweets_stage0_raw.parquet"
    df.to_parquet(out_path, index=False)
    print(f"[✅ Done] Saved {len(df):,} rows → {out_path}")

if __name__ == "__main__":
    main()
