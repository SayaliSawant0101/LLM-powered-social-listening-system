import argparse
import pandas as pd
from src.features.clean import load_any, basic_date_parse
from src.llm.summary import summarize_tweets

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to CSV/XLSX/Parquet with tweets")
    ap.add_argument("--start", required=False, help="YYYY-MM-DD")
    ap.add_argument("--end", required=False, help="YYYY-MM-DD")
    ap.add_argument("--keyword", required=False)
    ap.add_argument("--out", default="summary_out.json", help="Path to write JSON")
    args = ap.parse_args()

    df = load_any(args.data)
    df = basic_date_parse(df)
    res = summarize_tweets(df, start_date=args.start, end_date=args.end, keyword=args.keyword)
    import json
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
