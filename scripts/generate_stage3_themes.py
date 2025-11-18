#!/usr/bin/env python3
"""
Generate Stage 3 themes parquet file from Stage 2 aspects parquet.
This script creates tweets_stage3_themes.parquet which is required for the /api/themes/:id/tweets endpoint.
"""
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.themes import compute_themes_payload
from dotenv import load_dotenv

def main():
    load_dotenv()  # Load .env file for OPENAI_API_KEY if available
    
    parser = argparse.ArgumentParser(
        description="Generate Stage 3 themes parquet file from Stage 2 aspects parquet"
    )
    parser.add_argument(
        "--stage2",
        default="data/tweets_stage2_aspects.parquet",
        help="Path to Stage 2 aspects parquet file (default: data/tweets_stage2_aspects.parquet)"
    )
    parser.add_argument(
        "--n-clusters",
        type=int,
        default=6,
        help="Number of themes/clusters to generate (default: 6)"
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use (default: sentence-transformers/all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date filter (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--no-merge",
        action="store_true",
        help="Disable merging of similar themes"
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Maximum number of tweets to use when clustering"
    )
    args = parser.parse_args()

    # Check if Stage 2 file exists
    if not os.path.exists(args.stage2):
        print(f"❌ Error: Stage 2 parquet file not found: {args.stage2}")
        print("\n📋 To generate the required files, you need to run the pipeline in order:")
        print("   1. Stage 0: Convert raw data to parquet (scripts/save_to_parquet.py)")
        print("   2. Stage 1: Run sentiment analysis (scripts/run_sentiment_stage1.py)")
        print("   3. Stage 2: Run aspect analysis (scripts/run_aspects_stage2.py)")
        print("   4. Stage 3: Generate themes (this script)")
        print(f"\n💡 Looking for: {os.path.abspath(args.stage2)}")
        return 1

    print(f"✅ Found Stage 2 file: {args.stage2}")
    print(f"📊 Generating themes with {args.n_clusters} clusters...")
    print(f"🤖 Using model: {args.model}")
    
    # Get OpenAI API key from environment
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        print("⚠️  Warning: OPENAI_API_KEY not found in environment.")
        print("   Theme names and summaries will use TF-IDF keywords instead of LLM-generated content.")
    else:
        print("✅ OpenAI API key found. Will generate LLM-powered theme names and summaries.")

    try:
        # Generate themes parquet file
        payload = compute_themes_payload(
            df=None,  # Will load from parquet_stage2
            parquet_stage2=args.stage2,
            n_clusters=args.n_clusters,
            emb_model=args.model,
            start_date=args.start_date,
            end_date=args.end_date,
            openai_api_key=openai_key,
            merge_similar=not args.no_merge,
            max_rows=args.max_rows,
        )

        # Check if parquet file was created
        parquet_path = "data/tweets_stage3_themes.parquet"
        if os.path.exists(parquet_path):
            import pandas as pd
            df = pd.read_parquet(parquet_path)
            print(f"\n✅ Success! Generated parquet file: {parquet_path}")
            print(f"   📊 Total rows: {len(df):,}")
            print(f"   🏷️  Themes: {df['theme'].nunique() if 'theme' in df.columns else 'N/A'}")
            print(f"\n📋 Theme summary:")
            for theme in payload.get("themes", [])[:5]:  # Show first 5 themes
                print(f"   Theme {theme['id']}: {theme['name']} ({theme['tweet_count']} tweets)")
            if len(payload.get("themes", [])) > 5:
                print(f"   ... and {len(payload.get('themes', [])) - 5} more themes")
            
            print(f"\n🎉 The parquet file is ready! Your server can now serve tweet details.")
            return 0
        else:
            print(f"❌ Error: Parquet file was not created at {parquet_path}")
            return 1

    except Exception as e:
        print(f"❌ Error generating themes: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

