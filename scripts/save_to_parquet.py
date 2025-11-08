# scripts/save_to_parquet.py
import os, sys
sys.path.insert(0, os.getcwd())

from dotenv import load_dotenv
load_dotenv()  # Load .env file

from src.features.clean import load_from_athena, basic_date_parse
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Fetch data from AWS Athena and save as Stage 0 parquet file"
    )
    parser.add_argument(
        "--sql",
        default=None,
        help="SQL query to execute (overrides ATHENA_SQL from .env)"
    )
    parser.add_argument(
        "--database",
        default=None,
        help="Athena database name (overrides ATHENA_SCHEMA from .env)"
    )
    parser.add_argument(
        "--workgroup",
        default=None,
        help="Athena workgroup (overrides ATHENA_WORKGROUP from .env)"
    )
    parser.add_argument(
        "--staging-dir",
        default=None,
        help="S3 staging directory (overrides ATHENA_STAGING_DIR from .env)"
    )
    parser.add_argument(
        "--region",
        default=None,
        help="AWS region (overrides AWS_REGION from .env)"
    )
    parser.add_argument(
        "--output",
        default="data/tweets_stage0_raw.parquet",
        help="Output parquet file path"
    )
    
    args = parser.parse_args()
    
    # Check if required environment variables are set
    if not args.sql and not os.getenv("ATHENA_SQL"):
        print("❌ Error: SQL query not provided!")
        print("   Either set ATHENA_SQL in .env file or use --sql argument")
        print("   Run 'python scripts/setup_aws_env.py' to set up your .env file")
        return 1
    
    if not os.getenv("AWS_ACCESS_KEY_ID") or not os.getenv("AWS_SECRET_ACCESS_KEY"):
        print("❌ Error: AWS credentials not found!")
        print("   Please set AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in .env file")
        print("   Run 'python scripts/setup_aws_env.py' to set up your .env file")
        return 1
    
    print("=" * 70)
    print("Stage 0: Fetching data from AWS Athena")
    print("=" * 70)
    
    try:
        print("\n📡 Connecting to AWS Athena...")
        df = load_from_athena(
            sql=args.sql,
            database=args.database,
            workgroup=args.workgroup,
            s3_staging_dir=args.staging_dir,
            aws_region=args.region,
        )
        
        print(f"✅ Retrieved {len(df):,} rows, {len(df.columns)} columns")
        print(f"   Columns: {', '.join(df.columns[:10])}{'...' if len(df.columns) > 10 else ''}")
        
        print("\n📅 Parsing dates...")
        try:
            df = basic_date_parse(df)
            print("✅ Dates parsed successfully")
        except Exception as e:
            print(f"⚠️  Warning: Could not parse dates automatically: {e}")
            print("   The data will be saved but dates may need manual processing")
        
        print("\n🔧 Standardizing tweet columns...")
        from src.features.clean import standardize_tweet_columns
        df = standardize_tweet_columns(df)
        
        # Save to Parquet
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        print(f"\n💾 Saving to: {args.output}")
        df.to_parquet(args.output, index=False)
        
        print(f"\n✅ Success! Created Stage 0 parquet file")
        print(f"   📊 Rows: {len(df):,}")
        print(f"   📁 File: {os.path.abspath(args.output)}")
        print(f"\n📋 Next steps:")
        print(f"   1. Run Stage 1: python scripts/run_sentiment_stage1.py")
        print(f"   2. Run Stage 2: python scripts/run_aspects_stage2.py")
        print(f"   3. Run Stage 3: python scripts/generate_stage3_themes.py")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Troubleshooting:")
        print("   - Check your AWS credentials in .env file")
        print("   - Verify your SQL query is correct")
        print("   - Ensure your IAM role has permissions to query Athena and read from S3")
        print("   - Check that the S3 staging directory exists and is accessible")
        return 1

if __name__ == "__main__":
    exit(main())
