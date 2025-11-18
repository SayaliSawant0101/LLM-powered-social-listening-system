"""
Standalone script to generate themes dynamically for API calls.
This can be called from Node.js server to generate themes on the fly.
"""
import sys
import os
import json
import argparse
import codecs

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features.themes import compute_themes_payload
from dotenv import load_dotenv

def main():
    # Load .env from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env_path = os.path.join(project_root, ".env")
    load_dotenv(env_path)  # Explicitly specify .env path
    
    parser = argparse.ArgumentParser(description="Generate themes dynamically")
    parser.add_argument("--start-date", type=str, default=None, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument("--n-clusters", type=int, default=12, help="Number of clusters (default: 12)")
    parser.add_argument("--parquet", type=str, default="data/tweets_stage2_aspects.parquet", help="Input parquet file")
    parser.add_argument("--max-rows", type=int, default=None, help="Maximum number of tweets to use (after filtering)")
    parser.add_argument("--openai-key", type=str, default=None, help="OpenAI API key")
    
    args = parser.parse_args()
    
    # Get OpenAI key - prioritize command line arg, then env var
    openai_key = args.openai_key
    if not openai_key:
        openai_key = os.getenv("OPENAI_API_KEY")
    
    # Debug: Check if key is loaded
    import sys
    if openai_key:
        print(f"[DEBUG] OpenAI key loaded: {openai_key[:20]}... (length: {len(openai_key)})", file=sys.stderr)
    else:
        print("[DEBUG] ❌ OpenAI key NOT loaded from .env or args", file=sys.stderr)
    
    try:
        # Redirect stderr to capture debug messages (but don't let them break JSON)
        import sys
        original_stderr = sys.stderr
        
        # Generate themes dynamically
        payload = compute_themes_payload(
            df=None,  # Will load from parquet
            parquet_stage2=args.parquet,
            n_clusters=args.n_clusters,
            start_date=args.start_date,
            end_date=args.end_date,
            openai_api_key=openai_key,
            merge_similar=True,
            max_rows=args.max_rows,
        )
        
        # Output JSON to stdout (Node.js will read this)
        # Make sure stderr is restored
        sys.stderr = original_stderr
        print(json.dumps(payload, ensure_ascii=False))
        return 0
        
    except Exception as e:
        error_obj = {"error": str(e), "type": type(e).__name__}
        print(json.dumps(error_obj, ensure_ascii=False))
        import traceback
        sys.stderr.write(traceback.format_exc())
        return 1

if __name__ == "__main__":
    exit(main())
