# src/features/themes.py 
from __future__ import annotations
import os, json, re, traceback, time, sys
from typing import Dict, List, Optional
import pandas as pd

FORCE_TFIDF = os.getenv("THEMES_EMB_BACKEND", "").lower() == "tfidf"

_STOP = {
    "walmart","rt","amp","https","http","co","www","com","org","net",
    "user","users","you","your","yours","u","ur","me","we","us","they","them",
    "im","ive","dont","didnt","cant","couldnt","wont","wouldnt","shouldnt",
    "like","just","get","got","one","two","three","also","still","even",
    "going","go","gotta","gonna","really","please","thanks","thank","help",
    "hey","hi","hello","ok","okay","any","every","everyone","someone","anyone",
    "today","yesterday","tomorrow","now","time","back","make","made","see",
    "store","stores","shop","shopping","customer","customers","people",
    "good","bad","great","best","worst","better","worse",
    "buy","bought","purchase","purchased","sale","sales",
    "app","apps","site","website","httpst","httpsco","tco",
    # Profanity and inappropriate words
    "fuck","fucking","fucked","shit","damn","hell","ass","bitch",
    # Usernames and mentions (add more as needed)
    "wutangkids","grok","don","flipkart","flipkartsupport","walmartinc",
    "sarahhuckabee","payoneer","_kalyan_k","h1b","radioactive",
    "sentomcotton","spencerhakimian","solporttom","mrpunkdoteth","matt_vanswol",
    "awsten","flagusanetwork","cz_binance","twentyonepilots",
    # Generic words that don't add meaning
    "topics","things","stuff","way","ways","thing","something","anything",
    "new","day","global","debt","fbi","their","there","they","them","these",
    "images","divine","waterparks","waterpark","target","amazon",
    "about","never","always","every","all","some","many","much","more","most",
    "very","really","quite","just","only","also","still","even","yet","already",
    "nctsmtown","nct","smtown",
    "motif","motifs",
    # More generic words
    "would","could","should","will","can","may","might","must","shall",
    "because","since","while","when","where","what","which","who","whom",
    "company","business","themes","theme","school","awesome","doing","right",
    "bullish","judgment","spiritual","available","students"
}
_URL_MENTION_HASHTAG = re.compile(r"https?://\S+|[@#]\w+")
_NON_ALNUM = re.compile(r"[^a-z0-9\s']")
_MULTI_SP = re.compile(r"\s+")

def _normalize(text: str) -> str:
    t = text.lower()
    t = _URL_MENTION_HASHTAG.sub(" ", t)
    t = _NON_ALNUM.sub(" ", t)
    t = _MULTI_SP.sub(" ", t).strip()
    return t

def _is_username_or_noise(word: str) -> bool:
    """Check if a word looks like a username, mention, or noise."""
    word_lower = word.lower()
    
    # Check stop words
    if word_lower in _STOP:
        return True
    
    # Check if it looks like a username (contains underscore, all lowercase with numbers, etc.)
    if '_' in word or (word.islower() and any(c.isdigit() for c in word) and len(word) > 5):
        return True
    
    # Check if it's profanity
    profanity_words = ['fuck', 'fucking', 'fucked', 'shit', 'damn', 'hell', 'ass', 'bitch']
    if word_lower in profanity_words:
        return True
    
    # Check if it's a known username pattern
    known_usernames = ['sentomcotton', 'spencerhakimian', 'solporttom', 'mrpunkdoteth', 
                       'matt_vanswol', 'awsten', 'flagusanetwork', 'cz_binance', 
                       'twentyonepilots', 'wutangkids', 'grok', 'flipkart']
    if word_lower in known_usernames:
        return True
    
    # Check if it's too short or generic
    if len(word) < 3:
        return True
    
    # Check for generic pronouns/articles/adverbs/auxiliary verbs
    generic_words = ['their', 'there', 'they', 'them', 'these', 'this', 'that', 'those',
                     'about', 'never', 'always', 'every', 'all', 'some', 'many', 'much',
                     'more', 'most', 'very', 'really', 'quite', 'just', 'only', 'also',
                     'still', 'even', 'yet', 'already', 'nctsmtown', 'nct', 'smtown',
                     'would', 'could', 'should', 'will', 'can', 'may', 'might', 'must',
                     'because', 'since', 'while', 'when', 'where', 'what', 'which', 'who',
                     'company', 'business', 'themes', 'theme', 'school', 'awesome',
                     'doing', 'right', 'bullish', 'judgment', 'spiritual', 'available', 'students',
                     'motif', 'motifs']
    if word_lower in generic_words:
        return True
    
    return False

def _is_theme_meaningless(theme_name: str) -> bool:
    """Check if a theme name is too generic or meaningless."""
    theme_lower = theme_name.lower()
    words = [w.strip('.,!?;:') for w in theme_lower.split()]
    
    # Check for generic combinations
    generic_patterns = [
        ['their', 'and'], ['there', 'and'], ['they', 'and'], ['them', 'and'],
        ['about', 'and'], ['never', 'and'], ['always', 'and'], ['every', 'and'],
        ['would', 'and'], ['because', 'and'], ['should', 'and'], ['will', 'and'],
        ['doing', 'and'], ['right', 'and'], ['bullish', 'and'], ['judgment', 'and'],
        ['spiritual', 'and'],
        ['target', 'and', 'waterparks'], ['images', 'and', 'divine'],
        ['target', 'and', 'amazon'], ['amazon', 'and', 'customer', 'experience'],
        ['amazon', 'and', 'customer', 'feedback'], ['prices', 'and', 'tariffs'],
        ['new', 'and', 'day'], ['global', 'and', 'debt'],
        ['nctsmtown', 'and'], ['nct', 'and'], ['smtown', 'and'],
        ['customer', 'service', 'and', 'never'], ['customer', 'service', 'and', 'always'],
        ['customer', 'experience', 'and', 'about'], ['customer', 'experience', 'and', 'would'],
        ['customer', 'experience', 'and', 'because'], ['customer', 'service', 'and', 'company'],
        ['customer', 'service', 'and', 'right'], ['customer', 'experience', 'and', 'right'],
        ['customer', 'service', 'and', 'doing'], ['customer', 'experience', 'and', 'bullish'],
        ['business', 'and', 'customer', 'experience'], ['themes', 'and', 'customer', 'experience'],
        ['school', 'and', 'customer', 'support'], ['bullish', 'and', 'customer', 'experience']
    ]
    
    for pattern in generic_patterns:
        if all(p in words for p in pattern):
            return True
    
    # Check if both parts are too generic
    if 'and' in words:
        and_idx = words.index('and')
        if and_idx > 0 and and_idx < len(words) - 1:
            part1 = ' '.join(words[:and_idx])
            part2 = ' '.join(words[and_idx+1:])
            
            # Check if either part is a generic word
            generic_single = ['their', 'there', 'they', 'them', 'these', 'this', 'that', 'new', 'day', 
                             'global', 'debt', 'images', 'divine', 'about', 'never', 'always', 'every',
                             'all', 'some', 'many', 'much', 'more', 'most', 'very', 'really', 'quite',
                             'just', 'only', 'also', 'still', 'even', 'yet', 'already', 'nctsmtown', 'nct', 'smtown',
                             'would', 'could', 'should', 'will', 'can', 'may', 'might', 'must',
                             'because', 'since', 'while', 'when', 'where', 'what', 'which', 'who',
                             'company', 'business', 'themes', 'theme', 'school', 'awesome',
                             'doing', 'right', 'bullish', 'judgment', 'spiritual', 'available', 'students']
            if part1 in generic_single or part2 in generic_single:
                return True
            
            # Check if it's just "X and Customer Service/Feedback/Experience" where X is generic
            generic_first = ['target', 'amazon', 'their', 'there', 'images', 'divine', 'waterparks',
                           'about', 'never', 'always', 'nctsmtown', 'nct', 'smtown', 'would', 'because',
                           'should', 'company', 'business', 'themes', 'school', 'doing', 'right',
                           'bullish', 'judgment', 'spiritual']
            generic_second = ['customer service', 'customer feedback', 'customer experience', 'delivery', 
                            'business', 'company', 'never', 'always', 'support', 'right', 'doing']
            if part1 in generic_first and part2 in generic_second:
                return True
            
            # Check if second part is a generic adverb/word
            if part2 in ['never', 'always', 'about', 'every', 'all', 'some', 'many', 'much', 'company', 'business']:
                return True
            
            # Check if first part is too generic when paired with customer experience
            if part2 == 'customer experience' and part1 in ['would', 'because', 'should', 'business', 'themes', 'school']:
                return True
    
    # Check if theme is too short or has too few meaningful words
    meaningful_words = [w for w in words if w not in ['and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by']]
    if len(meaningful_words) < 3:
        return True
    
    return False


# ===== Business Taxonomy & Validation Helpers ===== #
BANNED_THEME_WORDS = {
    "doing","right","bullish","judgment","spiritual","available","students","motif","motifs",
    "would","could","should","will","can","may","might","must","shall",
    "because","since","while","when","where","what","which","who","whom",
    "company","business","themes","theme","school","awesome"
}

THEME_TAXONOMY = [
    "Operational Issues",
    "Multi-Aspect Feedback",
    "Customer Service Experience",
    "Service Quality",
    "Customer Support",
    "Order Fulfillment",
    "Delivery Challenges",
    "Pricing Concerns",
    "Tariff Impact",
    "Product Quality",
    "Product Availability",
    "Shopping Experience",
    "Store Visits",
    "Walmart Brand",
    "Corporate Topics",
    "Employee Relations",
    "Workplace Feedback",
    "Customer Appreciation",
    "Positive Feedback",
    "Product Reviews",
    "Recommendations",
    "American Products",
    "International Products",
    "Food Products",
    "Shrimp Product Quality",
    "Digital Experience"
]

THEME_TAXONOMY_TEXT = "\n".join(f"- '{term}'" for term in THEME_TAXONOMY)
THEME_CANONICAL_DEFAULT = "Customer Service Experience Concerns"


def _validate_theme_title(title: str) -> bool:
    if not title:
        return False
    if _is_theme_meaningless(title):
        return False
    words = [w.lower().strip('.,!?;:') for w in title.split()]
    if any(w in BANNED_THEME_WORDS for w in words):
        return False
    title_lower = title.lower()
    if not any(term.lower() in title_lower for term in THEME_TAXONOMY):
        return False
    return True


def _sanitize_theme_title(title: str, fallback: str) -> str:
    """Return a cleaned, validated theme title or a professional fallback."""
    def _clean(text: str) -> str:
        if not text:
            return ""
        cleaned = text.strip().strip('"').strip("'")
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned

    primary = _clean(title)
    if _validate_theme_title(primary):
        return primary

    fb = _clean(fallback)
    if _validate_theme_title(fb):
        return fb

    return THEME_CANONICAL_DEFAULT

def _top_keywords(texts: List[str], top_k: int = 8) -> List[str]:
    from collections import Counter
    cnt = Counter()
    
    # Enhanced filtering for professional keywords
    business_relevant_patterns = {
        'service', 'support', 'help', 'assistance', 'staff', 'employee', 'worker',
        'price', 'pricing', 'cost', 'tariff', 'fee', 'charge', 'expensive', 'cheap',
        'delivery', 'shipping', 'fulfillment', 'order', 'purchase', 'buy',
        'product', 'item', 'quality', 'defect', 'broken', 'damaged',
        'return', 'refund', 'exchange', 'warranty',
        'store', 'location', 'visit', 'shopping', 'experience',
        'app', 'website', 'online', 'digital', 'mobile',
        'customer', 'client', 'satisfaction', 'complaint', 'issue', 'problem',
        'availability', 'stock', 'inventory', 'out', 'sold',
        'brand', 'corporate', 'company', 'business'
    }
    
    for t in texts:
        normalized = _normalize(str(t))
        tokens = normalized.split()
        for tok in tokens:
            # Skip if too short, is stop word, is digit, or is a username/mention-like pattern
            if len(tok) < 3 or tok in _STOP or tok.isdigit():
                continue
            # Skip if looks like username (starts with @ or _)
            if tok.startswith('@') or tok.startswith('_'):
                continue
            # Skip if all caps (likely acronyms/mentions)
            if tok.isupper() and len(tok) > 5:
                continue
            # Prefer business-relevant terms
            if tok in business_relevant_patterns:
                cnt[tok] += 2  # Boost business-relevant terms
            else:
                cnt[tok] += 1
    
    # Get top keywords, but filter out noise aggressively
    keywords = [w for w, _ in cnt.most_common(top_k * 3)]  # Get more candidates
    # Filter to keep only meaningful business terms
    filtered = []
    for kw in keywords:
        # Skip if it's noise, username, or profanity
        if _is_username_or_noise(kw):
            continue
        
        # Only keep business-relevant terms
        if kw.lower() in business_relevant_patterns:
            filtered.append(kw)
        elif len(kw) > 4 and kw.isalpha() and not kw.isupper():  # Longer, meaningful words
            # Double-check it's not a username
            if not _is_username_or_noise(kw):
                filtered.append(kw)
        
        if len(filtered) >= top_k:
            break
    
    # If we don't have enough, fall back to business terms only
    if len(filtered) < top_k:
        business_only = [w for w, _ in cnt.most_common(top_k * 5) 
                        if w.lower() in business_relevant_patterns and not _is_username_or_noise(w)]
        filtered.extend(business_only[:top_k - len(filtered)])
    
    return filtered[:top_k] if filtered else ['customer', 'service', 'product', 'quality', 'price', 'delivery', 'experience', 'support']

def _merge_similar_themes(themes: List[dict], similarity_threshold: float = 0.7) -> List[dict]:
    """Merge themes that are semantically similar based on name and summary similarity."""
    if len(themes) <= 1:
        return [
            {
                **theme,
                "component_ids": list(theme.get("component_ids", [theme["id"]])),
            }
            for theme in themes
        ]
    
    # Enhanced similarity check based on common words and semantic meaning
    def _calculate_similarity(theme1: dict, theme2: dict) -> float:
        name1 = _normalize(theme1["name"]).split()
        name2 = _normalize(theme2["name"]).split()
        
        # Remove common stop words
        name1 = [w for w in name1 if w not in _STOP and len(w) > 2]
        name2 = [w for w in name2 if w not in _STOP and len(w) > 2]
        
        if not name1 or not name2:
            return 0.0
        
        # Calculate Jaccard similarity
        set1, set2 = set(name1), set(name2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        if union == 0:
            return 0.0
        
        jaccard_sim = intersection / union
        
        # Additional semantic similarity checks
        semantic_similarity = 0.0
        
        # Check for semantic word pairs (synonyms/related terms)
        semantic_pairs = [
            ("availability", "stock"), ("stock", "inventory"), ("product", "item"),
            ("order", "purchase"), ("fulfillment", "delivery"), ("processing", "handling"),
            ("customer", "client"), ("service", "support"), ("issue", "problem"),
            ("concern", "issue"), ("problem", "issue"), ("complaint", "issue"),
            ("experience", "interaction"), ("humor", "funny"), ("joke", "humor"),
            ("availability", "concern"), ("stock", "concern"), ("product", "availability"),
            ("delivery", "fulfillment"), ("pricing", "tariff")
        ]
        
        for word1, word2 in semantic_pairs:
            if (word1 in set1 and word2 in set2) or (word2 in set1 and word1 in set2):
                semantic_similarity += 0.3
                break
        
        # Check for substring matches (e.g., "Product Availability" vs "Product Availability Concerns")
        name1_str = " ".join(name1)
        name2_str = " ".join(name2)
        
        if name1_str in name2_str or name2_str in name1_str:
            semantic_similarity += 0.4
        
        # Check for high word overlap (e.g., "Product Availability and Stock Issues" vs "Product Availability Concerns")
        if len(set1.intersection(set2)) >= 2:  # At least 2 common words
            semantic_similarity += 0.3
        
        # Return the maximum of Jaccard similarity and semantic similarity
        return max(jaccard_sim, semantic_similarity)
    
    merged_themes = []
    used_indices = set()
    
    for i, theme1 in enumerate(themes):
        if i in used_indices:
            continue
            
        # Find similar themes to merge
        similar_themes = [theme1]
        for j, theme2 in enumerate(themes[i+1:], i+1):
            if j in used_indices:
                continue
                
            similarity = _calculate_similarity(theme1, theme2)
            if similarity >= similarity_threshold:
                similar_themes.append(theme2)
                used_indices.add(j)
        
        # Merge similar themes
        if len(similar_themes) > 1:
            # Combine tweet counts and sentiment
            total_tweets = sum(t["tweet_count"] for t in similar_themes)
            total_positive = sum(t["positive"] for t in similar_themes)
            total_negative = sum(t["negative"] for t in similar_themes)
            total_neutral = sum(t["neutral"] for t in similar_themes)
            
            # Use the theme name with highest tweet count
            main_theme = max(similar_themes, key=lambda x: x["tweet_count"])
            
            # Combine summaries - use the best summary without merge indicators
            summaries = [t["summary"] for t in similar_themes if t["summary"]]
            combined_summary = main_theme["summary"]
            # Don't add merge indicators - just use the best summary
            
            merged_theme = {
                "id": main_theme["id"],
                "name": main_theme["name"],
                "summary": combined_summary,
                "tweet_count": total_tweets,
                "positive": total_positive,
                "negative": total_negative,
                "neutral": total_neutral,
                "component_ids": list({cid for t in similar_themes for cid in t.get("component_ids", [t["id"]])}),
            }
            merged_themes.append(merged_theme)
        else:
            merged_themes.append({
                **theme1,
                "component_ids": list(theme1.get("component_ids", [theme1["id"]])),
            })
        
        used_indices.add(i)
    
    # Sort by tweet count again after merging
    return sorted(merged_themes, key=lambda x: x["tweet_count"], reverse=True)

# =========================
# A) Diversity-aware selection (MMR)
# =========================
def _mmr_select(themes_all: List[dict], texts_by_cluster: Dict[int, List[str]], n: int, lam: float = 0.65) -> List[dict]:
    """
    Maximal Marginal Relevance selection of N themes.
    lam balances size/coverage vs diversity: lam in [0..1]. Higher -> favors size more.
    """
    # Build a TF-IDF over cluster "documents" (join texts per cluster) for cosine similarity between themes
    joined = []
    ids = []
    for t in themes_all:
        cid = int(t["id"])
        ids.append(cid)
        joined.append(" ".join(texts_by_cluster.get(cid, [])[:400]))  # limit per-cluster for speed

    if not joined:
        return themes_all[:n]

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize as _sk_normalize
    import numpy as np

    vec = TfidfVectorizer(max_features=4000, stop_words="english", ngram_range=(1,2))
    X = vec.fit_transform(joined).astype("float32")
    X = _sk_normalize(X)
    sims = (X @ X.T).toarray()  # cosine sim (clusters x clusters) - use toarray() instead of .A for scipy compatibility

    # Normalize tweet_count to 0..1 for comparability
    sizes = np.array([t["tweet_count"] for t in themes_all], dtype=float)
    if sizes.max() > 0:
        sizes = sizes / sizes.max()
    else:
        sizes = np.zeros_like(sizes)

    selected = []
    selected_idx = []
    # Greedy MMR
    for _ in range(min(n, len(themes_all))):
        best_j = None
        best_score = -1e9
        for j in range(len(themes_all)):
            if j in selected_idx:
                continue
            # diversity penalty = max similarity to anything already selected
            if selected_idx:
                div_pen = max(sims[j, k] for k in selected_idx)
            else:
                div_pen = 0.0
            score = lam * sizes[j] - (1 - lam) * div_pen
            if score > best_score:
                best_score = score
                best_j = j
        selected_idx.append(best_j)
        selected.append(themes_all[best_j])
    return selected

def compute_themes_payload(
    df: Optional[pd.DataFrame] = None,
    parquet_stage2: str = "data/tweets_stage2_aspects.parquet",
    n_clusters: int = 12,  # Default to 12 professional themes
    emb_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    openai_api_key: Optional[str] = None,
    merge_similar: bool = True,
    max_rows: Optional[int] = None,
) -> dict:
    """Return {"updated_at": ts, "themes": [{id,name,summary,tweet_count,positive,negative,neutral}], "used_llm": bool}."""
    if df is None:
        assert os.path.exists(parquet_stage2), f"Missing {parquet_stage2}"
        df = pd.read_parquet(parquet_stage2)

    id_col = next((c for c in ["tweet_id", "id_str", "id", "status_id"] if c in df.columns), None)
    if id_col:
        before = len(df)
        df = df.drop_duplicates(subset=[id_col])
        removed = before - len(df)
        if removed > 0:
            print(f"[Theme Generation] Removed {removed} duplicate tweets via {id_col}", file=sys.stderr)

    text_col = next((c for c in ["text_used","clean_tweet","text","fulltext"] if c in df.columns), None)
    if not text_col:
        raise KeyError("No text column among ['text_used','clean_tweet','text','fulltext'].")

    date_col = next((c for c in ["createdat","created_dt","created_at","tweet_date","date","dt"] if c in df.columns), None)
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce", utc=True).dt.tz_localize(None)
        if start_date:
            df = df[df[date_col] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df[date_col] <= pd.to_datetime(end_date)]

    if max_rows is not None and max_rows > 0 and len(df) > max_rows:
        if date_col:
            df = df.sort_values(by=date_col, ascending=False)
        df = df.head(max_rows).copy()
        print(f"[Theme Generation] Limited dataset to {len(df)} tweets (max_rows={max_rows})", file=sys.stderr)

    if df.empty:
        return {
            "updated_at": pd.Timestamp.utcnow().isoformat(),
            "themes": [],
            "used_llm": False,
            "source_row_count": 0,
        }

    texts = df[text_col].astype(str).tolist()

    # ---------- Optimized Processing: Sample data for faster processing ----------
    max_samples = 3000  # Reduced to 3000 for faster processing (was 5000)
    if len(df) > max_samples:
        df_sample = df.sample(n=max_samples, random_state=42)
        texts_to_process = df_sample[text_col].astype(str).tolist()
    else:
        df_sample = df
        texts_to_process = texts

    # ---------- Embeddings: ST if available, else TF-IDF ----------
    emb = None
    if not FORCE_TFIDF:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
            model = SentenceTransformer(emb_model)
            emb = model.encode(
                texts_to_process, batch_size=32, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False
            )
        except Exception:
            emb = None

    if emb is None:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.preprocessing import normalize
        vec = TfidfVectorizer(max_features=3000, stop_words="english", ngram_range=(1,2))  # Reduced features
        emb = vec.fit_transform(texts_to_process).astype("float32")
        emb = normalize(emb)

    # ---------- Clustering (Optimized) ----------
    from sklearn.cluster import KMeans
    if n_clusters is None:
        n_clusters = 12  # Default to 12 professional themes
    k = max(2, int(n_clusters))
    # Generate exactly k clusters to avoid double counting
    k_actual = k
    print(f"[Theme Generation] Generating {k_actual} clusters (target: {n_clusters})...", file=sys.stderr)
    km = KMeans(n_clusters=k_actual, random_state=42, n_init=1)  # Reduced to 1 for maximum speed
    labels = km.fit_predict(emb)
    
    # Map labels back to full dataset
    if len(df) > max_samples:
        # For sampled data, assign clusters to nearest centroids for remaining data
        remaining_texts = df[~df.index.isin(df_sample.index)][text_col].astype(str).tolist()
        if remaining_texts:
            if not FORCE_TFIDF and 'model' in locals():
                remaining_embeddings = model.encode(remaining_texts, batch_size=32, show_progress_bar=False)
            else:
                remaining_embeddings = vec.transform(remaining_texts).astype("float32")
                remaining_embeddings = normalize(remaining_embeddings)
            
            remaining_labels = km.predict(remaining_embeddings)
            
            # Combine labels
            df.loc[df_sample.index, "theme"] = labels
            df.loc[~df.index.isin(df_sample.index), "theme"] = remaining_labels
        else:
            df["theme"] = labels
    else:
        df["theme"] = labels

    os.makedirs("data", exist_ok=True)
    df.to_parquet("data/tweets_stage3_themes.parquet", index=False)

    # ---------- TF-IDF keywords per theme (Optimized) ----------
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf_kws: Dict[int, List[str]] = {}
    for tid, sub in df.groupby("theme"):
        sub_texts = sub[text_col].astype(str).tolist()
        if not sub_texts:
            tfidf_kws[int(tid)] = []
            continue
        # Sample texts if too many for faster processing
        if len(sub_texts) > 200:  # Reduced for speed
            sub_texts = sub_texts[:200]
        vec = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1,2))  # Reduced for speed
        X = vec.fit_transform(sub_texts)
        # Convert sparse matrix mean to dense array
        import numpy as np
        mean_matrix = X.mean(axis=0)
        if hasattr(mean_matrix, 'toarray'):
            scores = mean_matrix.toarray().flatten()
        else:
            scores = np.array(mean_matrix).flatten()
        feats = vec.get_feature_names_out()
        top_idx = scores.argsort()[::-1][:10]  # Get more candidates
        # Filter out noise words more aggressively
        toks = [f for f in feats[top_idx] if f.split()[0].lower() not in _STOP and f.split()[0].lower() not in ['wutangkids', 'grok', 'don', 'flipkart', 'payoneer', 'sarahhuckabee', 'walmartinc']]
        tfidf_kws[int(tid)] = toks[:6]

    # ---------- Sentiment counts ----------
    pos_counts, neg_counts, neu_counts = {}, {}, {}
    if "sentiment_label" in df.columns:
        for tid, sub in df.groupby("theme"):
            vc = sub["sentiment_label"].value_counts().to_dict()
            pos_counts[int(tid)] = int(vc.get("positive", 0))
            neg_counts[int(tid)] = int(vc.get("negative", 0))
            neu_counts[int(tid)] = int(vc.get("neutral", 0))
    else:
        for tid in df["theme"].unique():
            pos_counts[int(tid)] = 0
            neg_counts[int(tid)] = 0
            neu_counts[int(tid)] = 0

    # ---------- Name & summarize with OpenAI if key provided ----------
    client = None
    
    if openai_api_key:
        api_key = str(openai_api_key).strip()
        if api_key and len(api_key) > 20:  # Basic validation
            if not api_key.startswith("sk-"):
                print(f"[OpenAI] Warning: API key doesn't start with 'sk-' (got: {api_key[:10]}...)", file=sys.stderr)
            try:
                from openai import OpenAI
                client = OpenAI(api_key=api_key)
                # Test the client works by making a simple call (optional - can remove if too slow)
                print(f"[OpenAI] ✅ Client initialized successfully (key: {api_key[:20]}...)", file=sys.stderr)
            except ImportError:
                print(f"[OpenAI] ❌ openai package not installed. Install with: pip install openai", file=sys.stderr)
                client = None
            except Exception as e:
                print(f"[OpenAI] ❌ Client init error: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc(file=sys.stderr)
                client = None
        else:
            print(f"[OpenAI] ❌ Invalid API key (empty or too short: length={len(api_key) if api_key else 0})", file=sys.stderr)
            client = None
    else:
        print(f"[OpenAI] ❌ No OpenAI API key provided (openai_api_key is None or empty)", file=sys.stderr)

    theme_names: Dict[int, str] = {}
    summaries: Dict[int, str] = {}

    TITLE_SYSTEM = (
        "You are a professional retail insights analyst creating theme names for Walmart customer feedback analysis reports. "
        "Your theme names will be used in executive presentations and stakeholder reports. "
        "Generate ONLY professional, business-appropriate theme names that generalize customer concerns into meaningful business categories. "
        "IGNORE specific brand names, usernames, or mentions - focus on the UNDERLYING BUSINESS THEME. "
        "Use only the approved taxonomy terms when naming themes. "
        f"Approved taxonomy terms:\n{THEME_TAXONOMY_TEXT}\n"
    )
    TITLE_USER_TMPL = (
        "Based on customer feedback keywords and samples, create ONE professional theme name for a Walmart customer insights report.\n\n"
        "CRITICAL INSTRUCTIONS:\n"
        "1. IGNORE ALL usernames, mentions, profanity, and noise words (like 'sentomcotton', 'fuck', 'matt_vanswol', 'awsten', 'spencerhakimian')\n"
        "2. If you see profanity, usernames, or meaningless words in keywords, COMPLETELY IGNORE them\n"
        "3. Focus ONLY on the BUSINESS THEME or CUSTOMER CONCERN from the sample tweets\n"
        "4. NEVER create generic names like 'Target and Customer Experience', 'Prices and Pricing', 'New and Day', or 'Global and Debt'\n"
        "5. NEVER include usernames, profanity, or competitor names unless it's a legitimate business comparison\n"
        "6. Use DISTINCT business categories - avoid repeating the same category in different themes\n"
        "7. If keywords are all noise, infer the theme from sample tweets (e.g., if tweets mention delivery issues → 'Order Fulfillment and Delivery Challenges')\n\n"
        "REQUIRED FORMAT:\n"
        "'[Specific Business Category] and [Specific Customer Concern/Focus]'\n\n"
        "GOOD EXAMPLES (use these EXACT formats as your guide - these are the REQUIRED style):\n"
        "- 'Operational Issues and Multi-Aspect Feedback'\n"
        "- 'Customer Service Experience Concerns'\n"
        "- 'Shrimp Product Quality and Availability'\n"
        "- 'Shopping Experience and Store Visits'\n"
        "- 'Walmart Brand and Corporate Topics'\n"
        "- 'Pricing Concerns and Tariff Impact'\n"
        "- 'Order Fulfillment Issues'\n"
        "- 'American Product Availability Concerns'\n"
        "- 'Service Quality and Support'\n"
        "- 'Product Reviews and Recommendations'\n"
        "- 'Customer Appreciation and Feedback'\n"
        "- 'Employee Relations and Workplace Feedback'\n\n"
        "NOTE: Themes can be:\n"
        "- '[Business Category] and [Specific Focus]' (e.g., 'Order Fulfillment and Delivery Challenges')\n"
        "- '[Business Category] [Issue Type]' (e.g., 'Customer Service Experience Concerns', 'Order Fulfillment Issues')\n"
        "- '[Product Category] [Quality/Availability]' (e.g., 'Shrimp Product Quality and Availability')\n\n"
        "BAD EXAMPLES (DO NOT create these - these are REAL bad examples):\n"
        "- 'Would and Customer Experience' (generic auxiliary verb, meaningless)\n"
        "- 'Because and Customer Experience' (generic conjunction, meaningless)\n"
        "- 'Customer Service and Company' (too generic, 'company' adds no value)\n"
        "- 'Business and Customer Experience' (too generic, 'business' is too broad)\n"
        "- 'Themes and Customer Experience' (meta word 'themes', meaningless)\n"
        "- 'School and Customer Support' (unrelated topic 'school', not retail-relevant)\n"
        "- 'Sentomcotton and Customer Feedback' (contains username)\n"
        "- 'Fuck and Fucking' (contains profanity)\n"
        "- 'Target and Awsten' (contains username/competitor)\n"
        "- 'Their and Customer Service' (generic pronoun, meaningless)\n"
        "- 'About and Customer Experience' (generic word 'about', meaningless)\n"
        "- 'Customer Service and Never' (generic adverb 'never', meaningless)\n"
        "- 'Nctsmtown and Customer Experience' (brand/username, not business-relevant)\n"
        "- 'Target and Waterparks' (unrelated, not business-relevant)\n"
        "- 'Amazon and Customer Experience' (too generic, competitor name)\n"
        "- 'Prices and Tariffs' (redundant, both mean pricing)\n\n"
        "STRICT RULES:\n"
        "1. Must be 4-7 words total\n"
        "2. Use Title Case\n"
        "3. Use DISTINCT business categories - each theme should cover a different aspect\n"
        "4. Be SPECIFIC - 'Pricing Concerns and Tariff Impact' not 'Prices and Tariffs' (redundant)\n"
        "5. NEVER use: 'user', 'tweet', 'feedback', 'topics' as main terms\n"
        "6. NEVER use: generic words like 'their', 'there', 'they', 'about', 'never', 'always', 'would', 'because', 'should', 'company', 'business', 'themes', 'school'\n"
        "7. NEVER use: auxiliary verbs or conjunctions (like 'Would', 'Because', 'Should', 'Will', 'Can')\n"
        "8. NEVER use: brand names or usernames (like 'Nctsmtown', 'Target', 'Amazon' unless it's a legitimate business comparison)\n"
        "9. NEVER use: competitor names alone (like 'Target and Waterparks' or 'Amazon and Customer Experience')\n"
        "10. NEVER create: redundant combinations (e.g., 'Prices and Tariffs' - both mean pricing)\n"
        "11. NEVER use: generic adverbs, pronouns, or auxiliary verbs as theme parts\n"
        "12. NEVER use: meta words like 'themes', 'theme', or unrelated topics like 'school'\n"
        "13. NEVER use: overly broad words like 'company', 'business' as standalone theme parts\n"
        "14. Focus on: Service, Pricing, Products, Delivery, Experience, Quality, Availability, Staff, Returns, App, etc.\n"
        "15. Make each theme UNIQUE and MEANINGFUL for business decisions\n"
        "16. Each part of the theme (before and after 'and') must be a SPECIFIC business concept\n"
        "17. Both parts must be NOUNS or NOUN PHRASES describing business categories, NOT generic words, verbs, or adverbs\n"
        "18. Use SPECIFIC business terminology: 'Operational Issues', 'Service Quality', 'Product Availability', 'Pricing Concerns', etc.\n\n"
        "Keywords (filtered, ignore noise): {kws}\n\n"
        "Generate ONE professional, specific theme name. Return ONLY the name. No quotes, no explanations."
    )

    SUMMARY_SYSTEM = (
        "You are a retail insights analyst. Write 3-4 lines describing the theme for a Walmart stakeholder. "
        "Focus on what customers are discussing, not sentiment counts or action items. Be descriptive and informative. "
        "IGNORE specific usernames, brand mentions, or noise words - focus on the business theme."
    )
    SUMMARY_USER_TMPL = (
        "Theme name: {title}\n"
        "Top keywords (filtered): {kws}\n"
        "Examples (up to 2): {examples}\n"
        "Output: 3-4 lines describing what customers are discussing in this theme. "
        "Focus on the BUSINESS THEME (e.g., customer service, product quality, pricing, delivery, etc.). "
        "Do NOT mention specific usernames, brand names, or noise words. Do not mention sentiment counts or suggest actions."
    )

    total_themes = len(df["theme"].unique())
    start_time = time.time()
    print(f"[Theme Generation] Processing {total_themes} themes in parallel...", file=sys.stderr)
    
    # Prepare theme data for parallel processing
    theme_data_list = []
    for tid in sorted(df["theme"].unique().astype(int)):
        sub = df[df["theme"] == tid]
        kws_prompt = tfidf_kws.get(tid) or _top_keywords(sub[text_col].tolist(), 10)
        sample_tweets = sub[text_col].astype(str).head(3).tolist()  # Reduced to 3 for speed
        # Aggressively filter keywords - only keep meaningful business terms
        # Filter out all noise, usernames, and generic words
        clean_kws = [k for k in kws_prompt if not _is_username_or_noise(k) and len(k) > 3 and not k.isdigit() and k.isalpha()][:8]
        
        # Generate professional fallback title - filter out noise first
        filtered_clean_kws = [kw for kw in clean_kws if kw.lower() not in _STOP and len(kw) > 2]
        
        # Professional keyword mapping for business themes - use specific business terminology
        keyword_mapping = {
            'service': 'Customer Service Experience', 'support': 'Customer Support', 'help': 'Customer Support',
            'staff': 'Staff Interaction', 'employee': 'Employee Relations', 'worker': 'Workplace Feedback',
            'employees': 'Employee Relations', 'people': 'Staff Interaction',
            'price': 'Pricing Concerns', 'pricing': 'Pricing Concerns', 'cost': 'Pricing Concerns',
            'tariff': 'Tariff Impact', 'tariffs': 'Tariff Impact', 'prices': 'Pricing Concerns',
            'delivery': 'Delivery Challenges', 'shipping': 'Shipping Issues', 'fulfillment': 'Order Fulfillment',
            'order': 'Order Fulfillment', 'purchase': 'Order Fulfillment', 'orders': 'Order Fulfillment',
            'return': 'Returns', 'refund': 'Returns', 'exchange': 'Returns',
            'product': 'Product Quality', 'item': 'Product Quality', 'quality': 'Product Quality',
            'products': 'Product Quality', 'items': 'Product Quality',
            'shopping': 'Shopping Experience', 'store': 'Store Visits', 'visit': 'Store Visits',
            'customer': 'Customer Experience', 'experience': 'Customer Experience',
            'walmart': 'Walmart Brand', 'brand': 'Corporate Topics', 'corporate': 'Corporate Topics',
            'issue': 'Operational Issues', 'problem': 'Operational Issues', 'concern': 'Customer Concerns',
            'issues': 'Operational Issues', 'problems': 'Operational Issues', 'concerns': 'Customer Concerns',
            'availability': 'Product Availability', 'stock': 'Product Availability', 'inventory': 'Product Availability',
            'app': 'Digital Experience', 'website': 'Digital Experience', 'online': 'Digital Experience',
            'review': 'Product Reviews', 'recommendation': 'Product Recommendations', 'reviews': 'Product Reviews',
            'american': 'American Products', 'indian': 'International Products',
            'shrimp': 'Shrimp Product Quality', 'food': 'Food Products',
            'awesome': 'Customer Appreciation', 'positive': 'Positive Feedback', 'appreciation': 'Customer Appreciation',
            'feedback': 'Positive Feedback', 'multi': 'Multi-Aspect Feedback',
        }
        
        mapped_terms = []
        for kw in filtered_clean_kws[:4]:  # Check more keywords
            kw_lower = kw.lower()
            if kw_lower in keyword_mapping:
                mapped_term = keyword_mapping[kw_lower]
                if mapped_term not in mapped_terms:  # Avoid duplicates
                    mapped_terms.append(mapped_term)
            elif not _is_username_or_noise(kw):
                # Only add if it's a meaningful business term
                if len(kw) > 4 and kw.isalpha() and not kw.isupper():
                    mapped_terms.append(kw.title())

        if len(mapped_terms) >= 2:
            fallback_title = f"{mapped_terms[0]} and {mapped_terms[1]}"
            # Validate fallback title is not meaningless
            if _is_theme_meaningless(fallback_title):
                # Use professional fallback based on first term
                if mapped_terms[0] in ['Customer Service Experience', 'Service Quality']:
                    fallback_title = f"{mapped_terms[0]} Concerns"
                elif mapped_terms[0] in ['Order Fulfillment', 'Product Quality']:
                    fallback_title = f"{mapped_terms[0]} Issues"
                else:
                    fallback_title = f"{mapped_terms[0]} and Customer Experience"
        elif len(mapped_terms) == 1:
            # Create specific professional theme based on the term
            single = mapped_terms[0]
            if single in ['Customer Service Experience', 'Service Quality']:
                fallback_title = f"{single} Concerns"
            elif single in ['Order Fulfillment', 'Product Quality']:
                fallback_title = f"{single} Issues"
            elif single in ['Pricing Concerns', 'Product Availability']:
                fallback_title = f"{single}"
            else:
                fallback_title = f"{single} and Customer Experience"
        elif filtered_clean_kws:
            # Only use if keywords are meaningful
            valid_kws = [kw for kw in filtered_clean_kws[:2] if not _is_username_or_noise(kw) and len(kw) > 3]
            if len(valid_kws) >= 2:
                # Map keywords to professional terms
                kw1_mapped = keyword_mapping.get(valid_kws[0].lower(), valid_kws[0].title())
                kw2_mapped = keyword_mapping.get(valid_kws[1].lower(), valid_kws[1].title())
                fallback_title = f"{kw1_mapped} and {kw2_mapped}"
                if _is_theme_meaningless(fallback_title):
                    fallback_title = THEME_CANONICAL_DEFAULT
            else:
                fallback_title = THEME_CANONICAL_DEFAULT
        else:
            fallback_title = THEME_CANONICAL_DEFAULT
        
        fallback_title = _sanitize_theme_title(fallback_title, THEME_CANONICAL_DEFAULT)

        theme_data_list.append({
            'tid': tid,
            'sub': sub,
            'kws_prompt': kws_prompt,
            'clean_kws': clean_kws,
            'sample_tweets': sample_tweets,
            'fallback_title': fallback_title,
            'pos': pos_counts.get(tid, 0),
            'neg': neg_counts.get(tid, 0),
            'neu': neu_counts.get(tid, 0),
        })
    
    # Parallel processing function for OpenAI calls
    def process_theme_name(theme_data):
        tid = theme_data['tid']
        clean_kws = theme_data['clean_kws']
        sample_tweets = theme_data['sample_tweets']
        fallback_title = theme_data['fallback_title']
        
        if not client:
            return tid, fallback_title
        
        try:
            # Filter out noise words VERY aggressively
            filtered_kws = [k for k in clean_kws if not _is_username_or_noise(k) and len(k) > 3][:8]
            
            # Only keep business-relevant keywords (must be meaningful business terms)
            business_relevant = ['service', 'support', 'price', 'pricing', 'delivery', 'order', 'product', 
                               'quality', 'shopping', 'store', 'customer', 'staff', 'return', 'availability', 
                               'app', 'website', 'shipping', 'fulfillment', 'tariff', 'cost', 'employee', 
                               'experience', 'issue', 'problem', 'concern', 'review', 'recommendation',
                               'shrimp', 'food', 'employees', 'prices', 'tariffs']
            business_kws = [k for k in filtered_kws if k.lower() in business_relevant]
            
            # If we have no good keywords, use generic business terms
            if not business_kws:
                business_kws = ['customer service', 'product quality', 'order fulfillment', 'shopping experience']
            
            sample_context = "\n".join([f"{i+1}. {tweet[:150]}..." for i, tweet in enumerate(sample_tweets[:2])])  # Reduced to 2 samples
            
            enhanced_prompt = TITLE_USER_TMPL.format(
                kws=", ".join(business_kws[:8]),
                taxonomy=THEME_TAXONOMY_TEXT
            )
            if sample_context:
                enhanced_prompt += f"\n\nSample customer feedback:\n{sample_context}\n\n"
            enhanced_prompt += "Return ONLY the theme name. No explanations, no quotes. IGNORE any specific usernames or brand mentions in the keywords."
            
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":TITLE_SYSTEM},
                    {"role":"user","content":enhanced_prompt}
                ],
                temperature=0.2,
                max_tokens=50,
                timeout=30  # Reduced timeout since we're parallelizing
            )
            t = (resp.choices[0].message.content or "").strip().strip('"').strip("'")
            t = t.replace("Theme: ", "").replace("Title: ", "").replace("Name: ", "").replace("**", "").replace("*", "").strip()
            
            # Post-process: Remove any profanity or usernames that might have slipped through
            words = t.split()
            cleaned_words = []
            for word in words:
                word_lower = word.lower().strip('.,!?;:')
                if not _is_username_or_noise(word_lower):
                    cleaned_words.append(word)
                else:
                    # Skip profanity and usernames
                    if word_lower in ['fuck', 'fucking', 'shit', 'damn']:
                        continue  # Skip profanity
                    elif '_' in word_lower or word_lower in ['sentomcotton', 'spencerhakimian', 'awsten', 'matt_vanswol', 'solporttom', 'mrpunkdoteth']:
                        continue  # Skip usernames
            
            if cleaned_words:
                t = ' '.join(cleaned_words)
            
            # Final validation: reject if contains profanity, usernames, or is meaningless
            if t and len(t) > 5:
                t_words = [w.lower().strip('.,!?;:') for w in t.split()]
                if any(_is_username_or_noise(w) for w in t_words):
                    # If contains noise, use fallback
                    return tid, fallback_title
                
                # Check if theme is meaningless or too generic
                if _is_theme_meaningless(t):
                    # If meaningless, use fallback
                    return tid, fallback_title
                
                # Title case the cleaned result
                words = t.split()
                title_cased = []
                small_words = {'and', 'or', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
                for i, word in enumerate(words):
                    if i == 0 or word.lower() not in small_words:
                        title_cased.append(word.capitalize())
                    else:
                        title_cased.append(word.lower())
                final_title = ' '.join(title_cased)
                
                # Final check: reject if still meaningless after cleaning
                if _is_theme_meaningless(final_title):
                    return tid, fallback_title
                
                final_title = _sanitize_theme_title(final_title, fallback_title)
                return tid, final_title
            return tid, fallback_title
        except Exception as e:
            print(f"[Theme {tid}] OpenAI name error: {e}", file=sys.stderr)
            return tid, fallback_title
    
    def process_theme_summary(theme_data, title):
        tid = theme_data['tid']
        kws_prompt = theme_data['kws_prompt']
        sub = theme_data['sub']
        # Filter keywords for base summary - remove noise aggressively
        filtered_base_kws = [k for k in kws_prompt if not _is_username_or_noise(k) and len(k) > 3][:4]
        # Only keep business-relevant terms
        business_relevant = ['service', 'support', 'price', 'pricing', 'delivery', 'order', 'product', 
                           'quality', 'shopping', 'store', 'customer', 'staff', 'return', 'availability']
        filtered_base_kws = [k for k in filtered_base_kws if k.lower() in business_relevant or (len(k) > 5 and k.isalpha())]
        if not filtered_base_kws:
            filtered_base_kws = ['customer service', 'product quality', 'shopping experience', 'order fulfillment']

        base_summary = (
            f"{title}: This theme focuses on {', '.join(filtered_base_kws)}. "
            f"Customers are discussing various aspects related to this topic. "
            f"The discussions cover different perspectives and experiences."
        )
        
        if not client:
            return tid, base_summary
        
        try:
            ex = [str(x) for x in sub[text_col].astype(str).head(2).tolist()]
            # Filter keywords for summary - remove noise aggressively
            filtered_summary_kws = [k for k in kws_prompt if not _is_username_or_noise(k) and len(k) > 3][:6]
            # Only keep business-relevant terms
            business_relevant = ['service', 'support', 'price', 'pricing', 'delivery', 'order', 'product', 
                               'quality', 'shopping', 'store', 'customer', 'staff', 'return', 'availability']
            filtered_summary_kws = [k for k in filtered_summary_kws if k.lower() in business_relevant or (len(k) > 5 and k.isalpha())]
            if not filtered_summary_kws:
                filtered_summary_kws = ['customer', 'service', 'product']
            
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role":"system","content":SUMMARY_SYSTEM},
                    {"role":"user","content":SUMMARY_USER_TMPL.format(title=title, kws=", ".join(filtered_summary_kws), examples=ex)}
                ],
                temperature=0.3,
                max_tokens=100,  # Reduced for speed
                timeout=30  # Reduced timeout
            )
            s = (resp.choices[0].message.content or "").strip()
            return tid, s or base_summary
        except Exception as e:
            print(f"[Theme {tid}] OpenAI summary error: {e}", file=sys.stderr)
            return tid, base_summary
    
    # Process themes in parallel using ThreadPoolExecutor
    if client:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        print(f"[Theme Generation] Generating {total_themes} professional theme names in parallel...", file=sys.stderr)

        # Parallel name generation (batch of 12)
        with ThreadPoolExecutor(max_workers=12) as executor:
            name_futures = {executor.submit(process_theme_name, td): td for td in theme_data_list}
            for future in as_completed(name_futures):
                tid, title = future.result()
                theme_names[tid] = title
                print(f"[Theme Generation] ✓ Generated name for theme {tid}: {title}", file=sys.stderr)

        # Parallel summary generation (batch of 12)
        print(f"[Theme Generation] Generating {total_themes} theme summaries in parallel...", file=sys.stderr)
        with ThreadPoolExecutor(max_workers=12) as executor:
            summary_futures = {executor.submit(process_theme_summary, td, theme_names[td['tid']]): td for td in theme_data_list}
            for future in as_completed(summary_futures):
                tid, summary = future.result()
                summaries[tid] = summary
                print(f"[Theme Generation] ✓ Generated summary for theme {tid}", file=sys.stderr)
    else:
        # No OpenAI - use fallbacks
        for theme_data in theme_data_list:
            tid = theme_data['tid']
            theme_names[tid] = theme_data['fallback_title']
            summaries[tid] = (
                f"{theme_data['fallback_title']}: This theme focuses on {', '.join(theme_data['kws_prompt'][:4])}. "
                f"Customers are discussing various aspects related to this topic."
            )

    with open("data/theme_names.json", "w") as f:
        json.dump({int(k): v for k,v in theme_names.items()}, f, ensure_ascii=False, indent=2)
    with open("data/theme_summaries.json", "w") as f:
        json.dump({int(k): v for k,v in summaries.items()}, f, ensure_ascii=False, indent=2)

    # ---------- Build ranked list & diversity-aware seed (A) ----------
    counts = df["theme"].value_counts().to_dict()
    ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    elapsed = time.time() - start_time
    print(f"[Theme Generation] ✅ Completed processing {total_themes} themes in parallel! (took {elapsed:.1f}s)", file=sys.stderr)

    # Ensure we have exactly n_clusters themes (regenerate if needed)
    if len(ranked) < n_clusters:
        print(f"[Theme Generation] Only {len(ranked)} clusters found, regenerating with {n_clusters} clusters...", file=sys.stderr)
        # Regenerate with exact number needed
        from sklearn.cluster import KMeans
        if not FORCE_TFIDF:
            try:
                from sentence_transformers import SentenceTransformer
                if 'model' not in locals():
                    model = SentenceTransformer(emb_model)
                all_embeddings = model.encode(df[text_col].astype(str).tolist(), batch_size=64, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
            except:
                # Fallback to TF-IDF
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.preprocessing import normalize
                vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1,2))
                all_embeddings = vec.fit_transform(df[text_col].astype(str).tolist()).astype("float32")
                all_embeddings = normalize(all_embeddings)
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.preprocessing import normalize
            vec = TfidfVectorizer(max_features=2000, stop_words="english", ngram_range=(1,2))
            all_embeddings = vec.fit_transform(df[text_col].astype(str).tolist()).astype("float32")
            all_embeddings = normalize(all_embeddings)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=1)  # Use n_init=1 for speed
        df["theme"] = kmeans.fit_predict(all_embeddings)
        counts = df["theme"].value_counts().to_dict()
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        
        # Recalculate sentiment counts with new clusters
        pos_counts, neg_counts, neu_counts = {}, {}, {}
        if "sentiment_label" in df.columns:
            for tid, sub in df.groupby("theme"):
                vc = sub["sentiment_label"].value_counts().to_dict()
                pos_counts[int(tid)] = int(vc.get("positive", 0))
                neg_counts[int(tid)] = int(vc.get("negative", 0))
                neu_counts[int(tid)] = int(vc.get("neutral", 0))
        
        # Regenerate TF-IDF keywords for new clusters (quick version)
        tfidf_kws = {}
        for tid, sub in df.groupby("theme"):
            sub_texts = sub[text_col].astype(str).tolist()
            if len(sub_texts) > 200:
                sub_texts = sub_texts[:200]
            from sklearn.feature_extraction.text import TfidfVectorizer
            vec = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1,2))
            X = vec.fit_transform(sub_texts)
            import numpy as np
            mean_matrix = X.mean(axis=0)
            if hasattr(mean_matrix, 'toarray'):
                scores = mean_matrix.toarray().flatten()
            else:
                scores = np.array(mean_matrix).flatten()
            feats = vec.get_feature_names_out()
            top_idx = scores.argsort()[::-1][:10]
            toks = [f for f in feats[top_idx] if f.split()[0].lower() not in _STOP and f.split()[0].lower() not in ['wutangkids', 'grok', 'don', 'flipkart']]
            tfidf_kws[int(tid)] = toks[:6]

    # Build small corpus per cluster for similarity (A)
    texts_by_cluster: Dict[int, List[str]] = {}
    for tid, sub in df.groupby("theme"):
        texts_by_cluster[int(tid)] = sub[text_col].astype(str).head(300).tolist()

    # Build themes_all from ranked list (A)
    themes_all: List[dict] = []
    for k_, count in ranked:
        k = int(k_)
        themes_all.append({
            "id": k,
            "name": theme_names.get(k, f"Theme {k}"),
            "summary": summaries.get(k, ""),
            "tweet_count": int(count),
            "positive": pos_counts.get(k, 0),
            "negative": neg_counts.get(k, 0),
            "neutral": neu_counts.get(k, 0),
        })

    # Diversity-aware selection instead of raw top-N (A)
    themes = _mmr_select(themes_all, texts_by_cluster, n=n_clusters, lam=0.65)

    # Merge similar themes to avoid duplicates, but be less aggressive to preserve 12 themes
    if merge_similar:
        # Use higher threshold (0.85) to only merge truly duplicate themes, not similar ones
        # This ensures we keep distinct themes like "Pricing Concerns" vs "Product Pricing"
        final_themes = _merge_similar_themes(themes, similarity_threshold=0.85)
        # Only return themes with actual content (tweet_count > 0)
        final_themes = [t for t in final_themes if t["tweet_count"] > 0]
        merged_component_ids = set()
        for ft in final_themes:
            merged_component_ids.update(ft.get("component_ids", [ft["id"]]))
        
        print(f"[Theme Generation] After merging: {len(final_themes)} themes (target: {n_clusters})", file=sys.stderr)

        remaining_candidates = [t for t in themes if t["tweet_count"] > 0 and t["id"] not in merged_component_ids]

        # If merging collapses all clusters and we can't reach n_clusters, fall back to original themes
        if len(final_themes) + len(remaining_candidates) < n_clusters:
            final_themes = [{**t, "component_ids": [t["id"]]} for t in themes if t["tweet_count"] > 0]
            final_themes.sort(key=lambda x: x["tweet_count"], reverse=True)
            final_themes = final_themes[:n_clusters]
            merged_component_ids = {cid for ft in final_themes for cid in ft.get("component_ids", [ft["id"]])}
            print("[Theme Generation] ⚠️ Reverting merge to avoid double counting", file=sys.stderr)
            remaining_candidates = [t for t in themes if t["tweet_count"] > 0 and t["id"] not in merged_component_ids]
        
        # If we have fewer than n_clusters after merging, add back themes from original list
        if len(final_themes) < n_clusters:
            # Take top themes from original list to fill up to n_clusters
            existing_ids = set(merged_component_ids)
            remaining = remaining_candidates
            # Sort by tweet count and add until we have n_clusters
            remaining.sort(key=lambda x: x["tweet_count"], reverse=True)
            num_to_add = n_clusters - len(final_themes)
            final_themes.extend({**t, "component_ids": [t["id"]]} for t in remaining[:num_to_add])
            print(f"[Theme Generation] Added {num_to_add} themes to reach {n_clusters} total", file=sys.stderr)
        
        # Ensure we have exactly n_clusters (take top n_clusters by tweet count)
        final_themes.sort(key=lambda x: x["tweet_count"], reverse=True)
        final_themes = final_themes[:n_clusters]
        # Remove component IDs from added themes for consistent structure
        for ft in final_themes:
            ft.setdefault("component_ids", [ft["id"]])
    else:
        # Only return themes with actual content
        final_themes = [{**t, "component_ids": [t["id"]]} for t in themes if t["tweet_count"] > 0]
        final_themes.sort(key=lambda x: x["tweet_count"], reverse=True)
        final_themes = final_themes[:n_clusters]
    
    print(f"[Theme Generation] ✅ Final: {len(final_themes)} themes generated", file=sys.stderr)

    used_llm = bool(client)
    # Before returning, drop component ids to avoid leaking implementation detail
    sanitized_themes = []
    for ft in final_themes:
        theme_copy = dict(ft)
        theme_copy.pop("component_ids", None)
        sanitized_themes.append(theme_copy)

    return {
        "updated_at": pd.Timestamp.utcnow().isoformat(),
        "themes": sanitized_themes,
        "used_llm": used_llm,
        "source_row_count": len(df),
    }