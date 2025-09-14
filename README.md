# Quick description: LLM-powered-social-listening-system
Building an LLM-powered social listening system to track brand sentiment over time, aspect sentiment (staff, delivery, price, UX), event impact (Black Friday, campaigns), influencer reach + sentiment, and bot filtering, benchmarking with competitors sentiments.

# LLM/DL Social Listening — 1-Week Brand Intelligence (Bluesky)

> Turn **one week** of public Bluesky posts about a brand (e.g., Walmart) into **decision-grade insights**: trend & spikes, *why* sentiment moved (aspects), emerging themes, event impact, influencer impact, sponsored vs organic, bot filtering, and an exec summary. Ships as a **clickable web app** (Streamlit/Gradio) with CSV/XLSX exports.

---

## Table of Contents
- [Description / Overview](#description--overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Status](#project-status)
- [Future Roadmap (2-Week Sprints)](#future-roadmap-2week-sprints)
- [Authors & Acknowledgements](#authors--acknowledgements)

---

## Description / Overview
This project is a **modern AI social listening pipeline** (Transformers, sentence embeddings, deep time-series, schema-constrained LLM) that ingests public Bluesky posts for a **1-week window** and outputs:

- **Post-level sentiment** with calibrated confidence  
- **Aspect sentiment** at **sentence** level (price, delivery, returns, staff, app/UX)  
- **Unsupervised themes/topics** (emerging conversations)  
- **Event impact** vs a counterfactual baseline (uplift + uncertainty)  
- **Influencer impact & roles** (promoter/passive/detractor)  
- **Sponsored vs organic** detection & performance  
- **Bot filtering** to clean KPIs  
- **Executive brief** generated from computed stats (no hallucinated numbers)

All charts support a **confidence slider** and **bot toggle**, and each tab offers **CSV/XLSX exports**.

---

## Key Features
- **Sentiment trend (1-week):** POS/NEU/NEG lines + donut; **confidence slider** to trade coverage for quality  
- **Aspect-based sentiment (sentence level):** pricing, delivery, returns, staff, app/UX with **clickable evidence**  
- **Themes/Topics (unsupervised):** embedding-based clusters, **auto-named** by a small LLM; “**emerging themes**” panel  
- **Event impact:** **actual vs counterfactual** with **95% CI** uplift; pre/during/post comparison table  
- **Influencer modeling:** **two-tower** neural ranking by **predicted impact**; weekly **Promoter/Passive/Detractor** roles  
- **Sponsored detection:** high-precision classifier; **within-author lift** for paid vs organic posts  
- **Bot filter toggle:** fusion model removes high-risk accounts; **before/after** metrics recompute live  
- **Executive brief (LLM):** schema-constrained JSON → human-readable summary; numbers sourced from pipeline  
- **Exports & traceability:** per-tab **CSV/XLSX**, evidence sentences/posts, thresholds and settings logged

---

## Tech Stack
**Language & Core**
- Python 3.10+, pandas, numpy, scikit-learn

**Data Source**
- Bluesky / AT Protocol (`atproto` Python client)

**NLP / Modeling**
- **Sentiment:** DeBERTa-v3 / RoBERTa (**Hugging Face**, **PyTorch**) with **LoRA** adapters; **temperature calibration**  
- **Aspect Sentiment (ABSA):** multi-task Transformer (aspect detection + polarity) at **sentence** level  
- **Embeddings:** sentence-transformers (**e5-base-v2**, **bge-base-en-v1.5**)  
- **Topics:** **BERTopic** (UMAP + HDBSCAN + c-TF-IDF); topic naming via small **LLM** (JSON schema)  
- **Time-Series:** **Temporal Fusion Transformer (TFT)** or **NeuralProphet** with event regressors  
- **Influencer Ranking:** **two-tower** (content tower + metadata tower) with **BPR** ranking loss  
- **Sponsored Detection:** DeBERTa-v3 + LoRA (binary) at **high precision**  
- **Bot Detection:** fusion MLP over content/behavior/profile (+ optional **GNN** if graph available)  
- **Executive Brief:** hosted **LLM** with **function-calling / JSON schema**, temperature=0

**App & Deployment**
- Streamlit or Gradio; deploy on **Hugging Face Spaces (CPU)**

**Storage & Artifacts**
- Parquet/CSV/XLSX; optional 🤗 Datasets; cached embeddings (npz/parquet)

**Visualization**
- Plotly / Altair / Matplotlib

---

## Project Status
----

## Future Roadmap (2-Week Sprints)

### Phase 1 — Foundations & Demo Readiness
