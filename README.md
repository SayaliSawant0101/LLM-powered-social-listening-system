# LLM-powered-social-listening-system

Building an LLM-powered social listening system to track **Walmart’s Twitter trends** — monitoring brand sentiment over time, aspect sentiment (staff, delivery, price, UX), event impact (Black Friday, campaigns), influencer reach + sentiment, and bot filtering, benchmarking against competitors and executive brief.

---

## LLM/DL Social Listening — Quaterly Brand Intelligence Report

> Turn **3-months** of public Twitter posts about **Walmart** into **decision-grade insights**: trend & spikes, *why* sentiment moved (aspects), emerging themes, event impact, influencer impact, sponsored vs organic, bot filtering, competitor benchmarking, and an exec summary.  
> Ships as a **clickable web app** (Streamlit/Gradio) with CSV/XLSX exports.

---

## Table of Contents
- [Description / Overview](#description--overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Status](#project-status)
- [Future Roadmap (2-Week Sprints)](#future-roadmap-2-week-sprints)
- [Authors & Acknowledgements](#authors--acknowledgements)

---

## Description / Overview
This project is a **modern AI social listening pipeline** (Transformers, sentence embeddings, deep time-series, schema-constrained LLM) that ingests public Twitter posts for a **3-months window** and outputs insights specifically for **Walmart**:

- **Post-level sentiment** with calibrated confidence  
- **Aspect sentiment** at **sentence** level (price, delivery, returns, staff, app/UX)  
- **Unsupervised themes/topics** (emerging conversations)  
- **Event impact** vs a counterfactual baseline (uplift + uncertainty)  
- **Influencer impact & roles** (promoter/passive/detractor)  
- **Sponsored vs organic** detection & performance  
- **Bot filtering** to clean KPIs  
- **Competitor benchmarking** (Walmart vs Target, Amazon, Costco, etc.)  
- **Executive brief** generated from computed stats (no hallucinated numbers)

All charts support a **confidence slider** and **bot toggle**, and each tab offers **CSV/XLSX exports**.

---

## Key Features
- **Sentiment trend (1-week):** POS/NEU/NEG lines + donut; **confidence slider** to trade coverage for quality; **Walmart vs competitor benchmarking**  
- **Aspect-based sentiment (sentence level):** pricing, delivery, returns, staff, app/UX with **clickable evidence**; compare **Walmart’s performance vs Target, Amazon, Costco**  
- **Themes/Topics (unsupervised):** embedding-based clusters, **auto-named** by a small LLM; “**emerging themes**” panel with **cross-brand comparisons**  
- **Event impact:** **actual vs counterfactual** with **95% CI** uplift; pre/during/post comparison table; e.g., **Black Friday uplift for Walmart vs competitors**  
- **Influencer modeling:** **two-tower** neural ranking by **predicted impact**; weekly **Promoter/Passive/Detractor** roles; benchmark **Walmart influencers vs Target/Amazon/Costco**  
- **Sponsored detection:** high-precision classifier; **within-author lift** for paid vs organic posts; compare **Walmart’s campaigns vs competitors**  
- **Bot filter toggle:** fusion model removes high-risk accounts; **before/after** metrics recompute live; cross-brand bot prevalence (Walmart vs peers)  
- **Competitor benchmarking (Walmart vs Target, Amazon, Costco, etc.):** integrated throughout sentiment, aspects, events, influencers, and bots  
- **Executive brief (LLM):** schema-constrained JSON → human-readable summary; includes **Walmart vs competitor benchmarking**  
- **Exports & traceability:** per-tab **CSV/XLSX**, evidence sentences/posts, thresholds and settings logged

---

## Tech Stack

**Language & Core**
- Python 3.10+, pandas, numpy, scikit-learn  

**Data Source**
- Twitter API (via Apify or direct API access)  

**NLP / Modeling**
- **Sentiment:** DeBERTa-v3 / RoBERTa (**Hugging Face**, **PyTorch**) with **LoRA** adapters; **temperature calibration**; benchmarking Walmart vs Target, Amazon, Costco  
- **Aspect Sentiment (ABSA):** multi-task Transformer (aspect detection + polarity) at **sentence** level; outputs per-brand benchmarking for Walmart vs peers  
- **Embeddings:** sentence-transformers (**e5-base-v2**, **bge-base-en-v1.5**)  
- **Topics:** **BERTopic** (UMAP + HDBSCAN + c-TF-IDF); topic naming via small **LLM** (JSON schema); supports **Walmart vs competitor theme overlap**  
- **Time-Series:** **Temporal Fusion Transformer (TFT)** or **NeuralProphet** with event regressors; benchmarking event uplift for Walmart vs competitors  
- **Influencer Ranking:** **two-tower** (content tower + metadata tower) with **BPR** ranking loss; includes influencer tiering for Walmart vs Target/Amazon/Costco  
- **Sponsored Detection:** DeBERTa-v3 + LoRA (binary) at **high precision**, with campaign benchmarking across brands  
- **Bot Detection:** fusion MLP over content/behavior/profile (+ optional **GNN** if graph available); Walmart vs competitor bot prevalence reports  
- **Executive Brief:** hosted **LLM** with **function-calling / JSON schema**, temperature=0; includes **competitive benchmarking section (Walmart vs Target, Amazon, Costco)**  

**App & Deployment**
- Streamlit or Gradio; deploy on **Hugging Face Spaces (CPU)**  

**Storage & Artifacts**
- Parquet/CSV/XLSX; optional 🤗 Datasets; cached embeddings (npz/parquet)  

**Visualization**
- Plotly / Altair / Matplotlib  

---

## Project Status
🚧 In progress — initial ingestion of **Walmart Twitter data** and exploring baseline sentiment modeling is done.

---

## Future Roadmap (2-Week Sprints) - 
Track Project Roadmap on Notion: https://www.notion.so/2799430f209580cba14dd7bcab567429?v=2799430f209580128c67000ce95bbbfd&source=copy_link

<details>
<summary><b>Sprint 1 (Weeks 1–2) — Foundations</b></summary>

- Data ingestion pipeline (Twitter API/Apify → S3 → Glue → Athena)  
- Baseline post-level sentiment (DeBERTa/RoBERTa + LoRA fine-tuning)  
- Aspect-based sentiment module (delivery, pricing, staff, UX)  

</details>

<details>
<summary><b>Sprint 2 (Weeks 3–4) — Aspects & Events</b></summary>

- Topic modeling with BERTopic + LLM-generated topic labels  
- Event impact modeling (NeuralProphet / TFT for Black Friday case)
- Influencer modeling (two-tower neural ranker: content × profile) 

</details>

<details>
<summary><b>Sprint 3 (Weeks 5–6) — Influencers & Bots</b></summary>

- Bot detection (text + behavioral features classifier)  
- Executive brief generator (LLM-powered JSON schema reports)  

</details>

<details>
<summary><b>Sprint 4 (Weeks 7–8) — Executive & Benchmarking</b></summary>

- Walmart vs competitor benchmarking (all modules integrated)  
- Final evaluation, documentation, and presentation  
- Competitor benchmarking for Walmart — integrated executive-level report

</details>

---

## Authors & Acknowledgements

**Author:** [Sayali Sawant](https://github.com/SayaliSawant0101)  
- **Collaborator:** [Harini Balaji](https://github.com/Harini-Balaji11)  
- **Collaborator:** [Darsh Joshi](https://github.com/darshjoshi)  
- **Acknowledgements:**   
  - Open-source model providers (Hugging Face, PyTorch, BERTopic, sentence-transformers)  
  - AWS credits & Hugging Face Spaces for compute/deployment  
  - Twitter API / Apify community  
