
import os
import pandas as pd
import streamlit as st
from datetime import date
from src.features.clean import load_any, basic_date_parse
from src.llm.summary import summarize_tweets

st.set_page_config(page_title="Walmart Social Listener", layout="wide")

st.title("🛒 Walmart Social Media Listener — Executive Summary")
st.caption("Filter by date & keyword → generate a business-ready LLM summary.")

data_file = st.sidebar.text_input("Data file path (CSV/XLSX/Parquet)", "data/tweets_clean.parquet")
keyword = st.sidebar.text_input("Keyword filter (optional)", "")
col1, col2 = st.sidebar.columns(2)
start = col1.date_input("Start date", value=date(2025, 7, 1))
end = col2.date_input("End date", value=date(2025, 7, 31))

if st.sidebar.button("Load Data"):
    try:
        df = load_any(data_file)
        df = basic_date_parse(df)
        st.session_state["df"] = df
        st.success(f"Loaded {len(df):,} rows.")
    except Exception as e:
        st.error(f"Failed to load: {e}")

df = st.session_state.get("df")
if df is not None:
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("✨ Summarize"):
        with st.spinner("Calling LLM..."):
            res = summarize_tweets(
                df=df,
                start_date=str(start),
                end_date=str(end),
                keyword=keyword or None,
            )
        st.subheader("Executive Brief + JSON")
        st.write(res["executive_text"])

        st.subheader("Structured (parsed)")
        st.json(res["structured"])

        st.subheader("Stats")
        st.json(res["stats"])
else:
    st.info("Provide a valid file path in the sidebar and click 'Load Data'.")
