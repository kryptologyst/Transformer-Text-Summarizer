"""Streamlit dashboard for transformer text summarization."""

from __future__ import annotations

import streamlit as st

from src.data import SAMPLE_TEXTS, load_sample_text
from src.model import TextSummarizer

st.set_page_config(page_title="Text Summarizer", layout="wide")
st.title("Transformer Text Summarizer")

tab1, tab2, tab3 = st.tabs(["Summarize", "Compare Models", "Bullet Points"])

with tab1:
    st.header("Summarize Text")
    col1, col2 = st.columns(2)
    with col1:
        model_choice = st.selectbox("Model", ["bart", "t5", "pegasus", "distilbart"], index=0)
        max_len = st.slider("Max Summary Length", 30, 500, 150)
        min_len = st.slider("Min Summary Length", 10, 200, 40)
    with col2:
        topic = st.selectbox("Sample Topic", list(SAMPLE_TEXTS.keys()), format_func=lambda x: x.replace("_", " ").title())
        custom_text = st.text_area("Or paste your own text", height=150)

    if st.button("Summarize", type="primary"):
        text = custom_text.strip() if custom_text.strip() else load_sample_text(topic)
        with st.spinner("Summarizing..."):
            summarizer = TextSummarizer(model_name=model_choice)
            result = summarizer.summarize(text, max_length=max_len, min_length=min_len)
        st.subheader("Original")
        st.write(text[:500] + ("..." if len(text) > 500 else ""))
        st.metric("Original Words", len(text.split()))
        st.subheader("Summary")
        st.success(result)
        st.metric("Summary Words", len(result.split()))

with tab2:
    st.header("Compare Models")
    topic2 = st.selectbox("Topic", list(SAMPLE_TEXTS.keys()), key="topic2")
    if st.button("Compare", type="primary"):
        text = load_sample_text(topic2)
        models = ["bart", "t5", "distilbart"]
        for m in models:
            with st.spinner(f"Running {m}..."):
                summarizer = TextSummarizer(model_name=m)
                result = summarizer.summarize(text)
            st.subheader(m.upper())
            st.info(result)
            st.caption(f"{len(result.split())} words")

with tab3:
    st.header("Bullet Point Summary")
    topic3 = st.selectbox("Topic", list(SAMPLE_TEXTS.keys()), key="topic3")
    num_bullets = st.slider("Number of Bullets", 2, 8, 3)
    if st.button("Generate Bullets", type="primary"):
        text = load_sample_text(topic3)
        with st.spinner("Generating..."):
            summarizer = TextSummarizer(model_name="bart")
            bullets = summarizer.summarize_bullets(text, num_bullets=num_bullets)
        for i, b in enumerate(bullets, 1):
            st.markdown(f"**{i}.** {b}")
