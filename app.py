import streamlit as st
import pickle
import re

# 1. Page Configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰")

# 2. Load the Artifacts
@st.cache_resource # This keeps the model in memory so it doesn't reload every time
def load_assets():
    with open('artifacts/fake_news_model.pkl', 'rb') as m_file:
        model = pickle.load(m_file)
    with open('artifacts/tfidf_vectorizer.pkl', 'rb') as v_file:
        vectorizer = pickle.load(v_file)
    return model, vectorizer

model, vectorizer = load_assets()

