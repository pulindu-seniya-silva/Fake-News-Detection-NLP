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

# 3. The Cleaning Function (Must match your training logic)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 4. User Interface
st.title("📰 Fake News Detection System")
st.markdown("Enter the news article or headline below to verify its authenticity.")

# Text input area
user_input = st.text_area("Paste News Here:", height=200, placeholder="Type or paste the news article content...")

if st.button("Analyze News"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        # Step A: Clean the input
        cleaned_input = clean_text(user_input)
        
        
        # Step B: Vectorize (Transform) the text
        # We only use .transform(), NOT .fit_transform()
        vectorized_input = vectorizer.transform([cleaned_input])
        
        # Step C: Prediction
        prediction = model.predict(vectorized_input)[0]
        
        # Step D: Display Results
        st.subheader("Result:")
        if prediction == 1:
            st.error("🚨 WARNING: This news article is likely FAKE.")
        else:
            st.success("✅ RELIABLE: This news article appears to be REAL.")

# Footer
st.info("Note: This model is a baseline tool and should be used for educational purposes.")