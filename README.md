# 📰 Fake News Detection System

A machine learning-powered web application that classifies news articles as **Real** or **Fake** using Natural Language Processing (NLP). This project utilizes a Passive Aggressive Classifier and is deployed via a Streamlit web interface.

## 🚀 Overview
In an era of rapid information spread, misinformation is a significant challenge. This project aims to provide a tool that analyzes the linguistic patterns of news articles to determine their authenticity. 

The model was trained on a dataset of 20,000 articles, achieving an accuracy of approximately **99%**.

## 🛠️ Technology Stack
- **Language:** Python 3.x
- **Libraries:** Pandas, NumPy, Scikit-learn
- **NLP Techniques:** TF-IDF Vectorization, Stop-word removal, Text Cleaning
- **Model:** Passive Aggressive Classifier
- **Web Framework:** Streamlit
- **Environment:** Virtualenv (venv)

## 📁 Project Structure
```text
fake_news_project/
├── .venv/               # Virtual environment (ignored by Git)
├── artifacts/           # Saved model and vectorizer
│   ├── fake_news_model.pkl
│   └── tfidf_vectorizer.pkl
├── data/
│   └── fake_news_dataset.csv  # Raw dataset
├── app.py               # Streamlit web application
├── requirements.txt     # Project dependencies
├── .gitignore           # Files excluded from version control
└── README.md            # Project documentation
