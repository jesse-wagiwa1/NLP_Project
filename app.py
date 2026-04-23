import streamlit as st
import pickle
import re
import pandas as pd
import matplotlib.pyplot as plt

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Load vectorizer
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Load dataset (for graphs)
df = pd.read_csv("language_data.csv")

# Preprocess
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Predict
def predict_language(text):
    text = preprocess_text(text)
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    probs = model.predict_proba(vec)[0]
    return prediction, probs

# UI
st.title("🌍 Language Identification System")

# Input
user_input = st.text_area("Enter text:")

if st.button("Predict"):
    pred, probs = predict_language(user_input)
    st.success(f"Predicted Language: {pred}")

    # Probability graph
    st.subheader("Prediction Confidence")
    labels = model.classes_

    fig, ax = plt.subplots()
    ax.bar(labels, probs)
    ax.set_ylabel("Probability")
    ax.set_xlabel("Language")
    st.pyplot(fig)

# -----------------------------
# 📊 DATASET GRAPH
# -----------------------------
st.subheader("Dataset Distribution")

lang_counts = df['language'].value_counts()

fig2, ax2 = plt.subplots()
ax2.bar(lang_counts.index, lang_counts.values)
ax2.set_xlabel("Language")
ax2.set_ylabel("Count")

st.pyplot(fig2)