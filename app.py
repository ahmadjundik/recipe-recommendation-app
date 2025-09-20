# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# Load dataset
# ===========================
@st.cache_data
def load_data():
    df = pd.read_csv("data_cuisine.csv")
    df = df.dropna(subset=["name", "description", "cuisine", "course", "diet", "instructions"])
    df.reset_index(drop=True, inplace=True)
    return df

# ===========================
# Load model
# ===========================
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

# ===========================
# Load precomputed embeddings
# ===========================
@st.cache_resource
def load_embeddings():
    return np.load("embeddings.npy")

# ===========================
# Rekomendasi resep
# ===========================
def recommend(query, cuisine, course, diet, df, embeddings, model, top_n=15):
    query_text = query
    if cuisine != "All":
        query_text += " " + cuisine
    if course != "All":
        query_text += " " + course
    if diet != "All":
        query_text += " " + diet

    query_emb = model.encode([query_text])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = sims.argsort()[-top_n:][::-1]
    return df.iloc[top_idx], sims[top_idx]

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="🍴 Recipe Recommendation App", page_icon="🍲", layout="wide")

st.title("🍲 Recipe Recommendation System")
st.write("Rekomendasi makanan berdasarkan **description + cuisine + course + diet** menggunakan Sentence-BERT")

df = load_data()
model = load_model()
embeddings = load_embeddings()

# Pilih makanan sebagai query
query = st.selectbox("Pilih makanan sebagai kata kunci:", sorted(df["name"].unique()))

# Dropdown filter
cuisine = st.selectbox("Pilih Cuisine:", ["All"] + sorted(df["cuisine"].dropna().unique()))
course = st.selectbox("Pilih Course:", ["All"] + sorted(df["course"].dropna().unique()))
diet = st.selectbox("Pilih Diet:", ["All"] + sorted(df["diet"].dropna().unique()))

# Jumlah rekomendasi fix 15
top_n = 15

if st.button("Dapatkan Rekomendasi"):
    results, scores = recommend(query, cuisine, course, diet, df, embeddings, model, top_n)
    st.subheader("🔎 Hasil Rekomendasi:")
    for i, (idx, row) in enumerate(results.iterrows()):
        st.markdown(f"### {i+1}. {row['name']}")
        st.write(f"**Cuisine:** {row['cuisine']} | **Course:** {row['course']} | **Diet:** {row['diet']}")
        st.write(f"**Description:** {row['description']}")
        st.write(f"**Instructions:** {row['instructions']}")
        st.write(f"**Prep Time:** {row['prep_time']}")
        st.caption(f"Relevance score: {scores[i]:.4f}")
        st.divider()
