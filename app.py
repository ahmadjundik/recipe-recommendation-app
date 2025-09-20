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

def top10_global_similarity(df, embeddings):
    # hitung similarity matrix
    sims = cosine_similarity(embeddings)
    np.fill_diagonal(sims, -1)  # supaya resep tidak dibandingkan dengan dirinya sendiri

    pairs = []
    n = sims.shape[0]
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((i, j, sims[i, j]))

    # urutkan berdasarkan similarity tertinggi
    top_pairs = sorted(pairs, key=lambda x: x[2], reverse=True)[:10]
    return [(df.iloc[i]["name"], df.iloc[j]["name"], score) for i, j, score in top_pairs]

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="Recipe Recommendation App", page_icon="🍴", layout="wide")

st.title("🍴 Recipe Recommendation App")
st.write("Cari resep berdasarkan **description + cuisine + course + diet**")

df = load_data()
model = load_model()
embeddings = load_embeddings()

# Pilihan mode rekomendasi
mode = st.radio(
    "Pilih jenis rekomendasi:",
    ["🔎 Berdasarkan Resep", "⭐ Top 10 per Kategori", "🔥 Top 10 Global Similarity"]
)

if mode == "🔎 Berdasarkan Resep":
    query = st.selectbox("Pilih Resep:", sorted(df["name"].unique()))
    cuisine = st.selectbox("Pilih Cuisine:", ["All"] + sorted(df["cuisine"].dropna().unique()))
    course = st.selectbox("Pilih Course:", ["All"] + sorted(df["course"].dropna().unique()))
    diet = st.selectbox("Pilih Diet:", ["All"] + sorted(df["diet"].dropna().unique()))

    if st.button("Dapatkan Rekomendasi"):
        results, scores = recommend(query, cuisine, course, diet, df, embeddings, model, top_n=15)
        for i, (idx, row) in enumerate(results.iterrows()):
            st.markdown(f"### 🍽️ {row['name']}")
            st.write(f"**Cuisine:** {row['cuisine']} | **Course:** {row['course']} | **Diet:** {row['diet']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Instructions:** {row['instructions'][:300]}...")
            st.caption(f"Relevance score: {scores[i]:.4f}")
            st.divider()

elif mode == "⭐ Top 10 per Kategori":
    kategori = st.selectbox("Pilih kategori:", ["Cuisine", "Course", "Diet"])
    nilai = st.selectbox(f"Pilih {kategori}:", sorted(df[kategori.lower()].dropna().unique()))

    if st.button("Tampilkan Top 10"):
        subset = df[df[kategori.lower()] == nilai].head(10)
        for _, row in subset.iterrows():
            st.markdown(f"### 🍽️ {row['name']}")
            st.write(f"**Cuisine:** {row['cuisine']} | **Course:** {row['course']} | **Diet:** {row['diet']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Instructions:** {row['instructions'][:300]}...")
            st.divider()

elif mode == "🔥 Top 10 Global Similarity":
    if st.button("Tampilkan Top 10 Global"):
        pairs = top10_global_similarity(df, embeddings)
        for rank, (r1, r2, score) in enumerate(pairs, 1):
            st.markdown(f"**#{rank}. {r1} ↔ {r2}**")
            st.caption(f"Similarity score: {score:.4f}")
            st.divider()
