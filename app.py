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
# Recommendation function
# ===========================
def recommend_by_recipe(query, df, embeddings, model, top_n=15):
    query_row = df[df["name"] == query].iloc[0]
    query_text = query_row["description"] + " " + str(query_row["cuisine"]) + " " + str(query_row["course"]) + " " + str(query_row["diet"])
    query_emb = model.encode([query_text])
    sims = cosine_similarity(query_emb, embeddings)[0]
    top_idx = sims.argsort()[-top_n-1:][::-1]  # +1 karena termasuk dirinya sendiri
    top_idx = [i for i in top_idx if df.iloc[i]["name"] != query][:top_n]  # hilangkan dirinya
    return df.iloc[top_idx], sims[top_idx]

def recommend_by_category(category_type, category_value, df, top_n=10):
    if category_type == "Cuisine":
        results = df[df["cuisine"] == category_value].head(top_n)
    elif category_type == "Course":
        results = df[df["course"] == category_value].head(top_n)
    elif category_type == "Diet":
        results = df[df["diet"] == category_value].head(top_n)
    else:
        results = pd.DataFrame()
    return results

# ===========================
# Streamlit UI
# ===========================
st.set_page_config(page_title="🍴 Recipe Recommendation App", page_icon="🍲", layout="wide")

st.title("🍲 Recipe Recommendation System")
st.write("Rekomendasi resep berdasarkan **Sentence-BERT** dan filter kategori")

df = load_data()
model = load_model()
embeddings = load_embeddings()

# Pilihan mode rekomendasi
mode = st.radio(
    "Pilih jenis rekomendasi:",
    ["🔎 Berdasarkan Resep", "⭐ Top 10 Resep per Kategori"]
)

if mode == "🔎 Berdasarkan Resep":
    query = st.selectbox("Pilih resep:", sorted(df["name"].unique()))
    if st.button("Dapatkan Rekomendasi"):
        results, scores = recommend_by_recipe(query, df, embeddings, model, top_n=15)
        st.subheader("🔎 Rekomendasi Resep Serupa:")
        for i, (idx, row) in enumerate(results.iterrows()):
            st.markdown(f"### {i+1}. {row['name']}")
            st.write(f"**Cuisine:** {row['cuisine']} | **Course:** {row['course']} | **Diet:** {row['diet']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Instructions:** {row['instructions']}")
            st.write(f"**Prep Time:** {row['prep_time']}")
            st.caption(f"Relevance score: {scores[i]:.4f}")
            st.divider()

elif mode == "⭐ Top 10 Resep per Kategori":
    category_type = st.selectbox("Pilih kategori:", ["Cuisine", "Course", "Diet"])
    if category_type == "Cuisine":
        category_value = st.selectbox("Pilih Cuisine:", sorted(df["cuisine"].dropna().unique()))
    elif category_type == "Course":
        category_value = st.selectbox("Pilih Course:", sorted(df["course"].dropna().unique()))
    else:
        category_value = st.selectbox("Pilih Diet:", sorted(df["diet"].dropna().unique()))
    
    if st.button("Tampilkan Resep"):
        results = recommend_by_category(category_type, category_value, df, top_n=10)
        st.subheader(f"⭐ Top 10 Resep untuk {category_type}: {category_value}")
        for i, (idx, row) in enumerate(results.iterrows()):
            st.markdown(f"### {i+1}. {row['name']}")
            st.write(f"**Cuisine:** {row['cuisine']} | **Course:** {row['course']} | **Diet:** {row['diet']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Instructions:** {row['instructions']}")
            st.write(f"**Prep Time:** {row['prep_time']}")
            st.divider()
