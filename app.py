# app.py
import streamlit as st
import pandas as pd
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
# Compute embeddings
# ===========================
@st.cache_resource
def compute_embeddings(df, model):
    texts = (
        df["name"].astype(str) + " " +
        df["description"].astype(str) + " " +
        df["cuisine"].astype(str) + " " +
        df["course"].astype(str) + " " +
        df["diet"].astype(str) + " " +
        df["instructions"].astype(str)
    ).tolist()
    return model.encode(texts, show_progress_bar=True)

# ===========================
# Rekomendasi resep
# ===========================
def recommend(query, cuisine, course, diet, df, embeddings, model, top_n=10):
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
st.set_page_config(page_title="Recipe Recommendation App", page_icon="🍴", layout="wide")

st.title("🍴 Recipe Recommendation App")
st.write("Cari resep berdasarkan **description + cuisine + course + diet**")

df = load_data()
model = load_model()
embeddings = compute_embeddings(df, model)

# Input query
query = st.text_input("Masukkan kata kunci (contoh: spicy curry, vegetarian, Italian)", "")

# Dropdown filter
cuisine = st.selectbox("Pilih Cuisine:", ["All"] + sorted(df["cuisine"].dropna().unique()))
course = st.selectbox("Pilih Course:", ["All"] + sorted(df["course"].dropna().unique()))
diet = st.selectbox("Pilih Diet:", ["All"] + sorted(df["diet"].dropna().unique()))

# Slider jumlah rekomendasi
top_n = st.slider("Jumlah rekomendasi:", 5, 20, 10)

# Tampilkan hasil
if st.button("Cari Rekomendasi"):
    if query.strip() == "" and cuisine == "All" and course == "All" and diet == "All":
        st.warning("Masukkan kata kunci atau pilih filter minimal 1.")
    else:
        results, scores = recommend(query, cuisine, course, diet, df, embeddings, model, top_n)
        for i, (idx, row) in enumerate(results.iterrows()):
            st.markdown(f"### 🍽️ {row['name']}")
            if "image_url" in row and pd.notna(row["image_url"]):
                st.image(row["image_url"], width=300)
            st.write(f"**Cuisine:** {row['cuisine']} | **Course:** {row['course']} | **Diet:** {row['diet']}")
            st.write(f"**Description:** {row['description']}")
            st.write(f"**Instructions:** {row['instructions']}")
            st.write(f"**Prep Time:** {row['prep_time']}")
            st.caption(f"Relevance score: {scores[i]:.4f}")
            st.divider()
