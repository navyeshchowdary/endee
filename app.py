import streamlit as st
from model import get_embeddings, model
from data import items
from store import store
from search import search

st.set_page_config(page_title="🎬 Movie Recommender", page_icon="🎬", layout="centered")

st.title("🎬 AI Movie Recommendation System")
st.markdown("Powered by **Endee Vector Database** + Sentence Transformers")
st.divider()

# Load and store embeddings once per session
@st.cache_resource(show_spinner="Loading movie embeddings into Endee...")
def load_data():
    embeddings = get_embeddings(items)
    store(items, embeddings)
    return True

load_data()

# UI
col1, col2 = st.columns([3, 1])
with col1:
    query = st.text_input("🔍 What kind of movie are you in the mood for?",
                          placeholder="e.g. horror, space adventure, romantic love story...")
with col2:
    genre = st.selectbox("Filter by Genre", 
                         ["All", "Sci-Fi", "Fantasy", "Action", "Romance", 
                          "Drama", "Horror", "Thriller", "Animation", "Crime"])

top_k = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

if st.button("🎯 Get Recommendations", use_container_width=True):
    if not query.strip():
        st.warning("Please enter something you're in the mood for!")
    else:
        with st.spinner("Searching..."):
            query_embedding = model.encode([query])[0]
            results = search(query_embedding, top_k=top_k, genre_filter=genre)

        if results:
            st.subheader("🍿 Recommended for you:")
            for i, r in enumerate(results):
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1:
                        st.markdown(f"**{i+1}. {r['title']}**")
                    with col2:
                        st.markdown(f"`{r['genre']}`")
                    with col3:
                        st.markdown(f"⭐ {r['score']}")
                    st.divider()
        else:
            st.info("No results found. Try a different query or remove the genre filter.")