import streamlit as st
import pickle
import pandas as pd
from recommender_functions import content_recommender, collaborative_recommender, hybrid_recommender, nlp_recommender

# --- Load Decoders ---
with open("language_decoder.pkl", "rb") as f:
    language_decoder = pickle.load(f)

with open("director_decoder.pkl", "rb") as f:
    director_decoder = pickle.load(f)

# Inject into module if needed
import recommender_functions
recommender_functions.language_decoder = language_decoder
recommender_functions.director_decoder = director_decoder

# --- Load Datasets ---
content_df = pd.read_csv("content_df.csv")
feature_cols = [
    'runtimeMinutes', 'director', 'originalLanguage',
    'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Entertainment', 'Faith & spirituality',
    'Fantasy', 'Health & wellness', 'History', 'Horror', 'Kids & family',
    'LGBTQ+', 'Music', 'Nature', 'Other', 'Reality', 'Romance', 'Sci-fi', 'Sports',
    'Variety Show', 'War', 'Western'
]
content_features = content_df[feature_cols]

# --- UI: Method Selection ---
st.title("🎬 Movie Recommender System")

method = st.selectbox("Select Recommendation Method:", [
    "Content-Based", "Collaborative Filtering", "Hybrid", "NLP-Based"
])

# === Content-Based Recommender ===
if method == "Content-Based":
    st.subheader("🎯 Content-Based Filtering")
    option = st.radio("Choose input type:", ("Movie Title", "Genre", "Language", "Release Year"))
    input_type_map = {
        "Movie Title": "title",
        "Genre": "genre",
        "Language": "language",
        "Release Year": "year"
    }
    input_type = input_type_map[option]

    if input_type == "genre":
        genres = [col for col in content_features.columns if col not in ['runtimeMinutes', 'director', 'originalLanguage']]
        user_input = st.selectbox("Select Genre", genres)
    elif input_type == "language":
        languages = sorted(set(language_decoder.values()))
        user_input = st.selectbox("Select Language", languages)
    elif input_type == "year":
        user_input = st.number_input("Enter Release Year", min_value=1900, max_value=2100, step=1)
    else:
        user_input = st.text_input("Enter Movie Title")

    if st.button("Get Recommendations"):
        result = content_recommender(user_input, content_df, content_features, top_n=5, input_type=input_type)

        if isinstance(result, str):
            st.warning(result)
        else:
            if result['movie_details']:
                st.subheader("🎥 Selected Movie")
                movie = result['movie_details']
                st.markdown(f"**Title**: {movie['title']}")
                st.markdown(f"**Genres**: {', '.join(movie['genres'])}")
                st.markdown(f"**Year**: {movie['year']}")
                st.markdown(f"**Director**: {movie['director']}")
                st.markdown(f"**Language**: {movie['original_language']}")
                st.markdown(f"**Runtime**: {movie['runtime_minutes']} minutes")

            st.subheader("🔍 Top Recommendations")
            for idx, rec in enumerate(result['recommendations'], 1):
                st.markdown(f"**{idx}. {rec['title']}**")
                st.markdown(f"- Genres: {', '.join(rec['genres'])}")
                st.markdown(f"- Year: {rec['year']}")
                st.markdown(f"- Director: {rec['director']}")
                st.markdown(f"- Language: {rec['original_language']}")
                st.markdown(f"- Runtime: {rec['runtime_minutes']} minutes")
                st.markdown(f"- TomatoMeter: {rec['tomatoMeter']}%")
                st.markdown("---")

# === Collaborative Filtering ===
elif method == "Collaborative Filtering":
    st.subheader("👥 Collaborative Filtering")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Get Recommendations"):
        recs = collaborative_recommender(user_id)
        st.subheader("🔍 Recommendations")
        for idx, rec in enumerate(recs, 1):
            st.markdown(f"**{idx}. {rec['title']}** - {rec['score']}")

# === Hybrid Filtering ===
elif method == "Hybrid":
    st.subheader("🧠 Hybrid Filtering")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)
    if st.button("Get Recommendations"):
        recs = hybrid_recommender(user_id)
        st.subheader("🔍 Recommendations")
        for idx, rec in enumerate(recs, 1):
            st.markdown(f"**{idx}. {rec['title']}** - {rec['score']}")

# === NLP-Based Filtering ===
elif method == "NLP-Based":
    st.subheader("💬 NLP-Based Filtering")
    description = st.text_area("Describe the type of movie you want to watch:")
    if st.button("Get Recommendations"):
        recs = nlp_recommender(description)
        st.subheader("🔍 Recommendations")
        for idx, rec in enumerate(recs, 1):
            st.markdown(f"**{idx}. {rec['title']}** - {rec['score']}")
