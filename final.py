import streamlit as st
import pandas as pd
import pickle
import os
import recommender_functions
from recommender_functions import (
    content_recommender,
    collaborative_recommender,
    hybrid_recommender,
    nlp_recommender
)

# --- Load Decoders ---
with open("language_decoder.pkl", "rb") as f:
    language_decoder = pickle.load(f)
with open("director_decoder.pkl", "rb") as f:
    director_decoder = pickle.load(f)

# Inject decoders into functions module
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

# --- Load Collaborative Dataset ---
collab_file = "top_movies_collab_df.csv"
if os.path.exists(collab_file):
    top_movies_collab_df = pd.read_csv(collab_file)
else:
    top_movies_collab_df = pd.DataFrame(columns=["userName", "title", "standardized_score"])

# --- UI Title ---
st.title("🎬 Movie Recommender System")

# --- Method Selection ---
method = st.selectbox("Select Recommendation Method:", [
    "Content-Based", "Collaborative Filtering", "Hybrid", "NLP-Based"
])

# === Content-Based Filtering ===
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
        recs = collaborative_recommender(user_id, top_movies_collab_df)  # ✅ fixed: added ratings_df
        if isinstance(recs, str):
            st.warning(recs)
        else:
            st.subheader("🔍 Recommendations")
            for idx, rec in enumerate(recs["recommendations"], 1):
                st.markdown(f"**{idx}. {rec['title']}** - {rec['score']:.2f}")

# === NLP-Based Filtering ===
elif method == "NLP-Based":
    st.subheader("💬 NLP-Based Filtering")
    description = st.text_area("Describe the type of movie you want to watch:")
    if st.button("Get Recommendations"):
        recs = nlp_recommender(description)
        st.subheader("🔍 Recommendations")
        for idx, rec in enumerate(recs, 1):
            st.markdown(f"**{idx}. {rec['title']}** - {rec['score']}")
            
# === Hybrid Filtering ===
elif method == "Hybrid":
    st.subheader("🧠 Hybrid Filtering")
    st.markdown("#### 👤 User Setup")
    user_id = st.text_input("Enter your username (for new or existing users):")

    st.markdown("#### 🎥 Pick 2 Movies You Like")
    movie_list = sorted(content_df['title'].unique())
    liked_movies = []
    for i in range(1, 3):  # Change here to select only 2 movies
        movie = st.selectbox(f"Select favorite movie {i}", movie_list, key=f"movie_{i}")
        if movie:
            liked_movies.append(movie)

    if st.button("Confirm Favorites and Get Recommendations"):
        new_rows = pd.DataFrame({
            'userName': [user_id] * len(liked_movies),
            'title': liked_movies,
            'standardized_score': [100] * len(liked_movies)
        })
        top_movies_collab_df = pd.concat([top_movies_collab_df, new_rows], ignore_index=True).drop_duplicates(subset=['userName', 'title'], keep='last')
        top_movies_collab_df.to_csv(collab_file, index=False)

        hybrid_recs = hybrid_recommender(
            user_id=user_id,
            movie_title=liked_movies[0],  # First favorite movie
            content_df=content_df,
            content_features=content_features,
            top_movies_collab_df=top_movies_collab_df
        )

        st.subheader("🔍 Hybrid Recommendations")
        for idx, (title, score) in enumerate(hybrid_recs, 1):
            movie_info = content_df[content_df['title'] == title]
            if not movie_info.empty:
                movie = movie_info.iloc[0]
                genres = [col for col in feature_cols if col not in ['runtimeMinutes', 'director', 'originalLanguage'] and movie.get(col, 0) == 1]
                st.markdown(f"**{idx}. {title}**")
                st.markdown(f"- Year: {movie.get('year', 'N/A')}")
                st.markdown(f"- Genres: {', '.join(genres)}")
                st.markdown(f"- Director: {director_decoder.get(movie.get('director'), 'Unknown')}")
                st.markdown(f"- Language: {language_decoder.get(movie.get('originalLanguage'), 'Unknown')}")
                st.markdown(f"- Runtime: {movie.get('runtimeMinutes', 'N/A')} minutes")
                st.markdown(f"- TomatoMeter: {movie.get('tomatoMeter', 'N/A')}%")
                st.markdown("---")


