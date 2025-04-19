import streamlit as st
import pickle
import pandas as pd
from recommender_functions import content_recommender

# Load encoders/decoders and data
with open("language_decoder.pkl", "rb") as f:
    language_decoder = pickle.load(f)

with open("director_decoder.pkl", "rb") as f:
    director_decoder = pickle.load(f)

# Inject into function module (simple for now)
import recommender_functions
recommender_functions.language_decoder = language_decoder
recommender_functions.director_decoder = director_decoder

# Load data
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

# --- Streamlit UI ---
st.title("🎬 Content-Based Movie Recommender")

option = st.radio("Choose input type:", ("Movie Title", "Genre", "Language", "Release Year"))
input_type_map = {
    "Movie Title": "title",
    "Genre": "genre",
    "Language": "language",
    "Release Year": "year"
}
input_type = input_type_map[option]

# --- Input UI ---
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
