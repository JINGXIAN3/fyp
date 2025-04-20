import streamlit as st
import pandas as pd
import pickle
import os
import recommender_functions
from recommender_functions import (
    content_recommender,
    collaborative_recommender,
    nlp_recommender,
    get_next_user_id,
    get_rating_matrix,
    update_rating_matrix,
    hybrid_recommender,
)

# --- Load Decoders ---
with open("language_decoder.pkl", "rb") as f:
    language_decoder = pickle.load(f)
with open("director_decoder.pkl", "rb") as f:
    director_decoder = pickle.load(f)

# Inject decoders into functions module
recommender_functions.language_decoder = language_decoder
recommender_functions.director_decoder = director_decoder

# --- Load Content-Based Datasets ---
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

# --- Load NLP Dataset ---
nlp_df = pd.read_csv("nlp_df.csv")

# --- UI Title ---
st.title("🎬 Movie Recommender System")

# --- Method Selection ---
method = st.selectbox("Select Recommendation Method:", [
    "Content-Based", "Collaborative", "NLP-Based","Hybrid"
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
elif method == "Collaborative":
    st.subheader("👥 Collaborative Filtering")
    user_id = st.number_input("Enter User ID:", min_value=1, step=1)

    if st.button("Get Recommendations"):
        recs = collaborative_recommender(user_id, top_movies_collab_df) 
        if isinstance(recs, str):
            st.warning(recs)
        else:
            st.subheader("🔍 Recommendations")
            for idx, rec in enumerate(recs["recommendations"], 1):
                st.markdown(f"**{idx}. {rec['title']}** - {rec['score']:.2f} %")

# === NLP-Based Filtering ===
elif method == "NLP-Based":
    st.subheader("💬 NLP-Based Filtering")
    description = st.text_area("Describe the type of movie you want to watch:")
    
    if st.button("Get Recommendations"):
        recs = nlp_recommender(description, nlp_df, language_decoder, director_decoder, w2v_model)
        st.subheader("🔍 Recommendations")
        for idx, rec in enumerate(recs.itertuples(), 1):
            st.markdown(f"**{idx}. {rec.title}** - Similarity Score: {rec.similarity:.2f}")
            
# === Hybrid Filtering ===
elif method == "Hybrid":
    st.subheader("🎬 Hybrid Movie Recommender")
    
    has_id = st.radio("Do you have a user ID?", ("Yes", "No"))

    if has_id == "Yes":
        user_id = st.text_input("Enter your user ID:")
        movie_title = st.selectbox("Select a movie you like:", sorted(content_df['title'].unique()))

        if st.button("Get Hybrid Recommendations"):
            recommendations = hybrid_recommender(
                user_id, movie_title,
                content_df, content_features,
                top_movies_collab_df  # Passing the DataFrame directly
            )

            # Display recommendations
            st.subheader("🔍 Recommendations")
            for idx, (title, score) in enumerate(recommendations, 1):
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

    else:
        user_id = get_next_user_id(rating_matrix)  # Auto-assign new user ID
        st.success(f"Your new user ID is: {user_id}")
        
        st.write("Select **two** movies you like and rate them (0-100):")

        movie1 = st.selectbox("Movie 1:", sorted(content_df['title'].unique()), key="movie1")
        score1 = st.slider("Score for Movie 1:", 0, 100, 80, key="score1")

        movie2 = st.selectbox("Movie 2:", sorted(content_df['title'].unique()), key="movie2")
        score2 = st.slider("Score for Movie 2:", 0, 100, 90, key="score2")

        if st.button("Get Hybrid Recommendations"):
            if user_id not in rating_matrix.index:
                rating_matrix.loc[user_id] = [0] * rating_matrix.shape[1]

            # Update rating matrix with the new user's input
            rating_matrix = update_rating_matrix(rating_matrix, user_id, [(movie1, score1/100), (movie2, score2/100)])

            # Convert the updated rating_matrix back to top_movies_collab_df format
            updated_df = []
            for user in rating_matrix.index:
                for movie in rating_matrix.columns:
                    score = rating_matrix.loc[user, movie]
                    if score > 0:
                        updated_df.append({'userName': user, 'title': movie, 'standardized_score': score})
            
            temp_collab_df = pd.DataFrame(updated_df)
            
            recommendations = hybrid_recommender(
                user_id, movie1,
                content_df, content_features,
                temp_collab_df  # Use the updated DataFrame
            )

            # Display recommendations
            st.subheader("🔍 Recommendations")
            for idx, (title, score) in enumerate(recommendations, 1):
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

            # Update top_movies_collab_df with the new rating entries
            new_rows = [
                {'userName': user_id, 'title': movie1, 'standardized_score': score1/100},
                {'userName': user_id, 'title': movie2, 'standardized_score': score2/100}
            ]
            top_movies_collab_df = pd.concat([top_movies_collab_df, pd.DataFrame(new_rows)], ignore_index=True)
