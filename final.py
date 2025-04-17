import streamlit as st
import pandas as pd
import numpy as np
from content_logic import content_recommender, evaluate_precision, get_language_decoder, get_director_decoder

# Page configuration
st.set_page_config(
    page_title="Movie Recommender System",
    page_icon="🎬",
    layout="wide"
)

# Get decoders
language_decoder = get_language_decoder()
director_decoder = get_director_decoder()

# Load data
@st.cache_data
def load_data():
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
    return content_df, content_features

content_df, content_features = load_data()

# Title
st.title("🎬 Movie Recommender System")

# Sidebar
option = st.sidebar.radio(
    "Choose Recommendation Method:",
    (
        "Content-Based Filtering",
        "Collaborative Filtering",
        "Hybrid Filtering"
    )
)

if option == "Content-Based Filtering":
    st.header("🎯 Content-Based Filtering")
    
    # Create tabs for different filter types
    tab1, tab2, tab3, tab4 = st.tabs(["Movie Title", "Genre", "Language", "Year"])
    
    # Movie Title Tab
    with tab1:
        st.subheader("Find movies similar to a title you enjoyed")
        movie_titles = sorted(content_df['title'].dropna().unique())
        selected_title = st.selectbox("Select a movie", movie_titles)
        
        if st.button("Get Recommendations", key="title_btn"):
            result = content_recommender(selected_title, content_df, content_features, top_n=5, input_type='title')
            
            if isinstance(result, str):
                st.error(result)
            else:
                if result.get("movie_details"):
                    details = result["movie_details"]
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader(f"You selected:")
                        st.markdown(f"### {details['title']}")
                        st.write(f"**Year:** {details['year']}")
                    with col2:
                        st.write(f"**Genres:** {', '.join(details['genres'])}")
                        st.write(f"**Director:** {details['director']}")
                        st.write(f"**Language:** {details['original_language']}")
                        st.write(f"**TomatoMeter:** {details['tomatoMeter']}%")
                
                st.markdown("### 📌 Recommendations:")
                for idx, rec in enumerate(result["recommendations"], 1):
                    st.markdown(f"**{idx}. {rec['title']}**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"- Genres: {', '.join(rec['genres'])}")
                        st.write(f"- Director: {rec['director']}")
                        st.write(f"- Year: {rec['year']}")
                    with col2:
                        st.write(f"- Language: {rec['original_language']}")
                        st.write(f"- Runtime: {rec['runtime_minutes']} minutes")
                        st.write(f"- TomatoMeter: {rec['tomatoMeter']}%")
                    st.write("---")
    
    # Genre Tab
    with tab2:
        st.subheader("Find movies by genre")
        genre_cols = [col for col in content_features.columns if col not in ['runtimeMinutes', 'director', 'originalLanguage']]
        selected_genre = st.selectbox("Select a genre", sorted(genre_cols))
        
        if st.button("Get Recommendations", key="genre_btn"):
            result = content_recommender(selected_genre, content_df, content_features, top_n=5, input_type='genre')
            
            if isinstance(result, str):
                st.error(result)
            else:
                st.markdown(f"### Top {selected_genre} Movies:")
                for idx, rec in enumerate(result["recommendations"], 1):
                    st.markdown(f"**{idx}. {rec['title']}**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"- Genres: {', '.join(rec['genres'])}")
                        st.write(f"- Director: {rec['director']}")
                        st.write(f"- Year: {rec['year']}")
                    with col2:
                        st.write(f"- Language: {rec['original_language']}")
                        st.write(f"- Runtime: {rec['runtime_minutes']} minutes")
                        st.write(f"- TomatoMeter: {rec['tomatoMeter']}%")
                    st.write("---")
                
                precision_val = evaluate_precision(result, 'genre', selected_genre)
                st.metric("Precision", f"{precision_val:.2f}")
    
    # Language Tab
    with tab3:
        st.subheader("Find movies by language")
        languages = sorted(set(language_decoder.values()))
        selected_language = st.selectbox("Select a language", languages)
        
        if st.button("Get Recommendations", key="language_btn"):
            result = content_recommender(selected_language, content_df, content_features, top_n=5, input_type='language')
            
            if isinstance(result, str):
                st.error(result)
            else:
                st.markdown(f"### Top Movies in {selected_language}:")
                for idx, rec in enumerate(result["recommendations"], 1):
                    st.markdown(f"**{idx}. {rec['title']}**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"- Genres: {', '.join(rec['genres'])}")
                        st.write(f"- Director: {rec['director']}")
                        st.write(f"- Year: {rec['year']}")
                    with col2:
                        st.write(f"- Language: {rec['original_language']}")
                        st.write(f"- Runtime: {rec['runtime_minutes']} minutes")
                        st.write(f"- TomatoMeter: {rec['tomatoMeter']}%")
                    st.write("---")
                
                precision_val = evaluate_precision(result, 'language', selected_language)
                st.metric("Precision", f"{precision_val:.2f}")
    
    # Year Tab
    with tab4:
        st.subheader("Find movies by release year")
        years = sorted(content_df['year'].dropna().unique())
        selected_year = st.selectbox("Select a year", years)
        
        if st.button("Get Recommendations", key="year_btn"):
            result = content_recommender(str(selected_year), content_df, content_features, top_n=5, input_type='year')
            
            if isinstance(result, str):
                st.error(result)
            else:
                st.markdown(f"### Top Movies from {selected_year}:")
                for idx, rec in enumerate(result["recommendations"], 1):
                    st.markdown(f"**{idx}. {rec['title']}**")
                    col1, col2 = st.columns([1, 1])
                    with col1:
                        st.write(f"- Genres: {', '.join(rec['genres'])}")
                        st.write(f"- Director: {rec['director']}")
                        st.write(f"- Year: {rec['year']}")
                    with col2:
                        st.write(f"- Language: {rec['original_language']}")
                        st.write(f"- Runtime: {rec['runtime_minutes']} minutes")
                        st.write(f"- TomatoMeter: {rec['tomatoMeter']}%")
                    st.write("---")
                
                precision_val = evaluate_precision(result, 'year', str(selected_year))
                st.metric("Precision", f"{precision_val:.2f}")

elif option == "Collaborative Filtering":
    st.header("👥 Collaborative Filtering")
    st.write("Recommend movies based on similar user preferences.")
    st.info("Collaborative Filtering coming soon!")

elif option == "Hybrid Filtering":
    st.header("🔀 Hybrid Filtering")
    st.write("Combine collaborative and content-based filtering.")
    st.info("Hybrid Filtering coming soon!")

# Footer
st.markdown("---")
st.markdown("💻 Movie Recommender System - Content-Based Filtering")
