import streamlit as st
import pandas as pd


content_df = pd.read_csv("content_df.csv")
top_movies_collab_df = pd.read_csv("top_movies_collab_df.csv")

feature_cols = [
    'runtimeMinutes', 'director', 'originalLanguage',
    'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime',
    'Documentary', 'Drama', 'Entertainment', 'Faith & spirituality',
    'Fantasy', 'Health & wellness', 'History', 'Horror', 'Kids & family',
    'LGBTQ+', 'Music', 'Nature','Other', 'Reality', 'Romance', 'Sci-fi', 'Sports',
    'Variety Show', 'War', 'Western'
]

content_features = content_df[feature_cols]
# Title
st.title("🎬 Movie Recommender System")

# Sidebar
option = st.sidebar.radio(
    "Choose Recommendation Method:",
    (
        "Collaborative Filtering",
        "Content-Based Filtering (Category Selection)",
        "Content-Based Filtering (Textbox Input)",
        "Hybrid Filtering"
    )
)

# Helper: convert decoder keys
def get_language_name(code):
    return language_decoder.get(code, "Unknown")

def get_director_name(code):
    return director_decoder.get(code, "Unknown")

# Content-Based: Category Selection
if option == "Content-Based Filtering (Category Selection)":
    st.subheader("🎯 Content-Based Filtering (Category Selection)")

    st.write("Choose a filter type for recommendations:")

    filter_type = st.selectbox("Select Filter Type", ["Genre", "Language", "Release Year", "Movie Title"])

    user_input = None

    if filter_type == "Genre":
        genre_cols = [col for col in content_features.columns if col not in ['runtimeMinutes', 'director', 'originalLanguage']]
        genre = st.selectbox("Select Genre", sorted(genre_cols))
        user_input = genre
        input_type = 'genre'

    elif filter_type == "Language":
        languages = sorted(set(language_decoder.values()))
        language = st.selectbox("Select Language", languages)
        user_input = language
        input_type = 'language'

    elif filter_type == "Release Year":
        years = sorted(content_df['year'].dropna().unique())
        year = st.selectbox("Select Year", years)
        user_input = str(year)
        input_type = 'year'

    elif filter_type == "Movie Title":
        movie_titles = sorted(content_df['title'].dropna().unique())
        selected_title = st.text_input("Enter Movie Title (exact match preferred)")
        if selected_title.strip():
            user_input = selected_title.strip()
            input_type = 'title'

    if user_input and st.button("Get Recommendations"):
    result = content_recommender(
        user_input=user_input,
        df=content_df,
        feature_df=content_features,
        top_n=5,
        input_type=input_type
    )

    if isinstance(result, str):
        st.error(result)
    else:
        if result.get("movie_details"):
            details = result["movie_details"]
            st.markdown(f"### 🎞️ Selected Movie: {details['title']}")
            st.write(f"**Genres:** {', '.join(details['genres'])}")
            st.write(f"**Director:** {details['director']}")
            st.write(f"**Language:** {details['original_language']}")
            st.write(f"**Year:** {details['year']}")
            st.write(f"**TomatoMeter:** {details['tomatoMeter']}%")

        st.markdown("### 📌 Top Recommendations:")
        for idx, rec in enumerate(result["recommendations"], 1):
            st.markdown(f"**{idx}. {rec['title']}**")
            st.write(f"- Genres: {', '.join(rec['genres'])}")
            st.write(f"- Director: {rec['director']}")
            st.write(f"- Language: {rec['original_language']}")
            st.write(f"- Year: {rec['year']}")
            st.write(f"- Runtime: {rec['runtime_minutes']} minutes")
            st.write(f"- TomatoMeter: {rec['tomatoMeter']}%")

        # Evaluation Metric
        if input_type in ['genre', 'language', 'year']:
            precision_val = evaluate_precision(result, input_type, user_input)
            st.markdown("---")
            st.write(f"🎯 **Precision@{len(result['recommendations'])}**: {precision_val:.2f}")


# Placeholder sections for other options
elif option == "Collaborative Filtering":
    st.subheader("👥 Collaborative Filtering")
    st.write("Recommend movies based on user-user or item-item similarities.")
    st.info("Collaborative Filtering logic coming soon!")

elif option == "Content-Based Filtering (Textbox Input)":
    st.subheader("📝 Content-Based Filtering (Textbox Input)")
    user_text = st.text_area("Enter a movie description:")
    if user_text and st.button("Get Recommendations"):
        st.warning("NLP-based content filtering not yet implemented!")

elif option == "Hybrid Filtering":
    st.subheader("🔀 Hybrid Filtering")
    st.write("Combine collaborative and content-based filtering.")
    st.info("Hybrid Filtering logic coming soon!")
