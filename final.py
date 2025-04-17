# interface.py
import streamlit as st
from recommender_logic import load_data, load_decoders, content_recommender, evaluate_precision

st.title("🎬 Movie Recommendation System")

df, feature_df = load_data()
language_decoder, director_decoder = load_decoders()

option = st.selectbox("Choose your recommendation type", ['title', 'genre', 'language', 'year'])

user_input = st.text_input(f"Enter a movie {option}:")

top_n = st.slider("Number of recommendations", 1, 10, 5)

if st.button("Get Recommendations"):
    result = content_recommender(
        user_input=user_input,
        df=df,
        feature_df=feature_df,
        language_decoder=language_decoder,
        director_decoder=director_decoder,
        top_n=top_n,
        input_type=option
    )

    if isinstance(result, str):
        st.warning(result)
    else:
        if result.get('movie_details'):
            st.subheader("🎞️ Selected Movie")
            st.write(result['movie_details'])

        st.subheader("📽️ Recommendations")
        for i, rec in enumerate(result['recommendations'], 1):
            st.markdown(f"**{i}. {rec['title']}**")
            st.markdown(f"Genres: {', '.join(rec['genres'])}")
            st.markdown(f"Year: {rec['year']} | Runtime: {rec['runtime_minutes']} mins")
            st.markdown(f"Director: {rec['director']} | Language: {rec['original_language']}")
            st.markdown(f"TomatoMeter: {rec['tomatoMeter']}%")
            st.markdown("---")

        if option in ['genre', 'language', 'year']:
            precision = evaluate_precision(result, option, user_input)
            if precision is not None:
                st.success(f"Precision@{top_n}: {precision:.2f}")
