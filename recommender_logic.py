import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

# Load the language decoder
with open("language_decoder.pkl", "wb") as f:
    pickle.dump(language_decoder, f)

with open("director_decoder.pkl", "wb") as f:
    pickle.dump(director_decoder, f)

def sort_by_tomatoMeter(df, similarities, top_n=5):
    matched = [(idx, df.iloc[idx], similarities[idx]) for idx in similarities.argsort()[::-1]]
    matched.sort(key=lambda x: float(x[1].get('tomatoMeter', 0) or 0), reverse=True)
    return [idx for idx, _, _ in matched[:top_n]]

# Recommender function
def content_recommender(user_input, df, feature_df, top_n=5, input_type='title'):
    df = df.reset_index(drop=True)
    feature_df = feature_df.reset_index(drop=True)

    explanation = {
        'movie_title': user_input,
        'movie_details': None,
        'recommendations': [],
        'input_type': input_type,
        'user_input': user_input
    }

    if input_type == 'title':
        try:
            movie_idx = df[df['title'].str.lower() == user_input.lower()].index[0]
        except IndexError:
            return f"Movie '{user_input}' not found."

        selected_movie = df.iloc[movie_idx]
        target_vector = feature_df.iloc[movie_idx].values.reshape(1, -1)
        all_vectors = feature_df.values
        similarities = cosine_similarity(target_vector, all_vectors)[0]

        selected_director = selected_movie['director']
        selected_language = selected_movie['originalLanguage']
        selected_genres = [col for col in feature_df.columns if selected_movie.get(col, 0) == 1]

        for idx in range(len(similarities)):
            movie = df.iloc[idx]
            if idx == movie_idx:
                similarities[idx] = -1  # Exclude the same movie
                continue
            if movie['director'] == selected_director:
                similarities[idx] += 0.1
            movie_genres = [col for col in feature_df.columns if movie.get(col, 0) == 1]
            genre_overlap = len(set(selected_genres).intersection(set(movie_genres)))
            similarities[idx] += 0.05 * genre_overlap
            if movie['originalLanguage'] == selected_language:
                similarities[idx] += 0.03

        top_indices = similarities.argsort()[::-1][:top_n]

        explanation['movie_details'] = {
            'title': selected_movie['title'],
            'genres': selected_genres,
            'director': director_decoder.get(selected_director, 'Unknown'),
            'original_language': language_decoder.get(selected_language, 'Unknown'),
            'runtime_minutes': selected_movie['runtimeMinutes'],
            'tomatoMeter': selected_movie.get('tomatoMeter', 'N/A'),
            'year': selected_movie.get('year', 'N/A')
        }

        explanation['reasoning'] = (
            f"Since you liked '{selected_movie['title']}', we're recommending movies that share similar "
            f"genres ({', '.join(selected_genres)}), the same director "
            f"({director_decoder.get(selected_director, 'Unknown')}), or are in the same language "
            f"({language_decoder.get(selected_language, 'Unknown')})."
        )

    elif input_type == 'genre':
        if user_input not in feature_df.columns:
            return f"Genre '{user_input}' not found."
        
        # Filter to include only movies with this genre
        matching_indices = feature_df[feature_df[user_input] == 1].index
        if len(matching_indices) == 0:
            return f"No movies found with genre '{user_input}'."
        
        # Use the filtered dataframes
        selected_df = df.loc[matching_indices].reset_index(drop=True)
        selected_features = feature_df.loc[matching_indices].reset_index(drop=True)
        
        target_vector = selected_features.mean().values.reshape(1, -1)
        all_vectors = selected_features.values
        similarities = cosine_similarity(target_vector, all_vectors)[0]
        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)
        
        df = selected_df
        feature_df = selected_features

    elif input_type == 'language':
        rev_language_decoder = {v.lower(): k for k, v in language_decoder.items()}
        lang_code = rev_language_decoder.get(user_input.lower())
        if lang_code is None:
            return f"Language '{user_input}' not found."

        matching_indices = df[df['originalLanguage'] == lang_code].index
        if matching_indices.empty:
            return f"No movies found for language '{user_input}'."

        target_vector = feature_df.loc[matching_indices].mean().values.reshape(1, -1)
        selected_df = df.loc[matching_indices]
        selected_features = feature_df.loc[matching_indices]
        all_vectors = selected_features.values
        similarities = cosine_similarity(target_vector, all_vectors)[0]
        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)

    elif input_type == 'year':
        try:
            year = int(user_input)
        except ValueError:
            return "Invalid year format. Please enter a number."
        matching_indices = df[df['year'] == year].index
        if matching_indices.empty:
            return f"No movies found for year '{user_input}'."

        target_vector = feature_df.loc[matching_indices].mean().values.reshape(1, -1)
        selected_df = df.loc[matching_indices]
        selected_features = feature_df.loc[matching_indices]
        all_vectors = selected_features.values
        similarities = cosine_similarity(target_vector, all_vectors)[0]
        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)

    else:
        return f"Unsupported input type: {input_type}"

    for idx in top_indices:
        movie_data = df.iloc[idx] if input_type == 'title' else (selected_df.iloc[idx] if input_type in ['language', 'year'] else df.iloc[idx])
        similarity_score = similarities[idx]
        genres = [
            col for col in feature_df.columns
            if col not in ['runtimeMinutes', 'director', 'originalLanguage'] and movie_data.get(col, 0) == 1
        ]
        director = director_decoder.get(movie_data['director'], 'Unknown')
        language = language_decoder.get(movie_data['originalLanguage'], 'Unknown')

        movie_features = {
            'title': movie_data['title'],
            'similarity_score': similarity_score,
            'genres': genres,
            'year': movie_data.get('year', 'N/A'),
            'director': director,
            'original_language': language,
            'runtime_minutes': movie_data['runtimeMinutes'],
            'tomatoMeter': movie_data.get('tomatoMeter', 'N/A')
        }

        explanation['recommendations'].append(movie_features)

    return explanation

# Evaluation function for Precision
def evaluate_precision(result, input_type, user_input):
    if isinstance(result, str):
        return None
    recommendations = result['recommendations']
    top_n = len(recommendations)
    if input_type == 'genre':
        return sum([1 if user_input in rec['genres'] else 0 for rec in recommendations]) / top_n
    elif input_type == 'language':
        return sum([1 if rec['original_language'].lower() == user_input.lower() else 0 for rec in recommendations]) / top_n
    elif input_type == 'year':
        try:
            year = int(user_input)
            return sum([1 if rec['year'] == year else 0 for rec in recommendations]) / top_n
        except ValueError:
            return None
    return None

def show_recommendations(result, df):
    if isinstance(result, str):
        print(result)
    else:
        if result['movie_details']:
            details = result['movie_details']
            print(f"\nSelected Movie: {details['title']}")
            print(f"   Genres: {', '.join(details['genres'])}")
            print(f"   Year: {(details['year'])}")
            print(f"   Runtime: {details['runtime_minutes']} minutes")
            print("----------------------------------------------------")

        print(f"\nRecommendations for '{result['movie_title']}':")
        for idx, recommendation in enumerate(result['recommendations'], 1):
            print(f"{idx}. Movie: {recommendation['title']}")
            print(f"   Genres: {', '.join(recommendation['genres'])}")
            print(f"   Year: {recommendation['year']}")
            print(f"   Runtime: {recommendation['runtime_minutes']} minutes")
            print(f"   TomatoMeter: {recommendation['tomatoMeter']}%")
            print("---")

        input_type = result['input_type']
        user_input = result['user_input']

        if input_type in ['genre', 'language', 'year']:
            precision_val = evaluate_precision(result, input_type, user_input)

            print("\n--- Evaluation Metrics ---")
            if precision_val is not None:
                print(f"Precision@{len(result['recommendations'])}: {precision_val:.2f}")

