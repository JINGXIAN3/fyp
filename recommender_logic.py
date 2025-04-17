# recommender_logic.py
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity

def load_data():
    content_df = pd.read_csv("content_df.csv")
    encoded_df = pd.read_csv("encoded_df.csv")
    return encoded_df, content_df

def load_decoders():
    with open("language_decoder.pkl", "rb") as f:
        language_decoder = pickle.load(f)
    with open("director_decoder.pkl", "rb") as f:
        director_decoder = pickle.load(f)
    return language_decoder, director_decoder

def sort_by_tomatoMeter(df, similarities, top_n=5):
    matched = [(idx, df.iloc[idx], similarities[idx]) for idx in similarities.argsort()[::-1]]
    matched.sort(key=lambda x: float(x[1].get('tomatoMeter', 0) or 0), reverse=True)
    return [idx for idx, _, _ in matched[:top_n]]

def content_recommender(user_input, df, feature_df, language_decoder, director_decoder, top_n=5, input_type='title'):
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
            movie_idx = df[df['title'] == user_input].index[0]
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
                similarities[idx] = -1
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

        matching_indices = feature_df[feature_df[user_input] == 1].index
        if len(matching_indices) == 0:
            return f"No movies found with genre '{user_input}'."

        selected_df = df.loc[matching_indices].reset_index(drop=True)
        selected_features = feature_df.loc[matching_indices].reset_index(drop=True)

        target_vector = selected_features.mean().values.reshape(1, -1)
        similarities = cosine_similarity(target_vector, selected_features.values)[0]
        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)

        df = selected_df
        feature_df = selected_features

    elif input_type == 'language':
        rev_decoder = {v.lower(): k for k, v in language_decoder.items()}
        lang_code = rev_decoder.get(user_input.lower())
        if lang_code is None:
            return f"Language '{user_input}' not found."

        matching_indices = df[df['originalLanguage'] == lang_code].index
        if matching_indices.empty:
            return f"No movies found for language '{user_input}'."

        selected_df = df.loc[matching_indices]
        selected_features = feature_df.loc[matching_indices]
        target_vector = selected_features.mean().values.reshape(1, -1)
        similarities = cosine_similarity(target_vector, selected_features.values)[0]
        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)

    elif input_type == 'year':
        try:
            year = int(user_input)
        except ValueError:
            return "Invalid year format."
        matching_indices = df[df['year'] == year].index
        if matching_indices.empty:
            return f"No movies found for year '{user_input}'."

        selected_df = df.loc[matching_indices]
        selected_features = feature_df.loc[matching_indices]
        target_vector = selected_features.mean().values.reshape(1, -1)
        similarities = cosine_similarity(target_vector, selected_features.values)[0]
        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)

    else:
        return f"Unsupported input type: {input_type}"

    for idx in top_indices:
        movie_data = df.iloc[idx]
        genres = [col for col in feature_df.columns if col not in ['runtimeMinutes', 'director', 'originalLanguage'] and movie_data.get(col, 0) == 1]
        recommendation = {
            'title': movie_data['title'],
            'similarity_score': similarities[idx],
            'genres': genres,
            'year': movie_data.get('year', 'N/A'),
            'director': director_decoder.get(movie_data['director'], 'Unknown'),
            'original_language': language_decoder.get(movie_data['originalLanguage'], 'Unknown'),
            'runtime_minutes': movie_data['runtimeMinutes'],
            'tomatoMeter': movie_data.get('tomatoMeter', 'N/A')
        }
        explanation['recommendations'].append(recommendation)

    return explanation

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
