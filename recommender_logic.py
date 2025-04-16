import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import sklearn
import re
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

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
            print(f"   Director: {details['director']}")
            print(f"   Original Language: {details['original_language']}")
            print(f"   Runtime: {details['runtime_minutes']} minutes")
            print("----------------------------------------------------")

        print(f"\nRecommendations for '{result['movie_title']}':")
        for idx, recommendation in enumerate(result['recommendations'], 1):
            print(f"{idx}. Movie: {recommendation['title']}")
            print(f"   Genres: {', '.join(recommendation['genres'])}")
            print(f"   Year: {recommendation['year']}")
            print(f"   Director: {recommendation['director']}")
            print(f"   Language: {recommendation['original_language']}")
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

def menu(df, feature_df):
    print("Choose your input type:")
    print("1. Movie Title")
    print("2. Genre")
    print("3. Language")
    print("4. Release Year")

    choice = input("Enter your choice (1-4): ")

    input_type_map = {
        '1': 'title',
        '2': 'genre',
        '3': 'language',
        '4': 'year'
    }

    if choice not in input_type_map:
        print("Invalid choice.")
        return

    input_type = input_type_map[choice]

    if input_type == 'genre':
        available_genres = [col for col in feature_df.columns if col not in ['runtimeMinutes', 'director', 'originalLanguage']]
        print("\nAvailable genres:")
        for i, genre in enumerate(available_genres, 1):
            print(f"{i}. {genre}")
        genre_index = int(input("\nEnter genre number: ")) - 1
        if genre_index < 0 or genre_index >= len(available_genres):
            print("Invalid genre selection.")
            return
        user_input = available_genres[genre_index]

    elif input_type == 'language':
        languages = sorted(set(language_decoder.values()))
        print("\nAvailable languages:")
        for i, lang in enumerate(languages, 1):
            print(f"{i}. {lang}")
        lang_index = int(input("\nEnter language number: ")) - 1
        if lang_index < 0 or lang_index >= len(languages):
            print("Invalid language selection.")
            return
        user_input = languages[lang_index]

    elif input_type == 'year':
        user_input = input("Enter the release year (e.g., 2020): ")
        try:
            year = int(user_input)
        except ValueError:
            print("Invalid year format. Please enter a valid year (e.g., 2020).")
            return
        if df[df['year'] == year].empty:
            print(f"No movies found for the year '{year}'.")
            return

    else:
        user_input = input(f"Enter {input_type}: ")

    result = content_recommender(user_input, df, feature_df, top_n=5, input_type=input_type)
    show_recommendations(result, df)
