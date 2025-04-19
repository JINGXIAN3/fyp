# recommender_functions.py

from sklearn.metrics.pairwise import cosine_similarity

# Reversed decoders assumed to be loaded elsewhere
language_decoder = {}  # Load with pickle
director_decoder = {}  # Load with pickle

def sort_by_tomatoMeter(df, similarities, top_n=5):
    matched = [(idx, df.iloc[idx], similarities[idx]) for idx in similarities.argsort()[::-1]]
    matched.sort(key=lambda x: float(x[1].get('tomatoMeter', 0) or 0), reverse=True)
    return [idx for idx, _, _ in matched[:top_n]]

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
        similarities = cosine_similarity(target_vector, feature_df.values)[0]

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

    elif input_type in ['genre', 'language', 'year']:
        if input_type == 'genre':
            if user_input not in feature_df.columns:
                return f"Genre '{user_input}' not found."
            matching_indices = feature_df[feature_df[user_input] == 1].index

        elif input_type == 'language':
            rev_decoder = {v.lower(): k for k, v in language_decoder.items()}
            lang_code = rev_decoder.get(user_input.lower())
            if lang_code is None:
                return f"Language '{user_input}' not found."
            matching_indices = df[df['originalLanguage'] == lang_code].index

        elif input_type == 'year':
            try:
                year = int(user_input)
                matching_indices = df[df['year'] == year].index
            except ValueError:
                return "Invalid year format."

        if matching_indices.empty:
            return f"No movies found for {input_type} '{user_input}'."

        selected_df = df.loc[matching_indices].reset_index(drop=True)
        selected_features = feature_df.loc[matching_indices].reset_index(drop=True)
        target_vector = selected_features.mean().values.reshape(1, -1)
        similarities = cosine_similarity(target_vector, selected_features.values)[0]
        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)

    else:
        return f"Unsupported input type: {input_type}"

    for idx in top_indices:
        movie_data = df.iloc[idx] if input_type == 'title' else selected_df.iloc[idx]
        similarity_score = similarities[idx]
        genres = [col for col in feature_df.columns if col not in ['runtimeMinutes', 'director', 'originalLanguage'] and movie_data.get(col, 0) == 1]
        director = director_decoder.get(movie_data['director'], 'Unknown')
        language = language_decoder.get(movie_data['originalLanguage'], 'Unknown')

        explanation['recommendations'].append({
            'title': movie_data['title'],
            'similarity_score': similarity_score,
            'genres': genres,
            'year': movie_data.get('year', 'N/A'),
            'director': director,
            'original_language': language,
            'runtime_minutes': movie_data['runtimeMinutes'],
            'tomatoMeter': movie_data.get('tomatoMeter', 'N/A')
        })

    return explanation
