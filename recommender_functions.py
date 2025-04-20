# recommender_functions.py
import numpy as np
import pandas as pd
import re
import nltk
from nltk.tokenize import word_tokenize
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# --- Load Datasets ---
content_df = pd.read_csv("content_df.csv")
top_movies_collab_df = pd.read_csv("top_movies_collab_df.csv")
nlp_df = pd.read_csv("nlp_df.csv")

# Reversed decoders assumed to be loaded elsewhere
language_decoder = {}  # Load with pickle
director_decoder = {}  # Load with pickle

#---------------------------------Content Based Filtering--------------------------------------
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
# ------------------------------------------------------------------------------------------


# --------------------------------Collaborative Filtering-----------------------------------
def collaborative_recommender(user_id, ratings_df, top_n=10):
    explanation = {
        "user_id": user_id,
        "most_similar_users": [],
        "recommendations": []
    }

    try:
        user_id = int(user_id)
    except ValueError:
        return f"Invalid input. Please enter a valid numeric user ID."

    # Step 1: Create user-item matrix
    rating_matrix = ratings_df.pivot(index='userName', columns='title', values='standardized_score')
    rating_matrix = rating_matrix.fillna(0)

    if user_id not in rating_matrix.index:
        return f"User '{user_id}' not found in the dataset."

    # Step 2: Apply SVD
    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_svd = svd.fit_transform(rating_matrix)

    # Step 3: Compute cosine similarity
    similarity_matrix = cosine_similarity(matrix_svd)
    user_idx = rating_matrix.index.get_loc(user_id)
    user_similarity = similarity_matrix[user_idx]

    # Step 4: Find similar users
    similar_users = [(int(rating_matrix.index[i]), float(sim)) for i, sim in enumerate(user_similarity) if i != user_idx]
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)[:3]
    explanation["most_similar_users"] = [{"user_id": uid, "similarity_score": score} for uid, score in similar_users]

    # Step 5: Recommend movies
    user_seen_movies = set(rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index)
    recommendations = {}

    for idx in np.argsort(user_similarity)[::-1]:
        similar_user_ratings = rating_matrix.iloc[idx]
        for movie, rating in similar_user_ratings.items():
            if rating > 0 and movie not in user_seen_movies:
                recommendations[movie] = recommendations.get(movie, 0) + rating

    if recommendations:
        max_score = max(recommendations.values())
        normalized = {movie: (score / max_score) * 100 for movie, score in recommendations.items()}
        sorted_recommendations = sorted(normalized.items(), key=lambda x: x[1], reverse=True)
        explanation["recommendations"] = [{"title": title, "score": score} for title, score in sorted_recommendations[:top_n]]

    return explanation
    
# ------------------------------------------------------------------------------------------

# ------------------------------NLP-Based Filtering------------------------------------------
def get_average_vector(text, model, vector_size=300):
    tokens = word_tokenize(text.lower())
    vectors = [model[word] for word in tokens if word in model]
    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)

def nlp_recommender(user_query, nlp_df, language_decoder, director_decoder, w2v_model, top_n=10):
    min_year, max_year = None, None
    query_language = None
    query_genre = None
    director_name = None
    min_score = None
    min_runtime, max_runtime = None, None

    genre_columns = [
        'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime', 'Documentary',
        'Drama', 'Entertainment', 'Faith & spirituality', 'Fantasy', 'Health & wellness',
        'History', 'Horror', 'Kids & family', 'LGBTQ+', 'Music', 'Nature', 'Other',
        'Reality', 'Romance', 'Sci-fi', 'Sports', 'Variety Show', 'War', 'Western'
    ]

    # Year filter
    year_range = re.findall(r'(\d{4})', user_query)
    if len(year_range) >= 2:
        min_year, max_year = int(year_range[0]), int(year_range[1])
    elif len(year_range) == 1:
        min_year = max_year = int(year_range[0])

    # Language
    all_languages = [v.lower() for v in language_decoder.values()]
    for lang in all_languages:
        if lang in user_query.lower():
            query_language = lang
            break

    # Genre
    for genre in genre_columns:
        if genre.lower() in user_query.lower():
            query_genre = genre
            break

    # TomatoMeter
    score_match = re.search(r'tomato.*?(\d+)', user_query.lower())
    if score_match:
        min_score = int(score_match.group(1))

    # Director
    director_match = re.search(r'director(?: is| named)? ([A-Za-z ]+)', user_query.lower())
    if director_match:
        director_name = director_match.group(1).strip().lower()

    # Runtime
    runtime_range = re.findall(r'(\d{2,3})\s*minutes?', user_query.lower())
    if any(term in user_query.lower() for term in ['under', 'less than']) and runtime_range:
        max_runtime = int(runtime_range[0])
    elif any(term in user_query.lower() for term in ['over', 'more than']) and runtime_range:
        min_runtime = int(runtime_range[0])

    # Clean query
    clean_query = user_query
    for lang in all_languages:
        clean_query = clean_query.replace(lang, '')
    if query_genre:
        clean_query = clean_query.replace(query_genre, '')
    clean_query = re.sub(r'\d{4}', '', clean_query)
    clean_query = re.sub(r'director.*?(is|named)? ?[A-Za-z ]+', '', clean_query, flags=re.IGNORECASE)
    clean_query = re.sub(r'tomato.*?\d+', '', clean_query, flags=re.IGNORECASE)
    clean_query = re.sub(r'\d{2,3}\s*minutes?', '', clean_query)
    clean_query = re.sub(r'(under|over|between|around)', '', clean_query, flags=re.IGNORECASE)

    # TF-IDF
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(nlp_df['movie_info'].fillna(''))
    query_tfidf = tfidf_vectorizer.transform([clean_query])
    tfidf_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

    # Word2Vec
    nlp_df['w2v_vector'] = nlp_df['movie_info'].apply(lambda x: get_average_vector(x, w2v_model))
    query_w2v_vector = get_average_vector(user_query, w2v_model)
    w2v_similarities = nlp_df['w2v_vector'].apply(
        lambda x: cosine_similarity([x], [query_w2v_vector])[0][0]
    )

    # Combine
    combined_similarity = 0.5 * tfidf_similarities + 0.5 * w2v_similarities
    nlp_df['similarity'] = combined_similarity
    filtered_df = nlp_df.copy()

    # Apply filters
    if min_year is not None:
        filtered_df = filtered_df[filtered_df['year'].between(min_year, max_year if max_year else min_year)]
    if query_language:
        matching_lang_ids = [k for k, v in language_decoder.items() if v.lower() == query_language]
        filtered_df = filtered_df[filtered_df['originalLanguage'].isin(matching_lang_ids)]
    if query_genre:
        filtered_df = filtered_df[filtered_df[query_genre] == 1]
    if director_name:
        matching_director_ids = [k for k, v in director_decoder.items() if director_name in v.lower()]
        if matching_director_ids:
            filtered_df = filtered_df[filtered_df['director'].isin(matching_director_ids)]
    if min_score is not None:
        filtered_df = filtered_df[filtered_df['tomatoMeter'] >= min_score]
    if min_runtime is not None:
        filtered_df = filtered_df[filtered_df['runtimeMinutes'] >= min_runtime]
    if max_runtime is not None:
        filtered_df = filtered_df[filtered_df['runtimeMinutes'] <= max_runtime]

    top_recommendations = filtered_df.sort_values(by='similarity', ascending=False).head(top_n)
    return top_recommendations
# ------------------------------------------------------------------------------------------

# ------------------------------Hybrid Filtering--------------------------------------------
def get_next_user_id(rating_matrix):
    # Assumes index is user ID or can be converted to int
    existing_ids = rating_matrix.index.astype(int)
    return str(existing_ids.max() + 1)


def update_rating_matrix(rating_matrix, user_id, liked_movies):
    # liked_movies = list of tuples: (movie_title, score)
    for title, rating in liked_movies:
        if title in rating_matrix.columns:
            rating_matrix.loc[user_id, title] = rating
        else:
            print(f"Warning: {title} not found in rating matrix.")
    return rating_matrix
    
def hybrid_recommender(user_id, movie_title, content_df, content_features, top_movies_collab_df,
                       content_weight=0.5, collab_weight=0.5, top_n=10):
    total_weight = content_weight + collab_weight
    content_weight /= total_weight
    collab_weight /= total_weight

    content_result = content_recommender(movie_title, content_df, content_features, top_n=top_n*2, input_type='title')

    content_recommendations = {}
    if not isinstance(content_result, str):
        content_recommendations = {
            rec['title']: rec['similarity_score'] 
            for rec in content_result['recommendations']
        }
        if content_recommendations:
            max_score = max(content_recommendations.values())
            content_recommendations = {m: (s / max_score) * 100 for m, s in content_recommendations.items()}

    collab_recommendations = {}
    rating_matrix = top_movies_collab_df.pivot(index='userName', columns='title', values='standardized_score').fillna(0)

    svd = TruncatedSVD(n_components=20, random_state=42)
    matrix_svd = svd.fit_transform(rating_matrix)
    similarity_matrix = cosine_similarity(matrix_svd)

    if user_id in rating_matrix.index:
        user_idx = rating_matrix.index.get_loc(user_id)
        user_similarity = similarity_matrix[user_idx]
        similar_users_idx = np.argsort(user_similarity)[::-1]
        user_seen = set(rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index)

        for idx in similar_users_idx:
            if idx == user_idx:
                continue
            sim_user_ratings = rating_matrix.iloc[idx]
            for movie, rating in sim_user_ratings.items():
                if rating > 0 and movie not in user_seen:
                    collab_recommendations[movie] = collab_recommendations.get(movie, 0) + rating

        if collab_recommendations:
            max_score = max(collab_recommendations.values())
            collab_recommendations = {m: (s / max_score) * 100 for m, s in collab_recommendations.items()}

    hybrid_scores = {}
    for movie, score in content_recommendations.items():
        hybrid_scores[movie] = content_weight * score
    for movie, score in collab_recommendations.items():
        hybrid_scores[movie] = hybrid_scores.get(movie, 0) + collab_weight * score

    return sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
