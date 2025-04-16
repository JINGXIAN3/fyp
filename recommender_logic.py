  {
"cell_type": "code",
"id": "9e999e95-6f67-4fbe-a9c7-b0b825321a78",
"metadata": {},
"outputs": [],
"source": [
 "encoded_df = cleaned_df.copy()\n",
 "encoded_df.head()"
]
  },
  {
"cell_type": "code",
"id": "89e64886-a812-4339-bb9f-89620c0c6caa",
"metadata": {},
"outputs": [],
"source": [
 "from sklearn.preprocessing import LabelEncoder\n",
 "\n",
 "# Initialize Label Encoder\n",
 "le = LabelEncoder()\n",
 "language_encoded = LabelEncoder()\n",
 "director_encoded = LabelEncoder()\n",
 "\n",
 "# Fit and transform the originalLanguage\n",
 "encoded_df['originalLanguage'] = language_encoded.fit_transform(cleaned_df['originalLanguage'])\n",
 "\n",
 "# Fit and transform the director\n",
 "encoded_df['director'] = director_encoded.fit_transform(cleaned_df['director'])\n",
 "\n",
 "# Create Decoders\n",
 "language_decoder = {i: lang for i, lang in enumerate(language_encoded.classes_)}\n",
 "director_decoder = {i: d for i, d in enumerate(director_encoded.classes_)}\n",
 "\n",
 "\n",
 "# Label Encoding for 'userName'\n",
 "encoded_df['userName'] = le.fit_transform(encoded_df['userName'])\n",
 "\n",
 "# Standardize values (strip spaces, convert to uppercase)\n",
 "encoded_df['scoreSentiment'] = encoded_df['scoreSentiment'].astype(str).str.strip().str.upper()\n",
 "encoded_df['reviewState'] = encoded_df['reviewState'].astype(str).str.strip().str.lower()\n",
 "\n",
 "# Define mappings\n",
 "sentiment_mapping = {'POSITIVE': 1, 'NEGATIVE': 0}\n",
 "review_state_mapping = {'fresh': 1, 'rotten': 0}\n",
 "\n",
 "# Apply the mappings with error handling\n",
 "encoded_df['scoreSentiment'] = encoded_df['scoreSentiment'].map(sentiment_mapping)\n",
 "encoded_df['reviewState'] = encoded_df['reviewState'].map(review_state_mapping)\n"
]
  },
  {
"cell_type": "code",
"id": "f9e26922-855b-41a3-8161-f723eb2377a8",
"metadata": {},
"outputs": [],
"source": [
 "encoded_df.head()"
]
  },
  {
"cell_type": "code",
"id": "e28b92ee-5230-4cd5-886d-a28b2b250d4a",
"metadata": {},
"outputs": [],
"source": [
 "final_encoded_df_cleaned = encoded_df"
]
  },
  {
"cell_type": "markdown",
"id": "f076f43a-72b9-462e-ac44-e27df012751e",
"metadata": {},
"source": [
 "## Remove the data that not use \n"
]
  },
  {
"cell_type": "code",

"id": "8e1719bf-73a4-467e-a7cf-2b5bd77feac3",
"metadata": {},
"outputs": [],
"source": [
 "columns_remove = ['id', 'genre', 'originalScore','original_score_flag','genre_list'] \n",
 "final_encoded_df_cleaned = final_encoded_df_cleaned.drop(columns=columns_remove)\n",
 "final_encoded_df_cleaned['year'] = final_encoded_df_cleaned['year'].astype('Int64')"
]
  },
  {
"cell_type": "code",

"id": "b7e1b8c0-7488-4005-9c05-688a335d1f8d",
"metadata": {},
"outputs": [],
"source": [
 "final_encoded_df_cleaned.head()"
]
  },
  {
"cell_type": "code",

"id": "76f1bc02-5542-40de-aa14-44c364f0b674",
"metadata": {},
"outputs": [],
"source": [
 "final_encoded_df_cleaned.info()"
]
  },
  {
"cell_type": "markdown",
"id": "55366ee6-a50f-4101-b0c0-4ccb2c83ba68",
"metadata": {},
"source": [
 "# DATA SPLITING"
]
  },
  {
"cell_type": "code",

"id": "c895b0f5-7676-4b1f-a0e7-c569daae4ddd",
"metadata": {},
"outputs": [],
"source": [
 "df = final_encoded_df_cleaned.copy()"
]
  },
  {
"cell_type": "markdown",
"id": "a9b0c01d-5a39-4ead-973e-9569978596e0",
"metadata": {},
"source": [
 "## A) Content-Based Filtering"
]
  },
  {
"cell_type": "code",

"id": "6773bf04-a030-4847-b287-cd3da376120b",
"metadata": {},
"outputs": [],
"source": [
 "content_df = df.copy(deep=True)"
]
  },
  {
"cell_type": "code",

"id": "1b2285b7-7d0d-45b3-ad48-ab20d9ef1ee6",
"metadata": {},
"outputs": [],
"source": [
 "columns_remove = ['audienceScore','userName', 'reviewId', 'reviewText','scoreSentiment', 'reviewState','review_text_flag', 'year_flag',\n",
 " 'standardized_score'] \n",
 "content_df = content_df.drop(columns=columns_remove)"
]
  },
  {
"cell_type": "code",

"id": "cc0c026b-7e13-450d-8ff5-95ff58518abf",
"metadata": {},
"outputs": [],
"source": [
 "content_df.head()"
]
  },
  {
"cell_type": "code",

"id": "5013b3b2-c7bd-4b54-9781-dae732ce604b",
"metadata": {},
"outputs": [],
"source": [
 "# Remove rows with duplicate titles\n",
 "content_df = content_df.drop_duplicates(subset='title')\n",
 "\n",
 "# Check the updated content_df after removing duplicate titles\n",
 "content_df.head()"
]
  },
  {
"cell_type": "code",

"id": "6f6a0bf9-0287-42ec-9726-0d01580e9e46",
"metadata": {},
"outputs": [],
"source": [
 "content_df.info()"
]
  },
  {
"cell_type": "code",

"id": "1bf7ada8-dbfc-4dd1-a614-ce5a8c8f6556",
"metadata": {},
"outputs": [],
"source": [
 "content_df.shape"
]
  },
  {
"cell_type": "markdown",
"id": "8eefec45-7a1e-4288-844e-6538556a961b",
"metadata": {},
"source": [
 "## B) Collaborative Filtering"
]
  },
  {
"cell_type": "code",

"id": "ae1d6bcf-7109-4d87-8c13-28a56a54fe5d",
"metadata": {},
"outputs": [],
"source": [
 "collaborative_df = df.copy(deep=True)"
]
  },
  {
"cell_type": "code",

"id": "83c47368-f18e-4a5b-8cff-6e40c80fd05a",
"metadata": {},
"outputs": [],
"source": [
 "collaborative_df.head()"
]
  },
  {
"cell_type": "code",

"id": "34ffbd1d-4e8a-46be-8b35-ee2f00b80959",
"metadata": {},
"outputs": [],
"source": [
 "genre_columns = [\n",
 " 'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime',\n",
 " 'Documentary', 'Drama', 'Entertainment', 'Faith & spirituality',\n",
 " 'Fantasy', 'Health & wellness', 'History', 'Horror', 'Kids & family',\n",
 " 'LGBTQ+', 'Music', 'Nature', 'Other', 'Reality', 'Romance', 'Sci-fi', 'Sports',\n",
 " 'Variety Show', 'War', 'Western'\n",
 "]\n",
 "\n",
 "columns_remove = ['movie_info', 'year', 'director','originalLanguage', 'runtimeMinutes','reviewId','reviewText',\n",
 "'scoreSentiment','reviewState','review_text_flag', 'year_flag', 'movie_info_flag'] + genre_columns \n",
 "collaborative_df = collaborative_df.drop(columns=columns_remove)"
]
  },
  {
"cell_type": "code",

"id": "53b50429-b103-4f93-9bcc-b8ca8f6a21e4",
"metadata": {},
"outputs": [],
"source": [
 "# Remove duplicates\n",
 "collaborative_df = collaborative_df.drop_duplicates(subset=['userName', 'title'])\n"
]
  },
  {
"cell_type": "code",

"id": "54d3e09a-0b88-42a5-ab57-29c26e223e46",
"metadata": {},
"outputs": [],
"source": [
 "collaborative_df.shape"
]
  },
  {
"cell_type": "markdown",
"id": "1eb73f47-6414-42cc-9a9b-a048585b6b9c",
"metadata": {},
"source": [
 "## C) NLP-Based Filtering"
]
  },
  {
"cell_type": "code",

"id": "4f05b23d-a785-4d85-9c25-39aab7c11490",
"metadata": {},
"outputs": [],
"source": [
 "nlp_df = content_df.copy(deep=True)"
]
  },
  {
"cell_type": "code",

"id": "e0219947-2f55-44e6-b77d-fd3beed03177",
"metadata": {},
"outputs": [],
"source": [
 "nlp_df"
]
  },
  {
"cell_type": "markdown",
"id": "ac1d32e5-73a4-472e-b46d-59361fba4347",
"metadata": {},
"source": [
 "## Summary"
]
  },
  {
"cell_type": "code",

"id": "dc3c7679-eb6b-4dbd-ab48-29975a6017de",
"metadata": {},
"outputs": [],
"source": [
 "# Display the shapes of the training and testing sets\n",
 "print(f\"Content-Based Dataset: {content_df.shape}\")\n",
 "print(f\"Collaborative Dataset: {collaborative_df.shape}\")"
]
  },
  {
"cell_type": "markdown",
"id": "a635124c-a533-4502-b76b-5c26a8c4b01a",
"metadata": {},
"source": [
 "# MODELLING"
]
  },
  {
"cell_type": "markdown",
"id": "721d0f6e-31dc-4b37-8b1e-dd7d3450d6ee",
"metadata": {},
"source": [
 "Content-Based Filtering uses the movie features (movie title, genres) to recommend movies that are similar to what the user has already watched or rated highly.\n"
]
  },
  {
"cell_type": "markdown",
"id": "e2b39b6f-cafc-40ac-94e4-dbbf311dfe08",
"metadata": {},
"source": [
 "## A) Content-Based Filtering"
]
  },
  {
"cell_type": "code",

"id": "aedcd193-fc71-4a3b-91a6-3570c4aeff30",
"metadata": {},
"outputs": [],
"source": [
 "content_df"
]
  },
  {
"cell_type": "code",

"id": "df9c2181-d82e-43b8-b1d5-9463d6aeca73",
"metadata": {},
"outputs": [],
"source": [
 "content_df.info()"
]
  },
  {
"cell_type": "code",

"id": "b3fe078d-87c6-42fc-bd2c-27560ef8d090",
"metadata": {},
"outputs": [],
"source": [
 "# Save content-based dataframe\n",
 "content_df.to_csv(\"content_df.csv\", index=False)"
]
  },
  {
"cell_type": "code",

"id": "4c6c9a64-ff74-4f4b-a1cd-2ea5a4da9df2",
"metadata": {},
"outputs": [],
"source": [
 "feature_cols = [\n",
 " 'runtimeMinutes', 'director', 'originalLanguage',\n",
 " 'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime',\n",
 " 'Documentary', 'Drama', 'Entertainment', 'Faith & spirituality',\n",
 " 'Fantasy', 'Health & wellness', 'History', 'Horror', 'Kids & family',\n",
 " 'LGBTQ+', 'Music', 'Nature','Other', 'Reality', 'Romance', 'Sci-fi', 'Sports',\n",
 " 'Variety Show', 'War', 'Western'\n",
 "]\n",
 "\n",
 "content_features = content_df[feature_cols]\n"
]
  },
  {
"cell_type": "code",

"id": "2f01d0fd-acf3-45b8-bf13-973d03db1b55",
"metadata": {},
"outputs": [],
"source": [
 "from sklearn.metrics.pairwise import cosine_similarity\n",
 "\n",
 "def sort_by_tomatoMeter(df, similarities, top_n=5):\n",
 " matched = [(idx, df.iloc[idx], similarities[idx]) for idx in similarities.argsort()[::-1]]\n",
 " matched.sort(key=lambda x: float(x[1].get('tomatoMeter', 0) or 0), reverse=True)\n",
 " return [idx for idx, _, _ in matched[:top_n]]\n",
 "\n",
 "# Recommender function\n",
 "def content_recommender(user_input, df, feature_df, top_n=5, input_type='title'):\n",
 " df = df.reset_index(drop=True)\n",
 " feature_df = feature_df.reset_index(drop=True)\n",
 "\n",
 " explanation = {\n",
 "  'movie_title': user_input,\n",
 "  'movie_details': None,\n",
 "  'recommendations': [],\n",
 "  'input_type': input_type,\n",
 "  'user_input': user_input\n",
 " }\n",
 "\n",
 " if input_type == 'title':\n",
 "  try:\n",
 "movie_idx = df[df['title'].str.lower() == user_input.lower()].index[0]\n",
 "  except IndexError:\n",
 "return f\"Movie '{user_input}' not found.\"\n",
 "\n",
 "  selected_movie = df.iloc[movie_idx]\n",
 "  target_vector = feature_df.iloc[movie_idx].values.reshape(1, -1)\n",
 "  all_vectors = feature_df.values\n",
 "  similarities = cosine_similarity(target_vector, all_vectors)[0]\n",
 "\n",
 "  selected_director = selected_movie['director']\n",
 "  selected_language = selected_movie['originalLanguage']\n",
 "  selected_genres = [col for col in feature_df.columns if selected_movie.get(col, 0) == 1]\n",
 "\n",
 "  for idx in range(len(similarities)):\n",
 "movie = df.iloc[idx]\n",
 "if idx == movie_idx:\n",
 " similarities[idx] = -1  # Exclude the same movie\n",
 " continue\n",
 "if movie['director'] == selected_director:\n",
 " similarities[idx] += 0.1\n",
 "movie_genres = [col for col in feature_df.columns if movie.get(col, 0) == 1]\n",
 "genre_overlap = len(set(selected_genres).intersection(set(movie_genres)))\n",
 "similarities[idx] += 0.05 * genre_overlap\n",
 "if movie['originalLanguage'] == selected_language:\n",
 " similarities[idx] += 0.03\n",
 "\n",
 "  top_indices = similarities.argsort()[::-1][:top_n]\n",
 "\n",
 "  explanation['movie_details'] = {\n",
 "'title': selected_movie['title'],\n",
 "'genres': selected_genres,\n",
 "'director': director_decoder.get(selected_director, 'Unknown'),\n",
 "'original_language': language_decoder.get(selected_language, 'Unknown'),\n",
 "'runtime_minutes': selected_movie['runtimeMinutes'],\n",
 "'tomatoMeter': selected_movie.get('tomatoMeter', 'N/A'),\n",
 "'year': selected_movie.get('year', 'N/A')\n",
 "  }\n",
 "\n",
 "  explanation['reasoning'] = (\n",
 "f\"Since you liked '{selected_movie['title']}', we're recommending movies that share similar \"\n",
 "f\"genres ({', '.join(selected_genres)}), the same director \"\n",
 "f\"({director_decoder.get(selected_director, 'Unknown')}), or are in the same language \"\n",
 "f\"({language_decoder.get(selected_language, 'Unknown')}).\"\n",
 "  )\n",
 "\n",
 " elif input_type == 'genre':\n",
 "  if user_input not in feature_df.columns:\n",
 "return f\"Genre '{user_input}' not found.\"\n",
 "  \n",
 "  # Filter to include only movies with this genre\n",
 "  matching_indices = feature_df[feature_df[user_input] == 1].index\n",
 "  if len(matching_indices) == 0:\n",
 "return f\"No movies found with genre '{user_input}'.\"\n",
 "  \n",
 "  # Use the filtered dataframes\n",
 "  selected_df = df.loc[matching_indices].reset_index(drop=True)\n",
 "  selected_features = feature_df.loc[matching_indices].reset_index(drop=True)\n",
 "  \n",
 "  target_vector = selected_features.mean().values.reshape(1, -1)\n",
 "  all_vectors = selected_features.values\n",
 "  similarities = cosine_similarity(target_vector, all_vectors)[0]\n",
 "  top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)\n",
 "  \n",
 "  df = selected_df\n",
 "  feature_df = selected_features\n",
 "\n",
 " elif input_type == 'language':\n",
 "  rev_language_decoder = {v.lower(): k for k, v in language_decoder.items()}\n",
 "  lang_code = rev_language_decoder.get(user_input.lower())\n",
 "  if lang_code is None:\n",
 "return f\"Language '{user_input}' not found.\"\n",
 "\n",
 "  matching_indices = df[df['originalLanguage'] == lang_code].index\n",
 "  if matching_indices.empty:\n",
 "return f\"No movies found for language '{user_input}'.\"\n",
 "\n",
 "  target_vector = feature_df.loc[matching_indices].mean().values.reshape(1, -1)\n",
 "  selected_df = df.loc[matching_indices]\n",
 "  selected_features = feature_df.loc[matching_indices]\n",
 "  all_vectors = selected_features.values\n",
 "  similarities = cosine_similarity(target_vector, all_vectors)[0]\n",
 "  top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)\n",
 "\n",
 " elif input_type == 'year':\n",
 "  try:\n",
 "year = int(user_input)\n",
 "  except ValueError:\n",
 "return \"Invalid year format. Please enter a number.\"\n",
 "  matching_indices = df[df['year'] == year].index\n",
 "  if matching_indices.empty:\n",
 "return f\"No movies found for year '{user_input}'.\"\n",
 "\n",
 "  target_vector = feature_df.loc[matching_indices].mean().values.reshape(1, -1)\n",
 "  selected_df = df.loc[matching_indices]\n",
 "  selected_features = feature_df.loc[matching_indices]\n",
 "  all_vectors = selected_features.values\n",
 "  similarities = cosine_similarity(target_vector, all_vectors)[0]\n",
 "  top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)\n",
 "\n",
 " else:\n",
 "  return f\"Unsupported input type: {input_type}\"\n",
 "\n",
 " for idx in top_indices:\n",
 "  movie_data = df.iloc[idx] if input_type == 'title' else (selected_df.iloc[idx] if input_type in ['language', 'year'] else df.iloc[idx])\n",
 "  similarity_score = similarities[idx]\n",
 "  genres = [\n",
 "col for col in feature_df.columns\n",
 "if col not in ['runtimeMinutes', 'director', 'originalLanguage'] and movie_data.get(col, 0) == 1\n",
 "  ]\n",
 "  director = director_decoder.get(movie_data['director'], 'Unknown')\n",
 "  language = language_decoder.get(movie_data['originalLanguage'], 'Unknown')\n",
 "\n",
 "  movie_features = {\n",
 "'title': movie_data['title'],\n",
 "'similarity_score': similarity_score,\n",
 "'genres': genres,\n",
 "'year': movie_data.get('year', 'N/A'),\n",
 "'director': director,\n",
 "'original_language': language,\n",
 "'runtime_minutes': movie_data['runtimeMinutes'],\n",
 "'tomatoMeter': movie_data.get('tomatoMeter', 'N/A')\n",
 "  }\n",
 "\n",
 "  explanation['recommendations'].append(movie_features)\n",
 "\n",
 " return explanation\n",
 "\n",
 "# Evaluation function for Precision\n",
 "def evaluate_precision(result, input_type, user_input):\n",
 " if isinstance(result, str):\n",
 "  return None\n",
 " recommendations = result['recommendations']\n",
 " top_n = len(recommendations)\n",
 " if input_type == 'genre':\n",
 "  return sum([1 if user_input in rec['genres'] else 0 for rec in recommendations]) / top_n\n",
 " elif input_type == 'language':\n",
 "  return sum([1 if rec['original_language'].lower() == user_input.lower() else 0 for rec in recommendations]) / top_n\n",
 " elif input_type == 'year':\n",
 "  try:\n",
 "year = int(user_input)\n",
 "return sum([1 if rec['year'] == year else 0 for rec in recommendations]) / top_n\n",
 "  except ValueError:\n",
 "return None\n",
 " return None\n",
 "\n",
 "def show_recommendations(result, df):\n",
 " if isinstance(result, str):\n",
 "  print(result)\n",
 " else:\n",
 "  if result['movie_details']:\n",
 "details = result['movie_details']\n",
 "print(f\"\\nSelected Movie: {details['title']}\")\n",
 "print(f\"Genres: {', '.join(details['genres'])}\")\n",
 "print(f\"Year: {(details['year'])}\")\n",
 "print(f\"Director: {details['director']}\")\n",
 "print(f\"Original Language: {details['original_language']}\")\n",
 "print(f\"Runtime: {details['runtime_minutes']} minutes\")\n",
 "print(\"----------------------------------------------------\")\n",
 "\n",
 "  print(f\"\\nRecommendations for '{result['movie_title']}':\")\n",
 "  for idx, recommendation in enumerate(result['recommendations'], 1):\n",
 "print(f\"{idx}. Movie: {recommendation['title']}\")\n",
 "print(f\"Genres: {', '.join(recommendation['genres'])}\")\n",
 "print(f\"Year: {recommendation['year']}\")\n",
 "print(f\"Director: {recommendation['director']}\")\n",
 "print(f\"Language: {recommendation['original_language']}\")\n",
 "print(f\"Runtime: {recommendation['runtime_minutes']} minutes\")\n",
 "print(f\"TomatoMeter: {recommendation['tomatoMeter']}%\")\n",
 "print(\"---\")\n",
 "\n",
 "  input_type = result['input_type']\n",
 "  user_input = result['user_input']\n",
 "\n",
 "  if input_type in ['genre', 'language', 'year']:\n",
 "precision_val = evaluate_precision(result, input_type, user_input)\n",
 "\n",
 "print(\"\\n--- Evaluation Metrics ---\")\n",
 "if precision_val is not None:\n",
 " print(f\"Precision@{len(result['recommendations'])}: {precision_val:.2f}\")\n",
 "\n",
 "def menu(df, feature_df):\n",
 " print(\"Choose your input type:\")\n",
 " print(\"1. Movie Title\")\n",
 " print(\"2. Genre\")\n",
 " print(\"3. Language\")\n",
 " print(\"4. Release Year\")\n",
 "\n",
 " choice = input(\"Enter your choice (1-4): \")\n",
 "\n",
 " input_type_map = {\n",
 "  '1': 'title',\n",
 "  '2': 'genre',\n",
 "  '3': 'language',\n",
 "  '4': 'year'\n",
 " }\n",
 "\n",
 " if choice not in input_type_map:\n",
 "  print(\"Invalid choice.\")\n",
 "  return\n",
 "\n",
 " input_type = input_type_map[choice]\n",
 "\n",
 " if input_type == 'genre':\n",
 "  available_genres = [col for col in feature_df.columns if col not in ['runtimeMinutes', 'director', 'originalLanguage']]\n",
 "  print(\"\\nAvailable genres:\")\n",
 "  for i, genre in enumerate(available_genres, 1):\n",
 "print(f\"{i}. {genre}\")\n",
 "  genre_index = int(input(\"\\nEnter genre number: \")) - 1\n",
 "  if genre_index < 0 or genre_index >= len(available_genres):\n",
 "print(\"Invalid genre selection.\")\n",
 "return\n",
 "  user_input = available_genres[genre_index]\n",
 "\n",
 " elif input_type == 'language':\n",
 "  languages = sorted(set(language_decoder.values()))\n",
 "  print(\"\\nAvailable languages:\")\n",
 "  for i, lang in enumerate(languages, 1):\n",
 "print(f\"{i}. {lang}\")\n",
 "  lang_index = int(input(\"\\nEnter language number: \")) - 1\n",
 "  if lang_index < 0 or lang_index >= len(languages):\n",
 "print(\"Invalid language selection.\")\n",
 "return\n",
 "  user_input = languages[lang_index]\n",
 "\n",
 " elif input_type == 'year':\n",
 "  user_input = input(\"Enter the release year (e.g., 2020): \")\n",
 "  try:\n",
 "year = int(user_input)\n",
 "  except ValueError:\n",
 "print(\"Invalid year format. Please enter a valid year (e.g., 2020).\")\n",
 "return\n",
 "  if df[df['year'] == year].empty:\n",
 "print(f\"No movies found for the year '{year}'.\")\n",
 "return\n",
 "\n",
 " else:\n",
 "  user_input = input(f\"Enter {input_type}: \")\n",
 "\n",
 " result = content_recommender(user_input, df, feature_df, top_n=5, input_type=input_type)\n",
 " show_recommendations(result, df)\n"
]
  },
  {
"cell_type": "code",

"id": "bbb31912-fff1-4592-886e-671796227034",
"metadata": {},
"outputs": [],
"source": [
 "menu(content_df, content_features)"
]
  },
  {
"cell_type": "markdown",
"id": "04f30630-fca0-4db3-b8ab-94f3c19dba1a",
"metadata": {},
"source": [
 "## NLP TECHNIQUES (TextBox)"
]
  },
  {
"cell_type": "code",

"id": "28370698-1da2-4f05-8e0b-19d7580f5bd8",
"metadata": {},
"outputs": [],
"source": [
 "from sklearn.feature_extraction.text import TfidfVectorizer\n",
 "from sklearn.metrics.pairwise import cosine_similarity\n",
 "import nltk\n",
 "import gensim.downloader as api\n",
 "\n",
 "# Download necessary data\n",
 "nltk.download('punkt')\n",
 "\n",
 "# Load real pre-trained Word2Vec model (GloVe)\n",
 "w2v_model = api.load(\"glove-wiki-gigaword-300\")\n",
 "\n",
 "# --- User query ---\n",
 "user_query = input(\"Describe the type of movie you're looking for: \").strip()\n",
 "\n",
 "# --- Clean and parse query for filters ---\n",
 "min_year, max_year = None, None\n",
 "query_language = None\n",
 "query_genre = None\n",
 "director_name = None\n",
 "min_score = None\n",
 "\n",
 "# Extract years\n",
 "year_range = re.findall(r'(\\d{4})', user_query)\n",
 "if len(year_range) >= 2:\n",
 " min_year, max_year = int(year_range[0]), int(year_range[1])\n",
 "elif len(year_range) == 1:\n",
 " min_year = max_year = int(year_range[0])\n",
 "\n",
 "# Extract all known languages from decoder\n",
 "all_languages = [v.lower() for v in language_decoder.values()]\n",
 "for lang in all_languages:\n",
 " if lang in user_query.lower():\n",
 "  query_language = lang\n",
 "  break\n",
 "\n",
 "# Genre extraction from list of known genres\n",
 "genre_columns = [\n",
 " 'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime', 'Documentary',\n",
 " 'Drama', 'Entertainment', 'Faith & spirituality', 'Fantasy', 'Health & wellness',\n",
 " 'History', 'Horror', 'Kids & family', 'LGBTQ+', 'Music', 'Nature', 'Other',\n",
 " 'Reality', 'Romance', 'Sci-fi', 'Sports', 'Variety Show', 'War', 'Western'\n",
 "]\n",
 "for genre in genre_columns:\n",
 " if genre.lower() in user_query.lower():\n",
 "  query_genre = genre\n",
 "  break\n",
 "\n",
 "# Extract tomatoMeter threshold\n",
 "score_match = re.search(r'tomato.*?(\\d+)', user_query.lower())\n",
 "if score_match:\n",
 " min_score = int(score_match.group(1))\n",
 "\n",
 "# Extract full director name\n",
 "director_match = re.search(r'director(?: is| named)? ([A-Za-z ]+)', user_query.lower())\n",
 "if director_match:\n",
 " director_name = director_match.group(1).strip().lower()\n",
 "\n",
 "# clean query for better TF-IDF\n",
 "clean_query = user_query\n",
 "for lang in all_languages:\n",
 " clean_query = clean_query.replace(lang, '')\n",
 "if query_genre:\n",
 " clean_query = clean_query.replace(query_genre, '')\n",
 "clean_query = re.sub(r'\\d{4}', '', clean_query)\n",
 "clean_query = re.sub(r'director.*?(is|named)? ?[A-Za-z ]+', '', clean_query, flags=re.IGNORECASE)\n",
 "clean_query = re.sub(r'tomato.*?\\d+', '', clean_query, flags=re.IGNORECASE)\n",
 "\n",
 "# --- TF-IDF similarity ---\n",
 "tfidf_vectorizer = TfidfVectorizer(stop_words='english')\n",
 "tfidf_matrix = tfidf_vectorizer.fit_transform(content_df['movie_info'].fillna(''))\n",
 "query_tfidf = tfidf_vectorizer.transform([clean_query])\n",
 "tfidf_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()\n",
 "\n",
 "# --- Real Word2Vec similarity ---\n",
 "def get_average_vector(text, model, vector_size=300):\n",
 " tokens = nltk.word_tokenize(text.lower())\n",
 " vectors = [model[word] for word in tokens if word in model]\n",
 " return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)\n",
 "\n",
 "content_df['w2v_vector'] = content_df['movie_info'].apply(lambda x: get_average_vector(x, w2v_model))\n",
 "query_w2v_vector = get_average_vector(user_query, w2v_model)\n",
 "\n",
 "w2v_similarities = content_df['w2v_vector'].apply(\n",
 " lambda x: cosine_similarity([x], [query_w2v_vector])[0][0]\n",
 ")\n",
 "\n",
 "# --- Combine similarities ---\n",
 "combined_similarity = 0.5 * tfidf_similarities + 0.5 * w2v_similarities\n",
 "content_df['similarity'] = combined_similarity\n",
 "\n",
 "# --- Filtering ---\n",
 "filtered_df = content_df.copy()\n",
 "\n",
 "# Year filter\n",
 "if min_year is not None:\n",
 " filtered_df = filtered_df[filtered_df['year'].between(min_year, max_year if max_year else min_year)]\n",
 "\n",
 "# Language filter\n",
 "if query_language:\n",
 " matching_lang_ids = [k for k, v in language_decoder.items() if v.lower() == query_language]\n",
 " filtered_df = filtered_df[filtered_df['originalLanguage'].isin(matching_lang_ids)]\n",
 "\n",
 "# Genre filter\n",
 "if query_genre:\n",
 " filtered_df = filtered_df[filtered_df[query_genre] == 1]\n",
 "\n",
 "# Director filter\n",
 "if director_name:\n",
 " matching_director_ids = [\n",
 "  k for k, v in director_decoder.items() if director_name in v.lower()\n",
 " ]\n",
 " if matching_director_ids:\n",
 "  filtered_df = filtered_df[filtered_df['director'].isin(matching_director_ids)]\n",
 "\n",
 "# Rotten Tomatoes score filter\n",
 "if min_score is not None:\n",
 " filtered_df = filtered_df[filtered_df['tomatoMeter'] >= min_score]\n",
 "\n",
 "# --- Final result ---\n",
 "top_recommendations = filtered_df.sort_values(by='similarity', ascending=False).head(10)\n",
 "\n",
 "# --- Display results ---\n",
 "print(\"\\nHere are 10 movies that match your description:\\n\")\n",
 "for i, (_, row) in enumerate(top_recommendations.iterrows(), 1):\n",
 " genres = [genre for genre in genre_columns if row.get(genre, 0) == 1]\n",
 " genre_str = \", \".join(genres) if genres else \"N/A\"\n",
 " language = language_decoder.get(row['originalLanguage'], 'Unknown')\n",
 " director = director_decoder.get(row['director'], 'Unknown')\n",
 " runtime = f\"{row['runtimeMinutes']} minutes\" if not pd.isna(row.get('runtimeMinutes')) else \"N/A\"\n",
 " tomato_score = f\"{row['tomatoMeter']}%\" if not pd.isna(row.get('tomatoMeter')) else \"N/A\"\n",
 "\n",
 " print(f\"{i}. Movie: {row['title']}\")\n",
 " print(f\"Genres: {genre_str}\")\n",
 " print(f\"Year: {int(row['year']) if not pd.isna(row['year']) else 'N/A'}\")\n",
 " print(f\"Director: {director}\")\n",
 " print(f\"Language: {language}\")\n",
 " print(f\"Runtime: {runtime}\")\n",
 " print(f\"TomatoMeter: {tomato_score}\")\n",
 " print()"
]
  },
  {
"cell_type": "code",

"id": "1f621203-223d-4a04-b285-0746d2ea9a5e",
"metadata": {},
"outputs": [],
"source": [
 "# Set display option to show full content in cells\n",
 "pd.set_option('display.max_colwidth', None)\n",
 "\n",
 "# Now display the full movie_info for rows where movie_info_flag == 1\n",
 "content_df[content_df['movie_info_flag'] == 1][['title', 'movie_info', 'year']]\n"
]
  },
  {
"cell_type": "markdown",
"id": "c824d494-fa26-4e5e-b87f-f836c21b93ea",
"metadata": {},
"source": [
 "## B) Collaborative Filtering"
]
  },
  {
"cell_type": "code",

"id": "32d7e081-6bfe-4929-8d20-11f6d0268fbe",
"metadata": {},
"outputs": [],
"source": [
 "collaborative_df"
]
  },
  {
"cell_type": "code",

"id": "7ecc59c8-8b34-4b29-a3c1-6658dd69a6f8",
"metadata": {},
"outputs": [],
"source": [
 "collaborative_df.shape"
]
  },
  {
"cell_type": "code",

"id": "9ce5e798-d763-4983-8524-aaafbeca4949",
"metadata": {},
"outputs": [],
"source": [
 "# Step 1: Get the top 100 movies based on rating count\n",
 "movie_count = collaborative_df.groupby('title').size().reset_index(name='rating_count')\n",
 "top_movies = movie_count.sort_values(by='rating_count', ascending=False).head(7800)\n",
 "\n",
 "# Step 2: Get the titles of the top 100 movies\n",
 "top_movie_titles = top_movies['title'].tolist()\n",
 "\n",
 "# Step 3: Filter collaborative_df to include only these top 100 movies\n",
 "top_movies_collab_df = collaborative_df[collaborative_df['title'].isin(top_movie_titles)]"
]
  },
  {
"cell_type": "code",

"id": "e746dae1-8be4-491e-a485-b8638cad3cf9",
"metadata": {},
"outputs": [],
"source": [
 "top_movies_collab_df.shape"
]
  },
  {
"cell_type": "code",

"id": "013f0a74-8091-4a61-97bf-95c2d8d2cdf1",
"metadata": {},
"outputs": [],
"source": [
 "# Save top collaborative recommendations dataframe\n",
 "top_movies_collab_df.to_csv(\"top_movies_collab_df.csv\", index=False)\n"
]
  },
  {
"cell_type": "code",

"id": "3414f506-bc14-49f1-941f-eab8c9a714db",
"metadata": {},
"outputs": [],
"source": [
 "# Create the user-item matrix\n",
 "rating_matrix = collaborative_df.pivot(index='userName', columns='title', values='standardized_score')\n",
 "\n",
 "# Fill NaN values with 0 \n",
 "rating_matrix = rating_matrix.fillna(0)\n"
]
  },
  {
"cell_type": "raw",
"id": "293644a9-2fac-4e76-af2e-cb79c3409a2f",
"metadata": {},
"source": [
 "rating_matrix"
]
  },
  {
"cell_type": "code",

"id": "af9b05b3-aed4-4136-a505-c3a1d9c31571",
"metadata": {},
"outputs": [],
"source": [
 "top_movies_collab_df.shape"
]
  },
  {
"cell_type": "code",

"id": "7d1b8685-3ecf-425e-b989-0245be0e0908",
"metadata": {},
"outputs": [],
"source": [
 "from sklearn.decomposition import TruncatedSVD\n",
 "from sklearn.metrics.pairwise import cosine_similarity\n",
 "import numpy as np\n",
 "\n",
 "def collaborative_menu(top_movies_collab_df, top_n=10):\n",
 " # Step 1: Create the user-item matrix\n",
 " rating_matrix = top_movies_collab_df.pivot(index='userName', columns='title', values='standardized_score')\n",
 " \n",
 " # Step 2: Fill missing values with 0\n",
 " rating_matrix = rating_matrix.fillna(0)\n",
 " \n",
 " # Step 3: Apply SVD for dimensionality reduction\n",
 " svd = TruncatedSVD(n_components=20, random_state=42)\n",
 " matrix_svd = svd.fit_transform(rating_matrix)\n",
 " \n",
 " # Step 4: Compute cosine similarity between users\n",
 " similarity_matrix = cosine_similarity(matrix_svd)\n",
 " \n",
 " # Step 5: Prompt for user input\n",
 " user_input = input(\"Please enter your userName: \").strip()\n",
 " \n",
 " if user_input:\n",
 "  try:\n",
 "user_id = int(user_input)\n",
 "  except ValueError:\n",
 "print(\"Invalid input. Please enter a valid numeric user ID.\")\n",
 "return\n",
 "\n",
 "  if user_id not in rating_matrix.index:\n",
 "print(f\"User '{user_id}' not found in the dataset.\")\n",
 "return\n",
 "\n",
 "  user_idx = rating_matrix.index.get_loc(user_id)\n",
 "  user_similarity = similarity_matrix[user_idx]\n",
 "\n",
 "  # Show top 3 similar users\n",
 "  most_similar_users = [\n",
 "(int(rating_matrix.index[i]), sim)\n",
 "for i, sim in enumerate(user_similarity)\n",
 "if i != user_idx\n",
 "  ]\n",
 "  most_similar_users = sorted(most_similar_users, key=lambda x: x[1], reverse=True)[:3]\n",
 "\n",
 "  print(\"\\nUsers with the most similar preferences to you:\")\n",
 "  for uid, score in most_similar_users:\n",
 "print(f\"User {uid} (Similarity Score: {score:.4f})\")\n",
 "\n",
 "  # Recommend based on similar users\n",
 "  similar_users_idx = np.argsort(user_similarity)[::-1]\n",
 "  user_seen_movies = set(rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index)\n",
 "\n",
 "  recommendations = {}\n",
 "  for idx in similar_users_idx:\n",
 "similar_user_ratings = rating_matrix.iloc[idx]\n",
 "for movie, rating in similar_user_ratings.items():\n",
 " if rating > 0 and movie not in user_seen_movies:\n",
 "  recommendations[movie] = recommendations.get(movie, 0) + rating\n",
 "\n",
 "  if recommendations:\n",
 "max_score = max(recommendations.values())\n",
 "percent_recommendations = {\n",
 " movie: (score / max_score) * 100 for movie, score in recommendations.items()\n",
 "}\n",
 "sorted_recommendations = sorted(percent_recommendations.items(), key=lambda x: x[1], reverse=True)\n",
 "  else:\n",
 "sorted_recommendations = []\n",
 "\n",
 "  # Display recommendations\n",
 "  print(f\"\\nHere are {top_n} movies we think you'll enjoy, based on your preferences:\\n\")\n",
 "  for movie, percent in sorted_recommendations[:top_n]:\n",
 "print(f\"{movie}: {percent:.1f}%\")\n"
]
  },
  {
"cell_type": "code",

"id": "3232541c-0208-43a6-8a6a-d0d3d506a6fc",
"metadata": {},
"outputs": [],
"source": [
 "collaborative_menu(top_movies_collab_df)"
]
  },
  {
"cell_type": "code",

"id": "fbc9b6bb-f7d2-4f9f-96d5-51f98c37d257",
"metadata": {},
"outputs": [],
"source": [
 "def compare_multiple_users(user_ids, df):\n",
 " # Pivot to get user-item matrix\n",
 " rating_matrix = df.pivot(index='userName', columns='title', values='standardized_score')\n",
 "\n",
 " # Check if all users exist\n",
 " missing_users = [uid for uid in user_ids if uid not in rating_matrix.index]\n",
 " if missing_users:\n",
 "  print(f\"User(s) not found in the dataset: {missing_users}\")\n",
 "  return\n",
 "\n",
 " # Collect all user ratings\n",
 " comparison = pd.DataFrame()\n",
 " for uid in user_ids:\n",
 "  comparison[f'User {uid}'] = rating_matrix.loc[uid]\n",
 "\n",
 " # Drop movies not rated by any of them\n",
 " comparison = comparison.dropna(how='all')\n",
 "\n",
 " # Print results\n",
 " print(f\"\\nComparison of Users {', '.join(map(str, user_ids))}:\\n\")\n",
 " print(comparison.sort_index())\n",
 "\n",
 "# Example usage:\n",
 "compare_multiple_users([4, 2693, 1241, 8220], top_movies_collab_df)\n"
]
  },
  {
"cell_type": "markdown",
"id": "5ed9f0e0-40fc-4318-abb6-f652354f150a",
"metadata": {},
"source": [
 "## HYBRID FILTERING"
]
  },
  {
"cell_type": "code",

"id": "50fcc850-0d93-4e25-913e-14424b4b50b9",
"metadata": {},
"outputs": [],
"source": [
 "import numpy as np\n",
 "from sklearn.decomposition import TruncatedSVD\n",
 "from sklearn.metrics.pairwise import cosine_similarity\n",
 "\n",
 "def hybrid_recommender(user_id, movie_title, content_df, content_features, top_movies_collab_df, \n",
 "  content_weight=0.5, collab_weight=0.5, top_n=10):\n",
 " \"\"\"\n",
 " Hybrid recommendation system that combines content-based and collaborative filtering\n",
 " \"\"\"\n",
 " # Normalize weights\n",
 " total_weight = content_weight + collab_weight\n",
 " content_weight = content_weight / total_weight\n",
 " collab_weight = collab_weight / total_weight\n",
 " \n",
 " # 1. Get content-based recommendations\n",
 " content_result = content_recommender(movie_title, content_df, content_features, \n",
 "  top_n=top_n*2, input_type='title')\n",
 " \n",
 " # Process content recommendations\n",
 " content_recommendations = {}\n",
 " if not isinstance(content_result, str):\n",
 "  content_recommendations = {\n",
 "rec['title']: rec['similarity_score'] \n",
 "for rec in content_result['recommendations']\n",
 "  }\n",
 "  # Normalize scores\n",
 "  if content_recommendations:\n",
 "max_content_score = max(content_recommendations.values())\n",
 "content_recommendations = {\n",
 " movie: (score / max_content_score) * 100 \n",
 " for movie, score in content_recommendations.items()\n",
 "}\n",
 " \n",
 " # 2. Get collaborative recommendations\n",
 " collab_recommendations = {}\n",
 " \n",
 " # Create the user-item matrix\n",
 " rating_matrix = top_movies_collab_df.pivot(index='userName', columns='title', values='standardized_score')\n",
 " rating_matrix = rating_matrix.fillna(0)\n",
 " \n",
 " # Apply SVD\n",
 " svd = TruncatedSVD(n_components=20, random_state=42)\n",
 " matrix_svd = svd.fit_transform(rating_matrix)\n",
 " similarity_matrix = cosine_similarity(matrix_svd)\n",
 " \n",
 " # Check if user exists\n",
 " if user_id in rating_matrix.index:\n",
 "  user_idx = rating_matrix.index.get_loc(user_id)\n",
 "  user_similarity = similarity_matrix[user_idx]\n",
 "  similar_users_idx = np.argsort(user_similarity)[::-1]\n",
 "  user_seen_movies = set(rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index)\n",
 "  \n",
 "  # Get recommendations from similar users\n",
 "  for idx in similar_users_idx:\n",
 "if idx == user_idx:\n",
 " continue\n",
 "similar_user_ratings = rating_matrix.iloc[idx]\n",
 "for movie, rating in similar_user_ratings.items():\n",
 " if rating > 0 and movie not in user_seen_movies:\n",
 "  collab_recommendations[movie] = collab_recommendations.get(movie, 0) + rating\n",
 "  \n",
 "  # Normalize collaborative scores\n",
 "  if collab_recommendations:\n",
 "max_collab_score = max(collab_recommendations.values())\n",
 "collab_recommendations = {\n",
 " movie: (score / max_collab_score) * 100 \n",
 " for movie, score in collab_recommendations.items()\n",
 "}\n",
 " \n",
 " # 3. Combine recommendations with weighted scores\n",
 " hybrid_scores = {}\n",
 " \n",
 " # Add content-based scores with weight\n",
 " for movie, score in content_recommendations.items():\n",
 "  hybrid_scores[movie] = content_weight * score\n",
 " \n",
 " # Add collaborative scores with weight\n",
 " for movie, score in collab_recommendations.items():\n",
 "  if movie in hybrid_scores:\n",
 "hybrid_scores[movie] += collab_weight * score\n",
 "  else:\n",
 "hybrid_scores[movie] = collab_weight * score\n",
 " \n",
 " # Sort and return top recommendations\n",
 " sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)\n",
 " \n",
 " return sorted_recommendations[:top_n]\n",
 "\n",
 "def show_hybrid_recommendations(recommendations, content_df, content_features, user_movie):\n",
 " # First, display details about the input movie\n",
 " movie_details = content_df[content_df['title'] == user_movie]\n",
 " if not movie_details.empty:\n",
 "  details = movie_details.iloc[0]\n",
 "  print(\"\\n===== Movie You Enjoyed =====\")\n",
 "  print(f\"Title: {details['title']}\")\n",
 "  print(f\"Year: {details.get('year', 'N/A')}\")\n",
 "  \n",
 "  genres = [col for col in content_features.columns \n",
 "  if col not in ['runtimeMinutes', 'director', 'originalLanguage'] \n",
 "  and details.get(col, 0) == 1]\n",
 "  print(f\"Genres: {', '.join(genres)}\")\n",
 "  \n",
 "  director_id = details.get('director', 'Unknown')\n",
 "  director_name = director_decoder.get(director_id, 'Unknown')\n",
 "  print(f\"Director: {director_name}\")\n",
 "  \n",
 "  language_id = details.get('originalLanguage', 'Unknown')\n",
 "  language_name = language_decoder.get(language_id, 'Unknown')\n",
 "  print(f\"Language: {language_name}\")\n",
 "  \n",
 "  print(f\"Runtime: {details.get('runtimeMinutes', 'N/A')} minutes\")\n",
 "  print(f\"TomatoMeter: {details.get('tomatoMeter', 'N/A')}%\")\n",
 "  print(\"=\" * 30)\n",
 " \n",
 " # Then show recommendations\n",
 " print(\"\\n===== Hybrid Recommendations =====\")\n",
 " for i, (movie, score) in enumerate(recommendations, 1):\n",
 "  # Find movie details in content_df\n",
 "  movie_details = content_df[content_df['title'] == movie]\n",
 "  \n",
 "  if not movie_details.empty:\n",
 "details = movie_details.iloc[0]\n",
 "print(f\"{i}. {movie} (Score: {score:.1f}%)\")\n",
 "print(f\"Year: {details.get('year', 'N/A')}\")\n",
 "\n",
 "genres = [col for col in content_features.columns \n",
 "if col not in ['runtimeMinutes', 'director', 'originalLanguage'] \n",
 "and details.get(col, 0) == 1]\n",
 "print(f\"Genres: {', '.join(genres)}\")\n",
 "\n",
 "director_id = details.get('director', 'Unknown')\n",
 "director_name = director_decoder.get(director_id, 'Unknown')\n",
 "print(f\"Director: {director_name}\")\n",
 "\n",
 "language_id = details.get('originalLanguage', 'Unknown')\n",
 "language_name = language_decoder.get(language_id, 'Unknown')\n",
 "print(f\"Language: {language_name}\")\n",
 "\n",
 "print(f\"Runtime: {details.get('runtimeMinutes', 'N/A')} minutes\")\n",
 "print(f\"TomatoMeter: {details.get('tomatoMeter', 'N/A')}%\")\n",
 "  else:\n",
 "print(f\"{i}. {movie} (Score: {score:.1f}%)\")\n",
 "  print(\"---\")\n",
 "\n",
 "def evaluate_hybrid_recommender(recommendations, user_id, movie_title, content_df, content_features, k=20000):\n",
 " \"\"\"\n",
 " Evaluate hybrid recommender system using precision, recall, and F1 score at K\n",
 " \"\"\"\n",
 " # Get actual genres of the original movie\n",
 " movie_details = content_df[content_df['title'] == movie_title]\n",
 " if movie_details.empty:\n",
 "  print(f\"Cannot find details for movie '{movie_title}'\")\n",
 "  return None\n",
 " \n",
 " original_movie = movie_details.iloc[0]\n",
 " \n",
 " # Get genres of the original movie\n",
 " original_genres = [col for col in content_features.columns \n",
 "  if col not in ['runtimeMinutes', 'director', 'originalLanguage'] \n",
 "  and original_movie.get(col, 0) == 1]\n",
 " \n",
 " # Get director of the original movie\n",
 " original_director = original_movie['director']\n",
 " \n",
 " # Get relevant movies (same genres or director)\n",
 " relevant_movies = set()\n",
 " for _, row in content_df.iterrows():\n",
 "  # Check if shares any genre\n",
 "  movie_genres = [col for col in content_features.columns \n",
 "  if col not in ['runtimeMinutes', 'director', 'originalLanguage'] \n",
 "  and row.get(col, 0) == 1]\n",
 "  \n",
 "  if any(genre in original_genres for genre in movie_genres) or row['director'] == original_director:\n",
 "relevant_movies.add(row['title'])\n",
 " \n",
 " # Remove the original movie from the relevant set\n",
 " if movie_title in relevant_movies:\n",
 "  relevant_movies.remove(movie_title)\n",
 " \n",
 " # Get recommended movies\n",
 " recommended_movies = [movie for movie, _ in recommendations[:k]]\n",
 " \n",
 " # Calculate metrics\n",
 " relevant_recommended = set(recommended_movies) & relevant_movies\n",
 " \n",
 " precision = len(relevant_recommended) / len(recommended_movies) if recommended_movies else 0\n",
 " recall = len(relevant_recommended) / len(relevant_movies) if relevant_movies else 0\n",
 " f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0\n",
 " \n",
 " metrics = {\n",
 "  f'precision@{k}': precision,\n",
 "  f'recall@{k}': recall,\n",
 "  f'f1@{k}': f1\n",
 " }\n",
 " \n",
 " return metrics\n",
 "\n",
 "def hybrid_menu(content_df, content_features, top_movies_collab_df):\n",
 " print(\"Welcome to the Hybrid Movie Recommender!\")\n",
 " \n",
 " # Get user ID\n",
 " user_id = input(\"Please enter your userName: \").strip()\n",
 " try:\n",
 "  user_id = int(user_id)\n",
 " except ValueError:\n",
 "  print(\"Invalid user ID. Please enter a numeric value.\")\n",
 "  return\n",
 " \n",
 " # Get a movie the user likes\n",
 " movie_title = input(\"Enter a movie you enjoyed: \").strip()\n",
 " \n",
 " # Define weights\n",
 " content_weight = 0.5\n",
 " collab_weight = 0.5\n",
 " \n",
 " # Get recommendations\n",
 " recommendations = hybrid_recommender(\n",
 "  user_id, movie_title, content_df, content_features, top_movies_collab_df, \n",
 "  content_weight, collab_weight\n",
 " )\n",
 " \n",
 " # Show recommendations\n",
 " show_hybrid_recommendations(recommendations, content_df, content_features, movie_title)\n",
 " \n",
 " # Evaluate the recommendations\n",
 " metrics = evaluate_hybrid_recommender(\n",
 "  recommendations, user_id, movie_title, content_df, content_features\n",
 " )\n",
 " \n",
 " # Display evaluation metrics\n",
 " if metrics:\n",
 "  print(\"\\n===== Evaluation Metrics =====\")\n",
 "  for metric_name, value in metrics.items():\n",
 "print(f\"{metric_name}: {value:.4f}\")"
]
  },
  {
"cell_type": "code",

"id": "1435e678-e586-4187-a66e-b7d3143cdfff",
"metadata": {},
"outputs": [],
"source": [
 "hybrid_menu(content_df, content_features, top_movies_collab_df)"
]
  },
  {
"cell_type": "code",

"id": "3fae7e3c-9750-4bbd-a035-b6ffeeacc5cc",
"metadata": {},
"outputs": [],
"source": [
 "# Show top recommended movies for user_id = 4\n",
 "user_id = 4\n",
 "user_recommendations = top_movies_collab_df[top_movies_collab_df['userName'] == user_id]\n",
 "\n",
 "# Display the results\n",
 "user_recommendations\n"
]
  }
 ],
 "metadata": {
  "kernelspec": {
"display_name": "Python 3 (ipykernel)",
"language": "python",
"name": "python3"
  },
  "language_info": {
"codemirror_mode": {
 "name": "ipython",
 "version": 3
},
"file_extension": ".py",
"mimetype": "text/x-python",
"name": "python",
"nbconvert_exporter": "python",
"pygments_lexer": "ipython3",
"version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
