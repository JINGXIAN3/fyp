{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "5aacd1fd-def1-4578-b8c3-20d7cc31fa8e",
   "metadata": {},
   "source": [
    "# <div align =center > Movie Recommender System </div>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f820419c-1098-45fa-8096-47e0b6a12c2d",
   "metadata": {},
   "source": [
    "# DATA LOADING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "84e8aac4-a725-4f4b-b2bf-05c56f432393",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import nltk\n",
    "import sklearn\n",
    "import re"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6ceb7e76-1999-434e-ac1c-b91c38cd7e81",
   "metadata": {},
   "outputs": [],
   "source": [
    "## pip install seaborn --upgrade"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92bff9be-f816-4f62-a1dd-61174151f7c7",
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "# Load the datasets (from rottenTomatoes and movielens)\n",
    "moviesRT_df = pd.read_csv(\"rotten_tomatoes_movies.csv\")\n",
    "reviewsRT_df = pd.read_csv(\"rotten_tomatoes_movie_reviews.csv\")\n",
    "moviesDescRT_df = pd.read_csv(\"movies_desc.csv\")\n",
    "movielens_df = pd.read_csv(\"movieslens.csv\")\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "145c304d-c90d-4eae-99a5-a16c1b8d87e2",
   "metadata": {},
   "source": [
    "# Initial Data Inspection"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b9f4c114-2f11-4be1-8574-660fd5ba5c1b",
   "metadata": {},
   "source": [
    "## Rotten Tomatoes Movies"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88b03330-83fd-4470-a16b-2b43a5d028d1",
   "metadata": {},
   "outputs": [],
   "source": [
    "moviesRT_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2a052918-b413-471f-9568-7d1c4919f548",
   "metadata": {},
   "outputs": [],
   "source": [
    "#show information of the dataset\n",
    "print(\" \" * 9, \"Dataset Information\")\n",
    "moviesRT_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ea06f6f1-3aea-4c79-897c-d92da22bfbc3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# View unique values in the each columns\n",
    "\n",
    "# Ratings\n",
    "ratings = moviesRT_df['rating'].unique()[:5]\n",
    "print(\"Unique rating values:\")\n",
    "print(ratings)\n",
    "print(\"\")\n",
    "\n",
    "#Rating Contents\n",
    "ratings_contents = moviesRT_df['ratingContents'].unique()[:5]\n",
    "print(\"Unique rating contents values:\")\n",
    "print(ratings_contents)\n",
    "print(\"\")\n",
    "\n",
    "#BoxOffice\n",
    "box = moviesRT_df['boxOffice'].unique()[:5]\n",
    "print(\"Unique boxOffice values:\")\n",
    "print(box)\n",
    "print(\"\")\n",
    "\n",
    "# Distributor\n",
    "distributor = moviesRT_df['distributor'].unique()[:5]\n",
    "print(\"Unique distributor values:\")\n",
    "print(distributor)\n",
    "print(\"\")\n",
    "\n",
    "# soundMix\n",
    "soundMix = moviesRT_df['soundMix'].unique()[:5]\n",
    "print(\"Unique soundMix values:\")\n",
    "print(soundMix)\n",
    "print(\"\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f5e87da-9c49-4a4b-8f64-0e47a4b9d206",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove unnecessary columns from df_movies \n",
    "columns_remove = ['ratingContents','releaseDateTheaters','releaseDateStreaming','boxOffice', 'distributor', 'soundMix', ]\n",
    "modify_movies = moviesRT_df.drop(columns=columns_remove, errors='ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "02faa575-0d14-4191-9a81-102bf86e26bb",
   "metadata": {},
   "outputs": [],
   "source": [
    "modify_movies"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "49e5d4c4-e7e3-4ef1-944c-4f93562c218e",
   "metadata": {},
   "source": [
    "## Rotten Tomatoes Reviews"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d65ed9f4-8c03-4a8b-88ef-2f2469ed991e",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviewsRT_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7dbfae63-5b8e-4158-956d-943467e5370f",
   "metadata": {},
   "outputs": [],
   "source": [
    "reviewsRT_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "08fdecf7-d606-4dd7-b789-64c931cdc9ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove unnecessary columns from the reviews dataset\n",
    "columns_remove = ['creationDate','isTopCritic','publicatioName','reviewUrl']\n",
    "modify_reviews = reviewsRT_df.drop(columns=columns_remove, errors='ignore')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f3589b45-3379-4850-a9d7-48240fd2ac1a",
   "metadata": {},
   "outputs": [],
   "source": [
    "modify_reviews.rename(columns={'criticName': 'userName'}, inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8aa2a6e1-f653-4f62-ae46-c7cd060bc9e9",
   "metadata": {},
   "outputs": [],
   "source": [
    "modify_reviews"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4aea1613-4090-428e-8761-abef1af7efd9",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Merge the modified datasets on the 'id' column\n",
    "merged_dataset = pd.merge(modify_movies,modify_reviews, on='id', how='inner')\n",
    "\n",
    "# Check the shape of the merged dataset\n",
    "print(f\"Shape of merged dataset: {merged_dataset.shape}\")\n",
    "merged_dataset"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "80cc5c9b-cd90-4501-8e66-e24d8b4c6554",
   "metadata": {},
   "source": [
    "## Rotten Tomatoes Movie Descriptions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2115f5ae-7793-4d98-8ca6-b9fd0f776e4f",
   "metadata": {},
   "outputs": [],
   "source": [
    "moviesDescRT_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4b37f96f-79ac-4423-9510-8cef08175a18",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Merge the datasets based on movie titles, director, and rating\n",
    "merged_dataset = merged_dataset.merge(moviesDescRT_df[['movie_title', 'movie_info', 'directors', 'content_rating']], \n",
    "                                      left_on=['title', 'director', 'rating'], \n",
    "                                      right_on=['movie_title', 'directors', 'content_rating'], \n",
    "                                      how='left')\n",
    "\n",
    "# Drop redundant columns after merging\n",
    "merged_dataset.drop(columns=['movie_title', 'directors', 'content_rating'], inplace=True)\n",
    "\n",
    "# Display the updated dataset\n",
    "merged_dataset\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "81cbc857-79b7-4c4e-86af-247c555b249b",
   "metadata": {},
   "source": [
    "## Movielens Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a54a39b4-6a42-40cc-bbd8-c36402646fe1",
   "metadata": {},
   "outputs": [],
   "source": [
    "movielens_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d65cb609-f5c2-4986-a492-ed638138acf6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to extract year\n",
    "def extract_year(title):\n",
    "    match = re.search(r'\\((\\d{4})\\)', title)\n",
    "    return int(match.group(1)) if match else None\n",
    "\n",
    "# Extract movie titles and years directly\n",
    "movielens_df['year'] = movielens_df['title'].apply(extract_year)\n",
    "movielens_df['clean_title'] = movielens_df['title'].str.replace(r' \\(\\d{4}\\)', '', regex=True)\n",
    "\n",
    "# Map movie years to merged dataset\n",
    "movie_years = dict(zip(movielens_df['clean_title'], movielens_df['year']))\n",
    "merged_dataset['year'] = merged_dataset['title'].map(movie_years)\n",
    "\n",
    "# Check results\n",
    "print(f\"Missing years: {merged_dataset['year'].isna().sum()}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7713c89f-f206-4231-ac87-f6e73273356a",
   "metadata": {},
   "outputs": [],
   "source": [
    "merged_dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "96717e5d-5274-48b0-bd28-1010e6d603c0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Reorder the attributes\n",
    "order = [\n",
    "    'id', 'title','movie_info', 'year', 'genre', 'director', 'writer', \n",
    "    'originalLanguage', 'runtimeMinutes', 'rating',\n",
    "    'audienceScore', 'tomatoMeter', 'originalScore',\n",
    "    'userName','reviewId', 'reviewText', 'scoreSentiment', 'reviewState'\n",
    "]\n",
    "\n",
    "# Reorder the columns \n",
    "merged_dataset = merged_dataset[[col for col in order if col in merged_dataset.columns]]\n",
    "\n",
    "combined_df = merged_dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "170d8154-1ce6-4132-a8bb-316a22c2cec3",
   "metadata": {},
   "outputs": [],
   "source": [
    "combined_df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d8db2462-afde-4c73-8d29-671a063e130a",
   "metadata": {},
   "source": [
    "The combined dataset consists of 1469543 rows and 17 columns"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "156d4987-85de-42a0-a3f6-39c6f9e29001",
   "metadata": {},
   "source": [
    "# EXPLORATARY DATA ANALYSIS (EDA)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ae52f23-7c11-4bcc-bb05-7114326ec68a",
   "metadata": {},
   "source": [
    "Before Data Preprocessing, we perform EDA to visualize patterns, detect outliers, and understand data distribution for the dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cdde67b8-d1e6-4d94-838f-431acb4d9ef8",
   "metadata": {},
   "outputs": [],
   "source": [
    "combined_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "192fcef0-74d0-4eba-8b69-90cf9515d3f3",
   "metadata": {},
   "outputs": [],
   "source": [
    "combined_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0354103d-403d-4984-9324-d27f42522fc4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Separate features into numerical and categorical\n",
    "numerical_features = ['year', 'runtimeMinutes', 'audienceScore', 'tomatoMeter']\n",
    "categorical_features = ['genre', 'director', 'writer','originalLanguage', 'rating', 'scoreSentiment', 'reviewState']\n",
    "text_features = ['title', 'reviewText', 'movie_info']\n",
    "id_features = ['id', 'reviewId']\n",
    "original_score = ['originalScore'] \n",
    "\n",
    "# Display the first few rows of each type\n",
    "print(\"Numerical features:\")\n",
    "print(combined_df[numerical_features].head())\n",
    "print(\"-\" * 50)\n",
    "\n",
    "print(\"Categorical features:\")\n",
    "print(combined_df[categorical_features].head())\n",
    "print(\"-\" * 50)\n",
    "\n",
    "print(\"Text features:\")\n",
    "print(combined_df[text_features].head())\n",
    "print(\"-\" * 50)\n",
    "\n",
    "print(\"Id features:\")\n",
    "print(combined_df[id_features].head())\n",
    "print(\"-\" * 50)\n",
    "\n",
    "print(\"Original Score:\")\n",
    "print(combined_df[original_score].head())\n",
    "print(\"-\" * 50)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c4fdbf66-6e67-4fcc-9f6a-71d9da986d16",
   "metadata": {},
   "source": [
    "## Numerical Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6bc0d12a-13fa-4b8f-8d29-cd44087dfca9",
   "metadata": {},
   "outputs": [],
   "source": [
    "with pd.option_context('display.float_format', '{:.2f}'.format):\n",
    "    print(combined_df[numerical_features].describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "33d23f8a-b305-45e5-89d5-8d6895a0de70",
   "metadata": {},
   "source": [
    "### Histogram"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "191296bb-c374-489f-89df-0ed0c5a16cc6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Histograms for numerical features - two per row\n",
    "plt.figure(figsize=(15, 8))\n",
    "for i, feature in enumerate(numerical_features, 1):\n",
    "    plt.subplot(2, len(numerical_features) // 2, i)\n",
    "    sns.histplot(combined_df[feature], kde=True)\n",
    "    plt.title(f'Histogram of {feature}')\n",
    "    plt.xlabel(feature)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f773ce61-496a-4bd2-9981-bb5b35405920",
   "metadata": {},
   "source": [
    "### Boxplot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "eaf77fc3-77e0-432d-8ad5-7743fbfc81c5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Boxplots to detect outliers\n",
    "plt.figure(figsize=(15, 8))\n",
    "for i, feature in enumerate(numerical_features, 1):\n",
    "    plt.subplot(2, len(numerical_features) //2, i)\n",
    "    sns.boxplot(y=combined_df[feature])\n",
    "    plt.title(f'Boxplot of {feature}')\n",
    "    plt.ylabel(feature)\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "06644ab3-f745-4660-b12d-93f6188b3ce8",
   "metadata": {},
   "source": [
    "### Scatter Plot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61779149-a180-4b59-abba-b160b3026abe",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create scatter plot matrix for numerical features\n",
    "sns.pairplot(combined_df[['year','runtimeMinutes', 'audienceScore', 'tomatoMeter']])\n",
    "plt.suptitle('Relationships Between Numerical Features', y=1.02)\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "22f905a8-1643-48a1-a9fe-c037dc2e9457",
   "metadata": {},
   "source": [
    "### Correlation Heatmap\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd9dfbb5-0f37-4a44-9d4c-c4f5d26c5008",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create correlation heatmap for numerical features\n",
    "plt.figure(figsize=(10, 8))\n",
    "numerical_data = combined_df[numerical_features].copy()\n",
    "\n",
    "# Calculate correlation matrix\n",
    "corr_matrix = numerical_data.corr()\n",
    "\n",
    "# Create heatmap\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')\n",
    "plt.title('Correlation Between Numerical Features')\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c17a7436-b68a-4af0-b04d-2a04d4ce6870",
   "metadata": {},
   "source": [
    "## Categorical Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e927d27-bb69-403b-b5a1-9e2251f57cb7",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Visualize categorical features\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "\n",
    "# List of categorical features\n",
    "categorical_features = [\n",
    "    'genre', \n",
    "    'director', \n",
    "    'originalLanguage', \n",
    "    'rating', \n",
    "    'scoreSentiment', \n",
    "    'reviewState'\n",
    "]\n",
    "\n",
    "# Set up the figure\n",
    "plt.figure(figsize=(15, 20))\n",
    "\n",
    "# Loop through each categorical feature\n",
    "for i, feature in enumerate(categorical_features, 1):\n",
    "    plt.subplot(3, 2, i)\n",
    "    \n",
    "    # Handle features with many unique values\n",
    "    if feature in ['director', 'genre']:\n",
    "        # Get top 10 most common values\n",
    "        value_counts = combined_df[feature].value_counts().head(10)\n",
    "    else:\n",
    "        # Get all values for features with fewer unique values\n",
    "        value_counts = combined_df[feature].value_counts()\n",
    "    \n",
    "    # Create the bar plot\n",
    "    sns.barplot(x=value_counts.index, y=value_counts.values)\n",
    "    \n",
    "    # Customize the plot\n",
    "    plt.title(f'Distribution of {feature}')\n",
    "    plt.xticks(rotation=45, ha='right')\n",
    "    plt.ylabel('Count')\n",
    "    \n",
    "    # Add value labels on top of bars\n",
    "    for j, v in enumerate(value_counts.values):\n",
    "        plt.text(j, v + 5, str(v), ha='center')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "823dde72-b061-422e-8aa8-62bd7bf3478b",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create contingency plots for related categorical features\n",
    "plt.figure(figsize=(12, 10))\n",
    "\n",
    "# 1. Genre vs. Score Sentiment\n",
    "plt.subplot(2, 1, 1)\n",
    "# Get top 5 genres\n",
    "top_genres = combined_df['genre'].value_counts().head(5).index\n",
    "genre_sentiment = pd.crosstab(\n",
    "    combined_df[combined_df['genre'].isin(top_genres)]['genre'],\n",
    "    combined_df[combined_df['genre'].isin(top_genres)]['scoreSentiment'],\n",
    "    normalize='index'\n",
    ")\n",
    "genre_sentiment.plot(kind='bar', stacked=True, ax=plt.gca())\n",
    "plt.title('Genre vs. Score Sentiment')\n",
    "plt.xlabel('Genre')\n",
    "plt.ylabel('Proportion')\n",
    "plt.legend(title='Sentiment')\n",
    "plt.xticks(rotation=30, ha='right')\n",
    "\n",
    "# 2. Original Language vs. Review State\n",
    "plt.subplot(2, 1, 2)\n",
    "# Get top 5 languages\n",
    "top_languages = combined_df['originalLanguage'].value_counts().head(5).index\n",
    "language_review = pd.crosstab(\n",
    "    combined_df[combined_df['originalLanguage'].isin(top_languages)]['originalLanguage'],\n",
    "    combined_df[combined_df['originalLanguage'].isin(top_languages)]['reviewState'],\n",
    "    normalize='index'\n",
    ")\n",
    "language_review.plot(kind='bar', stacked=True, ax=plt.gca())\n",
    "plt.title('Original Language vs. Review State')\n",
    "plt.xlabel('Language')\n",
    "plt.ylabel('Proportion')\n",
    "plt.legend(title='Review State')\n",
    "plt.xticks(rotation=30, ha='right')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a0f5bef-8761-4f6d-96b7-4cc19830b829",
   "metadata": {},
   "source": [
    "### Summarization of Categorical Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "826d87c2-d743-48fd-9081-e58c238051ad",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set figure size for better visibility\n",
    "plt.figure(figsize=(12, 15))\n",
    "\n",
    "# Define categorical features\n",
    "categorical_features = [\n",
    "    'genre', \n",
    "    'originalLanguage', \n",
    "    'scoreSentiment', \n",
    "    'reviewState'\n",
    "]\n",
    "\n",
    "# Create plots for categorical features\n",
    "for i, feature in enumerate(categorical_features):\n",
    "    plt.subplot(3, 2, i+1)\n",
    "    \n",
    "    if feature in ['genre', 'originalLanguage']:\n",
    "        # For features with many values, show top 10\n",
    "        counts = combined_df[feature].value_counts().head(10)\n",
    "        sns.barplot(x=counts.values, y=counts.index)\n",
    "        plt.title(f'Top 10 {feature}')\n",
    "        plt.xlabel('Count')\n",
    "    else:\n",
    "        # For features with fewer categories\n",
    "        sns.countplot(x=feature, data=combined_df)\n",
    "        plt.title(f'{feature} Distribution')\n",
    "        plt.xlabel(feature)\n",
    "\n",
    "# Add year distribution (though year is numerical)\n",
    "plt.subplot(3, 2, 5)\n",
    "sns.histplot(combined_df['year'], bins=30)\n",
    "plt.title('Movie Release Year Distribution')\n",
    "plt.xlabel('Year')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "28d69a02-1a32-43b7-9272-b4fe4edc8cee",
   "metadata": {},
   "source": [
    "### Cross Tabulation "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a6cc09d-d7b5-42e8-b88d-9e34017c1285",
   "metadata": {},
   "source": [
    "Cross-tabulation (cross-tab) is a way to organize data into a table to show the relationship between two or more categories. It helps compare groups by displaying counts or percentages"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4da2c2d5-3ae8-459a-a16b-bf59db1e8aef",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "\n",
    "# Create a figure for the cross-tabulations\n",
    "plt.figure(figsize=(16, 20))\n",
    "\n",
    "# 1. Genre vs. Average Audience Score\n",
    "plt.subplot(3, 2, 1)\n",
    "# Get top 10 genres by frequency\n",
    "top_genres = combined_df['genre'].value_counts().head(10).index\n",
    "# Calculate average audience score for each genre\n",
    "genre_ratings = combined_df[combined_df['genre'].isin(top_genres)].groupby('genre')['audienceScore'].mean().sort_values(ascending=False)\n",
    "sns.barplot(x=genre_ratings.values, y=genre_ratings.index)\n",
    "plt.title('Average Audience Score by Genre')\n",
    "plt.xlabel('Average Audience Score')\n",
    "plt.ylabel('Genre')\n",
    "\n",
    "# 2. Score Sentiment vs. Average Audience Score\n",
    "plt.subplot(3, 2, 2)\n",
    "sentiment_ratings = combined_df.groupby('scoreSentiment')['audienceScore'].mean()\n",
    "sns.barplot(x=sentiment_ratings.index, y=sentiment_ratings.values)\n",
    "plt.title('Average Audience Score by Review Sentiment')\n",
    "plt.xlabel('Sentiment')\n",
    "plt.ylabel('Average Audience Score')\n",
    "\n",
    "# 3. Review State vs. Average Tomato Meter\n",
    "plt.subplot(3, 2, 3)\n",
    "state_ratings = combined_df.groupby('reviewState')['tomatoMeter'].mean()\n",
    "sns.barplot(x=state_ratings.index, y=state_ratings.values)\n",
    "plt.title('Average Tomato Meter by Review State')\n",
    "plt.xlabel('Review State')\n",
    "plt.ylabel('Average Tomato Meter')\n",
    "\n",
    "# 4. Original Language vs. Average Audience Score\n",
    "plt.subplot(3, 2, 4)\n",
    "# Get top 10 languages by frequency\n",
    "top_languages = combined_df['originalLanguage'].value_counts().head(10).index\n",
    "language_ratings = combined_df[combined_df['originalLanguage'].isin(top_languages)].groupby('originalLanguage')['audienceScore'].mean().sort_values(ascending=False)\n",
    "sns.barplot(x=language_ratings.values, y=language_ratings.index)\n",
    "plt.title('Average Audience Score by Language')\n",
    "plt.xlabel('Average Audience Score')\n",
    "plt.ylabel('Language')\n",
    "\n",
    "# 5. Heatmap of average ratings by scoreSentiment and reviewState\n",
    "plt.subplot(3, 2, 5)\n",
    "heatmap_data = combined_df.pivot_table(\n",
    "    values='audienceScore', \n",
    "    index='scoreSentiment', \n",
    "    columns='reviewState', \n",
    "    aggfunc='mean'\n",
    ")\n",
    "sns.heatmap(heatmap_data, annot=True, cmap='YlGnBu', fmt='.1f')\n",
    "plt.title('Average Audience Score by Sentiment and Review State')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b1f2e225-125d-4741-bc62-d5a6deccc0a4",
   "metadata": {},
   "source": [
    "# DATA PREPROCESSING"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "11ba647c-4dd3-4a45-ad8c-597069cac072",
   "metadata": {},
   "source": [
    "## Check Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f6e4dfde-b880-422e-a7fe-dff512c69fa0",
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_values = combined_df.isnull().sum()\n",
    "print(missing_values)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "9a0ed326-cc69-4a32-af26-904435b66180",
   "metadata": {},
   "source": [
    "### 1. Remove Missing Values Rows "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c33c4e35-394c-4fee-9bd1-df26c4a45c48",
   "metadata": {},
   "source": [
    "### - Title"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9fc116f2-c815-4f68-aadf-fa4cd28de405",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Checks the missing titles\n",
    "missing_titles = combined_df[combined_df['title'].isnull()]\n",
    "\n",
    "print(missing_titles[['id', 'year', 'genre']])\n",
    "print(f\"Number of rows with missing titles: {len(missing_titles)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b67f3c61-1fb9-46d8-a823-113ad128255e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove rows with missing genre\n",
    "cleaned_df = combined_df.dropna(subset=['genre'])\n",
    "\n",
    "# Check how many rows were removed\n",
    "original_count = len(combined_df)\n",
    "new_count = len(cleaned_df)\n",
    "removed_count = original_count - new_count\n",
    "\n",
    "print(f\"Original dataset: {original_count} rows\")\n",
    "print(f\"After removing : {new_count} rows\")\n",
    "print(\"\")\n",
    "print(f\"{removed_count} rows have been deleted.\")\n",
    "print(\"-\" * 50)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a2d61d77-3cc2-40ab-9a14-57b45280a04f",
   "metadata": {},
   "outputs": [],
   "source": [
    "def extract_title_from_id(movie_id):\n",
    "    if pd.isna(movie_id):\n",
    "        return None\n",
    "    return \" \".join(word.capitalize() for word in movie_id.split(\"_\"))\n",
    "\n",
    "# Fill missing titles using extracted values\n",
    "cleaned_df.loc[cleaned_df['title'].isna(), 'title'] = cleaned_df.loc[cleaned_df['title'].isna(), 'id'].apply(extract_title_from_id)\n",
    "\n",
    "# Check for remaining missing values\n",
    "print(cleaned_df.isnull().sum())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "6c5acc31-2cfd-49ab-84a8-02bc74e51932",
   "metadata": {},
   "source": [
    "### Drop the column that is not affect very much in the recommender system"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3faef4ca-30c1-47e3-8e40-23f30c767304",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop Writer, Rating Values\n",
    "columns_drop = ['writer', 'rating']\n",
    "cleaned_df = cleaned_df.drop(columns=columns_drop)\n",
    "print(\"Dropped columns\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "53f5f458-078f-47f8-b8fc-3f2bb765c2b7",
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_values = cleaned_df.isnull().sum()\n",
    "print(missing_values)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0661acf6-c763-4fe8-a97e-97129fce5fca",
   "metadata": {},
   "source": [
    "## 2. Fill in the Missing Values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "533e7c67-3881-4176-85fc-49ca8852f9df",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill missing director values\n",
    "cleaned_df['director'] = cleaned_df['director'].fillna('Unknown')\n",
    "print(\"Filled missing director values with 'Unknown'\")\n",
    "\n",
    "# Fill missing originalLanguage values\n",
    "cleaned_df['originalLanguage'] = cleaned_df['originalLanguage'].fillna(cleaned_df['originalLanguage'].mode()[0])\n",
    "print(f\"Filled missing originalLanguage values with '{cleaned_df['originalLanguage'].mode()[0]}'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "475754c5-cd89-4841-b623-5682f4bbd7f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill missing runtimeMinutes values\n",
    "runtime_median = cleaned_df['runtimeMinutes'].median()\n",
    "cleaned_df['runtimeMinutes'] = cleaned_df['runtimeMinutes'].fillna(runtime_median)\n",
    "print(f\"Filled missing runtimeMinutes values with median: {runtime_median}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6b3a1dcb-62cf-4cce-b6b3-051bdc3d9ab0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 4. Audience Score - Fill with median \n",
    "cleaned_df['audienceScore'] = cleaned_df['audienceScore'].fillna(cleaned_df['audienceScore'].median())\n",
    "print(\"Filled missing audience scores with median\")\n",
    "\n",
    "# 5. Tomato Meter - Fill with median \n",
    "cleaned_df['tomatoMeter'] = cleaned_df['tomatoMeter'].fillna(cleaned_df['tomatoMeter'].median())\n",
    "print(\"Filled missing tomato meter scores with median\")\n",
    "\n",
    "# 6. Review Text - For NLP processing\n",
    "cleaned_df['review_text_flag'] = cleaned_df['reviewText'].notna().astype(int)\n",
    "cleaned_df['reviewText'] = cleaned_df['reviewText'].fillna(\"\")\n",
    "print(\"Created 'review_text_flag' flag and filled missing review text with empty strings\")\n",
    "\n",
    "# 7. original Score\n",
    "cleaned_df['original_score_flag'] = cleaned_df['originalScore'].notna().astype(int)\n",
    "cleaned_df['originalScore'] = cleaned_df['originalScore'].fillna(\"\")\n",
    "print(\"Created 'original_score_flag' flag and filled missing original score with empty strings\")\n",
    "\n",
    "# 8. Year\n",
    "cleaned_df['year_flag'] = cleaned_df['year'].notna().astype(int)\n",
    "cleaned_df['year'] = cleaned_df['year'].fillna(0)\n",
    "print(\"Created 'year_flag' flag and filled missing 'year' with 0\")\n",
    "\n",
    "# Create a flag to indicate missing values in \"movie_info\"\n",
    "cleaned_df['movie_info_flag'] = cleaned_df['movie_info'].notna().astype(int)\n",
    "cleaned_df['movie_info'] = cleaned_df['movie_info'].fillna(\"\")\n",
    "print(\"Created 'movie_info_flag' flag and filled missing movie info with empty strings\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b88673dd-fe55-4b6d-959c-2f2503d9095e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 7. Check remaining nulls\n",
    "remaining_nulls = cleaned_df.isnull().sum()\n",
    "print(\"\\nRemaining null values after cleaning:\")\n",
    "print(remaining_nulls)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7e5ce9c5-8e4a-4020-8901-c131ff9e98ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cb0d6767-9c6c-470d-8e40-fb933b47cecb",
   "metadata": {},
   "source": [
    "## 3. Adjust the Values\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "66cfe007-fc02-4275-9e28-99096bf43627",
   "metadata": {},
   "source": [
    "### A) OriginalLanguage"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9aa206fa-6c4f-48d4-b28f-e589d6f23bbd",
   "metadata": {},
   "outputs": [],
   "source": [
    "with pd.option_context('display.max_rows', None):\n",
    "    language_counts = cleaned_df['originalLanguage'].value_counts()\n",
    "    print(language_counts)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3e97a7eb-2caa-4b7b-aa6b-0b97a3737cdd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove anything in parentheses and strip trailing spaces\n",
    "cleaned_df['originalLanguage'] = cleaned_df['originalLanguage'].str.replace(r\"\\s*\\(.*?\\)\", \"\", regex=True).str.strip()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "56abb7db-20ed-458d-8bc2-919c258d3e6b",
   "metadata": {},
   "outputs": [],
   "source": [
    "invalid_languages = ['Unknown language', 'crp', 'shg', 'smi', 'nah']\n",
    "\n",
    "# Step 3: Remove invalid entries (i.e., keep valid ones)\n",
    "cleaned_df = cleaned_df[~cleaned_df['originalLanguage'].isin(invalid_languages)]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a6c66c13-99e2-4a4d-8df7-c69990be0b3d",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "42c8d301-1360-4e98-90d9-3e1b51935630",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f56c0bdf-ee40-4095-81eb-d7cb4ae271fe",
   "metadata": {},
   "source": [
    "### B) Genre"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58709a58-3b87-48e6-9440-13a6cfaee8a6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Count occurrences of each genre\n",
    "genre_counts = cleaned_df['genre'].value_counts().reset_index()\n",
    "genre_counts.columns = ['genre', 'count']\n",
    "\n",
    "# Display the results\n",
    "print(\"Total count for each genre:\")\n",
    "print(genre_counts)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "82521a50-251a-45d2-ae8d-a3c93c267d14",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove spaces at the front and back of the genre string and split by comma\n",
    "cleaned_df['genre_list'] = cleaned_df['genre'].str.split(',').apply(lambda x: [i.strip() for i in x])\n",
    "\n",
    "# Check the first few rows of the dataframe to verify\n",
    "print(cleaned_df[['genre', 'genre_list']].head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3940bb9a-a40b-4979-827b-f3081b7cb6d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Flatten the list and count unique genres\n",
    "unique_genres = set(genre for sublist in cleaned_df['genre_list'].dropna() for genre in sublist)\n",
    "\n",
    "# Print total count of unique genres\n",
    "print(f\"Total unique genres: {len(unique_genres)}\")\n",
    "print(\"List of unique genres:\", unique_genres)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "45f270ae-f29d-4e1c-99ff-6029905f8f41",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b131b3a4-c479-45c0-9041-879b26da55a7",
   "metadata": {},
   "source": [
    "### Mapping each Genre "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c124a33d-0eb4-4894-9537-8f803dff738c",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define genre mapping dictionary\n",
    "genre_mapping = {\n",
    "    'Sports': 'Sports',\n",
    "    'Mystery & thriller': 'Horror',\n",
    "    'Sci-fi': 'Sci-fi',\n",
    "    'Anime':'Anime',\n",
    "    'Comedy': 'Comedy',\n",
    "    'Documentary': 'Documentary',\n",
    "    'Nature': 'Nature',\n",
    "    'Drama': 'Drama',\n",
    "    'Action': 'Action',\n",
    "    'Music': 'Music',\n",
    "    'Crime': 'Crime',\n",
    "    'Reality': 'Reality',\n",
    "    'Health & wellness': 'Health & wellness',\n",
    "    'Horror': 'Horror',\n",
    "    'Variety': 'Variety Show',\n",
    "    'Biography': 'History',\n",
    "    'Kids & family': 'Kids & family',\n",
    "    'News': 'Reality',\n",
    "    'Faith & spirituality': 'Faith & spirituality',\n",
    "    'Western': 'Western',\n",
    "    'Musical': 'Music',\n",
    "    'History': 'History', \n",
    "    'Adventure': 'Adventure',\n",
    "    'Animation': 'Animation',\n",
    "    'Entertainment': 'Entertainment',\n",
    "    'Sports & fitness': 'Health & wellness',\n",
    "    'Romance': 'Romance',\n",
    "    'Lgbtq+': 'LGBTQ+',\n",
    "    'Stand-up': 'Comedy',\n",
    "    'War': 'War',\n",
    "    'Gay & lesbian': 'LGBTQ+',\n",
    "    'Fantasy': 'Fantasy',\n",
    "    'Short':'Other',\n",
    "    'Special interest':'Other',\n",
    "    'Holiday':'Other',\n",
    "    'Foreign':'Other',\n",
    "    'Other':'Other'\n",
    "}\n",
    "\n",
    "# Function to apply mapping to all genres\n",
    "def map_genres(genre_list):\n",
    "    # Apply genre mapping to each genre in the list\n",
    "    return [genre_mapping.get(genre, genre) for genre in genre_list]\n",
    "\n",
    "# Apply mapping to genre_list column\n",
    "cleaned_df['genre_list'] = cleaned_df['genre_list'].apply(map_genres)\n",
    "\n",
    "# Display the first few rows to check the mapping\n",
    "print(cleaned_df[['genre_list']].head())\n",
    "\n",
    "print(\"Genre mapping completed!\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d82cd30c-4ab9-470c-a4f2-ab10c8b785e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Find all genres that are still not mapped\n",
    "unmapped_genres = set(genre for sublist in cleaned_df['genre_list'] for genre in sublist if genre not in genre_mapping.values())\n",
    "\n",
    "# Count how many rows contain at least one unmapped genre\n",
    "unmapped_rows_count = cleaned_df['genre_list'].apply(lambda genres: any(g in unmapped_genres for g in genres)).sum()\n",
    "\n",
    "# Display results\n",
    "print(f\"Number of unique unmapped genres left: {len(unmapped_genres)}\")\n",
    "print(f\"Unique unmapped genres: {unmapped_genres}\")\n",
    "print(f\"Number of rows containing at least one unmapped genre: {unmapped_rows_count}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "68c08599-aee0-4332-bccd-95540182ef1d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Flatten the list and count unique genres\n",
    "unique_genres = set(genre for sublist in cleaned_df['genre_list'].dropna() for genre in sublist)\n",
    "\n",
    "# Print total count of unique genres\n",
    "print(f\"Total unique genres: {len(unique_genres)}\")\n",
    "print(\"List of unique genres:\", unique_genres)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c955a487-5397-4823-b37f-b00adb046cea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to check if a genre list has duplicates\n",
    "def has_duplicates(genre_list):\n",
    "    return len(genre_list) != len(set(genre_list))\n",
    "\n",
    "# Find rows where the genre_list contains duplicates\n",
    "duplicates_rows = cleaned_df[cleaned_df['genre_list'].apply(has_duplicates)]\n",
    "\n",
    "# Display the rows with duplicates\n",
    "print(\"Rows with duplicate genres in genre_list:\")\n",
    "print(duplicates_rows[['genre_list']])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4bb921de-81f0-433f-b082-eb65709e2651",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to remove duplicates from the genre list\n",
    "def remove_duplicates(genre_list):\n",
    "    # Convert to set to remove duplicates, then back to list\n",
    "    return list(set(genre_list))\n",
    "\n",
    "# Apply the function to clean the genre_list column\n",
    "cleaned_df['genre_list'] = cleaned_df['genre_list'].apply(remove_duplicates)\n",
    "\n",
    "# Display the first few rows to verify\n",
    "print(cleaned_df[['genre_list']].head())\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a174f6b4-40ad-4210-a882-65ac7f22f681",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Function to check if a genre list has duplicates\n",
    "def has_duplicates(genre_list):\n",
    "    return len(genre_list) != len(set(genre_list))\n",
    "\n",
    "# Find rows where the genre_list contains duplicates\n",
    "duplicates_rows = cleaned_df[cleaned_df['genre_list'].apply(has_duplicates)]\n",
    "\n",
    "# Display the rows with duplicates\n",
    "print(\"Rows with duplicate genres in genre_list:\")\n",
    "print(duplicates_rows[['genre_list']])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bec4c8f0-6446-4aec-9e56-d4ce2c0ad156",
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_values = cleaned_df.isnull().sum()\n",
    "print(missing_values)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3998961-289c-446f-9ef4-b6d908a5c180",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Display the first few rows\n",
    "cleaned_df.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7bde4953-070c-4640-9835-f7a188ea7ea7",
   "metadata": {},
   "source": [
    "## One-Hot Encoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d3d2c040-e97a-4558-a795-d6ce15c5a866",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import MultiLabelBinarizer\n",
    "\n",
    "# Create MultiLabelBinarizer instance\n",
    "mlb = MultiLabelBinarizer()\n",
    "\n",
    "# Apply MultiLabelBinarizer\n",
    "genre_encoded = pd.DataFrame(mlb.fit_transform(cleaned_df['genre_list']), columns=mlb.classes_)\n",
    "cleaned_df = cleaned_df.join(genre_encoded)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b3e0ee2-3dfc-4229-9d4e-f66d954de247",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get unique genres from the entire genre list\n",
    "all_genres = set(genre for sublist in cleaned_df['genre_list'] for genre in sublist)\n",
    "\n",
    "# Create a column for each genre and assign 1 if the genre is present for that movie, else 0\n",
    "for genre in all_genres:\n",
    "    cleaned_df[genre] = cleaned_df['genre_list'].apply(lambda x: 1 if genre in x else 0)\n",
    "\n",
    "# Display the updated dataframe\n",
    "cleaned_df[['title', 'genre_list'] + list(all_genres)]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "63b0ef8f-bfe4-464f-ac5b-4a6a491337d6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Set pandas to display all columns\n",
    "pd.set_option('display.max_columns', None)\n",
    "# Find the rows where title is 'Love and Leashes'\n",
    "love_and_leashes = cleaned_df[cleaned_df['title'] == 'Love and Leashes']\n",
    "\n",
    "# Display the result\n",
    "love_and_leashes[['title'] + list(all_genres)]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e94fe66-5a79-4894-a4e0-3cba864a7c20",
   "metadata": {},
   "outputs": [],
   "source": [
    "missing_values = cleaned_df.isnull().sum()\n",
    "print(missing_values)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "55c3fe34-1f53-44c3-8701-fd11a9e53c36",
   "metadata": {},
   "source": [
    "### C) OriginalScore"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e203e9a-c44c-43f8-8bb8-0953b970c4bd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Get counts of all unique original score values\n",
    "score_counts = cleaned_df['originalScore'].unique()\n",
    "# Display total number of unique scores\n",
    "print(f\"Total unique original score values: {len(score_counts)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "829a41bb-972d-4335-bcc8-d77ef894bc28",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Extract unique values from 'originalScore' column\n",
    "unique_scores = cleaned_df['originalScore'].dropna().unique()\n",
    "\n",
    "# Function to categorize score formats\n",
    "def categorize_score(score):\n",
    "    if not isinstance(score, str):\n",
    "        return \"non-string\"\n",
    "    \n",
    "    score = score.strip()  # Remove leading/trailing spaces\n",
    "\n",
    "    # Check for common formats\n",
    "    if '/' in score:  # Fraction format like \"3/5\" or \"7.5/10\"\n",
    "        return \"fraction\"\n",
    "    elif score.upper() in ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'D-', 'E', 'F', 'F-']:\n",
    "        return \"letter grade\"\n",
    "    elif score.endswith('%'):  \n",
    "        return \"percentage\"\n",
    "    elif score.replace('.', '', 1).isdigit():  # Whole numbers and decimals\n",
    "        return \"number\"\n",
    "    elif score.isalpha():  # Only alphabetic characters (no digits, no special characters)\n",
    "        return \"words\"\n",
    "    else:\n",
    "        return \"other\"\n",
    "\n",
    "# Apply categorization to unique values\n",
    "format_types = {}\n",
    "for score in unique_scores:\n",
    "    category = categorize_score(str(score))  \n",
    "    if category not in format_types:\n",
    "        format_types[category] = []\n",
    "    if len(format_types[category]) < 5:  # Store up to 20 examples per category\n",
    "        format_types[category].append(score)\n",
    "\n",
    "# Print format categories with examples\n",
    "print(\"\\nOriginalScore format categories with examples:\")\n",
    "for category, examples in format_types.items():\n",
    "    print(f\"{category}: {examples}\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9dfc99fc-de41-4442-bdc7-4cf8d6e6e68a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check for numeric scores \n",
    "def categorize_score(score):\n",
    "    if not isinstance(score, str):\n",
    "        return 'non-string'\n",
    "    \n",
    "    # Fractions\n",
    "    if '/' in score:\n",
    "        try:\n",
    "            num, denom = map(float, score.split('/'))\n",
    "            if denom == 0:\n",
    "                return 'Invalid Fraction'\n",
    "            elif num <= denom:\n",
    "                return 'Proper Fraction'\n",
    "            else:\n",
    "                return 'Improper Fraction'\n",
    "        except:\n",
    "            return 'Invalid Fraction'\n",
    "\n",
    "    # Try to extract numeric value\n",
    "    try:\n",
    "        numeric_value = float(score)  # Convert to float\n",
    "            \n",
    "        # Categorize by range\n",
    "        if numeric_value <= 0:\n",
    "            return 'Less than 0'\n",
    "        elif numeric_value <= 100:\n",
    "            return 'Between 5 And 100'\n",
    "        else:\n",
    "            return 'Greater than 100'\n",
    "    except:\n",
    "        return 'Non-Numeric'\n",
    "\n",
    "# Apply categorization to originalScore\n",
    "score_categories = cleaned_df['originalScore'].apply(categorize_score)\n",
    "\n",
    "# Count each category\n",
    "category_counts = score_categories.value_counts()\n",
    "\n",
    "# Display results\n",
    "print(\"Original Score Distribution:\")\n",
    "print(category_counts)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0549b130-95a6-437e-a3b9-4a182811f6cf",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5d5b535-8a7b-464f-8d6c-4b7cb6a30a60",
   "metadata": {},
   "outputs": [],
   "source": [
    "categories_remove = [\n",
    "    'Invalid Fraction','Improper Fraction','Less than 0','Greater than 100'\n",
    "]\n",
    "\n",
    "# Step 3: Filter out the rows where the category matches any in the list\n",
    "cleaned_df = cleaned_df[~score_categories.isin(categories_remove)].copy()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "f06b0568-1621-48b9-8d26-c8dce550225a",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "405f6812-7eb8-48a1-8d30-0bacedf2e77a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Preprocess 'originalScore' before standardization\n",
    "cleaned_df['originalScore'] = (\n",
    "    cleaned_df['originalScore']\n",
    "    .astype(str)  # Ensure all are strings\n",
    "    .str.replace(r\"['\\\"]\", \"\", regex=True)  # Remove single & double quotes\n",
    "    .str.replace(r'([\\W_])$', '', regex=True)  # Remove any trailing non-alphanumeric characters\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "886003c3-4575-4fab-9c1d-e46896fd9a98",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4092eee3-4440-46b7-b87d-b82aaa2e0cc0",
   "metadata": {},
   "source": [
    "## PART 1 : NUMERICAL VALUES"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "51300864-1b15-405e-af39-1c1bab8b95f8",
   "metadata": {},
   "source": [
    "### FRACTION / PERCENTANGE / NUMBERS"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc9add3f-af57-4f5f-909e-526290f2d9d3",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Standardize Numeric\n",
    "def numeric_standardization(score):\n",
    "    if not isinstance(score, str):\n",
    "        return None  # Ensure non-strings remain NaN\n",
    "\n",
    "    # 1) \"out of\" and \"out\" formats (\"3 OUT OF 5\",\"4 OUT 10\")\n",
    "    score = re.sub(r'(\\d+)\\s+OUT(?:\\s+OF)?\\s+(\\d+)', r'\\1/\\2', score, flags=re.IGNORECASE)\n",
    "\n",
    "    # 2) Remove duplicate symbols (\"4..5/5\")\n",
    "    score = re.sub(r'([./])\\1+', r'\\1', score)  \n",
    "\n",
    "    # 3) Convert fraction format (\"3/4\")\n",
    "    if '/' in score:\n",
    "        try:\n",
    "            num, denom = map(float, score.split('/'))\n",
    "            return (num / denom) * 100  # Convert fraction to percentage\n",
    "        except:\n",
    "            return None  # Invalid fraction\n",
    "\n",
    "    # 4) Convert percentage format (80%)\n",
    "    if score.endswith('%'):\n",
    "        try:\n",
    "            return float(score.replace('%', ''))\n",
    "        except:\n",
    "            return None  \n",
    "\n",
    "    # 5) Convert whole numbers & decimals\n",
    "    try:\n",
    "        num_value = float(score)\n",
    "        if num_value <= 5:\n",
    "            return (num_value / 5) * 100  # Scale values ≤5 with denominator 5\n",
    "        elif num_value <= 10:\n",
    "            return (num_value / 10) * 100  # Scale values ≤10 with denominator 10\n",
    "        elif num_value > 100:\n",
    "            return None  # Remove values greater than 100\n",
    "        return num_value  # Keep values ≤100 unchanged\n",
    "    except:\n",
    "        return None  # Non-numeric values remain NaN\n",
    "\n",
    "# Apply function to create the new column\n",
    "cleaned_df['standardized_score'] = cleaned_df['originalScore'].apply(numeric_standardization)\n",
    "\n",
    "# Display first few rows\n",
    "print(cleaned_df[['originalScore', 'standardized_score']].head())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "588f5d71-f1bb-4372-81fd-dc825bff1c78",
   "metadata": {},
   "source": [
    "# TEST"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "01e69947-dc96-42c8-81fd-7f57abe50be1",
   "metadata": {},
   "outputs": [],
   "source": [
    "filter_df = cleaned_df[cleaned_df['originalScore'] == \"3/4\"] # Test:3 out of 5, 4..5/5, 3/4, 80%,2\n",
    "filter_df = filter_df[['originalScore', 'standardized_score']]  \n",
    "\n",
    "print(filter_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "92801c9e-0b82-468c-9217-ada17982fbde",
   "metadata": {},
   "outputs": [],
   "source": [
    "with pd.option_context('display.float_format', '{:.2f}'.format):\n",
    "    print(cleaned_df[['tomatoMeter', 'audienceScore', 'standardized_score']].describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "62b96c0c-0157-437a-b96a-7715fd1eaaaf",
   "metadata": {},
   "source": [
    "## Part 2 : Star Ratings"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8b58b88c-c868-4621-be76-1e1cea4f4708",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Mapping words to Numbers\n",
    "word_to_number = {\n",
    "    \"ZERO\": 0, \"ONE\": 1, \"TWO\": 2, \"THREE\": 3, \"FOUR\": 4, \"FIVE\": 5\n",
    "}\n",
    "\n",
    "# Standardize star ratings\n",
    "def star_standardization(value):\n",
    "    if isinstance(value, str):\n",
    "        value = value.strip().upper()  # Uppercase and remove spaces\n",
    "\n",
    "        # Convert word-based numbers (\"FOUR STARS\" -> 4)\n",
    "        for word, num in word_to_number.items():\n",
    "            if value.startswith(word):\n",
    "                return (num / 5) * 100   # Standardize\n",
    "\n",
    "                    \n",
    "        # Handle \"X 1/2 out of 5 stars\" (e.g., \"4 1/2 out of 5 stars\" -> 4.5)\n",
    "        combined_fraction_match = re.search(r'(\\d+)\\s*1/2\\s+OUT OF\\s+5\\s*STARS?', value)\n",
    "        if combined_fraction_match:\n",
    "            whole_number = float(combined_fraction_match.group(1))  # Extracted number\n",
    "            decimal_value = whole_number + 0.5\n",
    "            return (decimal_value / 5) * 100\n",
    "\n",
    "        # Handle fraction \"X 1/2 stars\" (3 1/2 stars -> 3.5)\n",
    "        fraction_match = re.search(r'(\\d+)\\s*1/2\\s*STARS?', value, flags=re.IGNORECASE)\n",
    "        if fraction_match:\n",
    "            whole_number = float(fraction_match.group(1))  # Extract front part(e.g., 3)\n",
    "            decimal_value = whole_number + 0.5  # 3 + 0.5 = 3.5\n",
    "            return (decimal_value / 5) * 100   # Standardize\n",
    "\n",
    "        # Handle \"X out of 5 stars\" (4 out of 5 stars -> 4.0)\n",
    "        out_of_match = re.search(r'(\\d+(\\.\\d+)?)\\s+OUT OF\\s+5\\s*STARS?', value, flags=re.IGNORECASE)\n",
    "        if out_of_match:\n",
    "            score = float(out_of_match.group(1))\n",
    "            return (score / 5) * 100   # Standardize\n",
    "    \n",
    "        # Handle \"X out Y\" (3 out 5 → 3/5)\n",
    "        out_format_match = re.search(r'(\\d+(\\.\\d+)?)\\s+OUT\\s+(\\d+(\\.\\d+)?)', value, flags=re.IGNORECASE)\n",
    "        if out_format_match:\n",
    "            num = float(out_format_match.group(1))  # Numerator (3)\n",
    "            denom = float(out_format_match.group(3))  # Denominator (5)\n",
    "            return (num / denom) * 100  # Standardize\n",
    "\n",
    "\n",
    "        # Handle \"X/X stars\" (\"2.5/5 stars\" -> 2.5)\n",
    "        fraction_div_match = re.search(r'(\\d+(\\.\\d+)?)/5\\s*STARS?', value)\n",
    "        if fraction_div_match:\n",
    "            score = float(fraction_div_match.group(1))\n",
    "            return (score / 5) * 100  # Standardize\n",
    "\n",
    "        # Handle standard whole number or decimal format (e.g., \"3 stars\")\n",
    "        number_match = re.search(r'^(\\d+(\\.\\d+)?)\\s*STARS?', value)\n",
    "        if number_match:\n",
    "            score = float(number_match.group(1))\n",
    "            return (score / 5) * 100  # Standardize\n",
    "\n",
    "    return None  # Return None if not recognized\n",
    "\n",
    "# Apply function\n",
    "cleaned_df['standardized_score'] = cleaned_df['standardized_score'].combine_first(cleaned_df['originalScore'].apply(star_standardization))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7c8d75c8-a92f-4584-ba3f-5b0a543b0cb8",
   "metadata": {},
   "source": [
    "# TEST"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5447bfff-9400-4674-9dad-94d5c224e964",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Test :FOUR STARS, 4 1/2 out of 5 stars, 3 1/2 stars, 4 out of 5 stars, 3 out 5, 2.5/5 stars, 3 stars\n",
    "filtered_df = cleaned_df[cleaned_df['originalScore'] == '1 star'] \n",
    "\n",
    "print(filtered_df[['originalScore', 'standardized_score']])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "37df1a18-1b34-48e1-9259-e3a71677e1af",
   "metadata": {},
   "source": [
    "## Part 3 : Words"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1fb63d3e-6b90-4a03-b74c-883c1a921896",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter for scores that contain both letters and numbers\n",
    "mixed_scores = cleaned_df[cleaned_df['originalScore'].apply(\n",
    "    lambda x: isinstance(x, str) and \n",
    "              bool(re.search(r'[A-Za-z]', x)) and \n",
    "              bool(re.search(r'[0-9]', x))\n",
    ")]\n",
    "\n",
    "# Get counts of these mixed scores\n",
    "mixed_score_counts = mixed_scores['originalScore'].value_counts()\n",
    "\n",
    "# Display results\n",
    "print(f\"Found {len(mixed_score_counts)} unique mixed (words and numbers) original score values\")\n",
    "print(\"\\nMixed Original Score Values and Counts:\")\n",
    "print(mixed_score_counts)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55b35982-6cdd-4a1f-b871-3c21a77a4d16",
   "metadata": {},
   "outputs": [],
   "source": [
    "def words_standardization(value):\n",
    "    if isinstance(value, str):\n",
    "        value = value.strip().upper()\n",
    "        \n",
    "        # Handle \"+X out of -4..+4\" format(+1 out of -4..+4 )\n",
    "        out_of_4_match = re.search(r'([\\+\\-]?\\d+)\\s+OUT OF\\s+-4\\.\\.+\\+4', value)\n",
    "        if out_of_4_match:\n",
    "            score = int(out_of_4_match.group(1))  # Extract value(1)\n",
    "            return ((score + 4) / 8) * 100  # Standardize\n",
    "        \n",
    "        # Handle \"X out of Y\" format (3.5 out of 5)\n",
    "        out_of_y_match = re.search(r'(\\d+(\\.\\d+)?)\\s+OUT OF\\s+(\\d+)', value)\n",
    "        if out_of_y_match:\n",
    "            score = float(out_of_y_match.group(1))\n",
    "            max_score = float(out_of_y_match.group(3))\n",
    "            return (score / max_score) * 100  # Standardize\n",
    "\n",
    "        # Handle \"X of Y\" format (e.g., \"4 of 5\")\n",
    "        fraction_match = re.search(r'(\\d+)\\s+OF\\s+(\\d+)', value)\n",
    "        if fraction_match:\n",
    "            x = float(fraction_match.group(1))  # Extract numerator\n",
    "            y = float(fraction_match.group(2))  # Extract denominator\n",
    "            if y > 0:  # Prevent division by zero\n",
    "                percentage = (x / y) * 100  # Standardize\n",
    "                return round(percentage, 2)\n",
    "        \n",
    "    return None  # Return None if not recognized\n",
    "\n",
    "# Apply function\n",
    "cleaned_df['standardized_score'] = cleaned_df['standardized_score'].combine_first(cleaned_df['originalScore'].apply(words_standardization))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "cd67380e-1bc3-4332-8113-9eef7e6b8a0b",
   "metadata": {},
   "source": [
    "# TEST"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e924f9e6-ba7d-4226-a79d-cc473755ab8b",
   "metadata": {},
   "outputs": [],
   "source": [
    "filtered_df = cleaned_df[cleaned_df['originalScore'] == '4 of 5'] #Test : +1 out of -4..+4, 3.5 out of 5, 4 of 5\n",
    "print(filtered_df[['originalScore', 'standardized_score']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e9f7581-c3c7-4109-a9ca-c9630755b148",
   "metadata": {},
   "outputs": [],
   "source": [
    "nan_filtered_df = cleaned_df[cleaned_df['standardized_score'].isna()][['originalScore', 'standardized_score']]\n",
    "nan_filtered_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "339463a3-f247-484a-97b6-bf9825b17691",
   "metadata": {},
   "outputs": [],
   "source": [
    "#Function Empty Strings\n",
    "def empty_original_standardize(row):\n",
    "    if isinstance(row['originalScore'], str) and row['originalScore'].strip() == \"\":\n",
    "        return row['originalScore']  # Empty string\n",
    "    return row['standardized_score']  # Existing standardized score\n",
    "\n",
    "# Apply function\n",
    "cleaned_df['standardized_score'] = cleaned_df.apply(empty_original_standardize, axis=1)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fc3d7dac-e5e1-420f-a0cd-d9e609f2c85b",
   "metadata": {},
   "outputs": [],
   "source": [
    "nan_filtered_df = cleaned_df[cleaned_df['standardized_score'].isna()][['originalScore', 'standardized_score']]\n",
    "nan_filtered_df.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b25fa263-2635-405f-bf72-33cd93cb6bf7",
   "metadata": {},
   "source": [
    "## Part 4 : Grades"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ece65bca-d2d1-4a83-93a7-971a1c724565",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Dictionary mapping standard letter grades to scores\n",
    "letter_to_score = {\n",
    "    'A+': 100, 'A': 95, 'A-': 90,\n",
    "    'B+': 85, 'B': 80, 'B-': 75,\n",
    "    'C+': 70, 'C': 65, 'C-': 60,\n",
    "    'D+': 55, 'D': 50, 'D-': 45,\n",
    "    'E+': 40, 'E': 35, 'E-': 30,\n",
    "    'F+': 25, 'F': 20, 'F-': 15\n",
    "}\n",
    "\n",
    "# Function to clean and convert letter grades\n",
    "def letter_standardization(value):\n",
    "    if isinstance(value, str):\n",
    "        # Uppercase and strip spaces\n",
    "        value = value.strip().upper()\n",
    "\n",
    "        # Etract first letter (A-F)\n",
    "        match = re.match(r'([A-F])', value)\n",
    "        if match:\n",
    "            letter = match.group(1)  # Extract letter grade\n",
    "\n",
    "            # PLUS or MINUS\n",
    "            if \"PLUS\" in value:\n",
    "                letter += \"+\"\n",
    "            elif \"MINUS\" in value:\n",
    "                letter += \"-\"\n",
    "\n",
    "            # Get numeric score\n",
    "            return letter_to_score.get(letter, None)\n",
    "    \n",
    "    return None  # Return None if not a valid letter grade\n",
    "\n",
    "# Apply function to standardize letter grades\n",
    "cleaned_df['standardized_score'] = cleaned_df['standardized_score'].combine_first(cleaned_df['originalScore'].apply(letter_standardization))\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "74bb30a3-4d4a-4685-a363-96867ae93692",
   "metadata": {},
   "source": [
    "## TEST "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0fe38a23-731b-4da6-a4cd-affedccc9008",
   "metadata": {},
   "outputs": [],
   "source": [
    "filtered_df = cleaned_df[cleaned_df['originalScore'] == 'C minus'] # A-F, Cplus, C minus, C-minus\n",
    "print(filtered_df[['originalScore', 'standardized_score']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "bd36998c-20b9-4a28-a32a-8453f2ed7469",
   "metadata": {},
   "outputs": [],
   "source": [
    "nan_filtered_df = cleaned_df[cleaned_df['standardized_score'].isna()][['originalScore', 'standardized_score']]\n",
    "nan_filtered_df.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "61dc1d7e-9200-4504-ab0a-8abb0091c4a3",
   "metadata": {},
   "outputs": [],
   "source": [
    "nan_counts = cleaned_df[cleaned_df['standardized_score'].isna()] \\\n",
    "    .groupby('originalScore') \\\n",
    "    .size() \\\n",
    "    .reset_index(name='NaN Count') \\\n",
    "    .sort_values(by='NaN Count', ascending=False)\n",
    "\n",
    "print(nan_counts)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "12e2e8c7-10af-4ab6-acca-087a5ef6a63f",
   "metadata": {},
   "outputs": [],
   "source": [
    "recommendation_mapping = {\n",
    "    \"HIGHLY RECOMMENDED\": 95,       # Almost perfect recommendation\n",
    "    \"RECOMMENDED\": 75,             # A good but not perfect recommendation\n",
    "    \"NEUTRAL\": 50,                 # Neutral category, set it in the middle\n",
    "    \"NOT RECOMMENDED\": 25 ,        # A low score, but not zero\n",
    "    \"STRONGLY NOT RECOMMENDED\" :0  # Very bad review\n",
    "}\n",
    "\n",
    "def standardize_recommendation(value):\n",
    "    if isinstance(value, str):\n",
    "        value = value.strip().upper()  # Upper and remove spaces\n",
    "        return recommendation_mapping.get(value, None)  # Return mapped value or None\n",
    "    return None  # Return None if not a string\n",
    "\n",
    "# Apply function\n",
    "cleaned_df['standardized_score'] = cleaned_df['standardized_score'].combine_first(cleaned_df['originalScore'].apply(standardize_recommendation))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1555879b-1d1a-499e-9303-97adc7acf912",
   "metadata": {},
   "source": [
    "## Test "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9ca4bd8c-54c6-42ad-92ad-c1898b24b284",
   "metadata": {},
   "outputs": [],
   "source": [
    "filtered_df = cleaned_df[cleaned_df['originalScore'] == 'Recommended']# RECOMMENDED\n",
    "print(filtered_df[['originalScore', 'standardized_score']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "798e4ab7-cf99-427a-b4af-9c79abdf1a20",
   "metadata": {},
   "outputs": [],
   "source": [
    "nan_filtered_df = cleaned_df[cleaned_df['standardized_score'].isna()][['originalScore', 'standardized_score']]\n",
    "nan_filtered_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9c22c39-6986-439d-8044-72ba807d2fe4",
   "metadata": {},
   "outputs": [],
   "source": [
    "nan_counts = cleaned_df[cleaned_df['standardized_score'].isna()] \\\n",
    "    .groupby('originalScore') \\\n",
    "    .size() \\\n",
    "    .reset_index(name='NaN Count') \\\n",
    "    .sort_values(by='NaN Count', ascending=False)\n",
    "\n",
    "print(nan_counts)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "359a1d36-117c-4ebd-b2a1-48a56bf7432a",
   "metadata": {},
   "source": [
    "## Part 5 : Minor Adjustment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "58a076c5-ef21-4c59-a14e-8b60412310f9",
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_and_standardize_score(value):\n",
    "    if not isinstance(value, str):\n",
    "        return None  # Ensure non-strings remain NaN\n",
    "\n",
    "    value = value.strip()  # Remove leading/trailing spaces\n",
    "\n",
    "    # Fix misplaced spaces in numbers (e.g., \"2. 5 / 5\" -> \"2.5/5\")\n",
    "    value = re.sub(r'(\\d+)\\s*\\.\\s*(\\d+)', r'\\1.\\2', value)  # Fix \"2. 5\" -> \"2.5\"\n",
    "    value = re.sub(r'(\\d+)\\s*/\\s*(\\d+)', r'\\1/\\2', value)  # Fix \"5 / 5\" -> \"5/5\"\n",
    "\n",
    "    # Fix misplaced dots (e.g., \"3.5./4\" -> \"3.5/4\")\n",
    "    value = re.sub(r'(?<=\\d)\\.(?=/)', '', value)  \n",
    "\n",
    "    # Convert \"X.X.X\" to \"X.X/5\" (e.g., \"3.5.5\" -> \"3.5/5\")\n",
    "    value = re.sub(r'^(\\d+\\.\\d+)\\.(\\d+)$', r'\\1/\\2', value)\n",
    "\n",
    "    # Convert \"X/X\" fraction format to percentage\n",
    "    if '/' in value:\n",
    "        try:\n",
    "            num, denom = map(float, value.split('/'))\n",
    "            return (num / denom) * 100  # Convert to percentage\n",
    "        except:\n",
    "            return None  # Return NaN if conversion fails\n",
    "\n",
    "    return None  # If no valid format found, return NaN\n",
    "\n",
    "# Apply function to cleaned_df\n",
    "cleaned_df['standardized_score'] = cleaned_df['standardized_score'].combine_first(\n",
    "    cleaned_df['originalScore'].apply(clean_and_standardize_score)\n",
    ")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1e1675a1-a46f-4a80-8204-77d0fba80f76",
   "metadata": {},
   "outputs": [],
   "source": [
    "filtered_df = cleaned_df[cleaned_df['originalScore'] == \"3.5./4\"] #Test:2. 5 / 5, 3.5./4, 3.5.5\n",
    "print(filtered_df[['originalScore', 'standardized_score']])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "8512e39d-4240-40d2-b41e-077949495261",
   "metadata": {},
   "outputs": [],
   "source": [
    "nan_filtered_df = cleaned_df[cleaned_df['standardized_score'].isna()][['originalScore', 'standardized_score']]\n",
    "nan_filtered_df.head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "46873ffd-8bc6-443f-baab-0b0b0c3c965e",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Before remove NaN Values\n",
    "old_cleaned_df= len(cleaned_df)\n",
    "\n",
    "# Remove NaN values in 'standardized_score'\n",
    "cleaned_df = cleaned_df.dropna(subset=['standardized_score'])\n",
    "\n",
    "# After remove\n",
    "new_cleaned_df = len(cleaned_df)\n",
    "\n",
    "# Print the number of deleted records\n",
    "deleted_num= old_cleaned_df - new_cleaned_df\n",
    "\n",
    "print(f\"{deleted_num} records have been deleted.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "920b67cb-508a-4682-86e3-094e8390f8c6",
   "metadata": {},
   "outputs": [],
   "source": [
    "print(old_cleaned_df)\n",
    "print(new_cleaned_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "27052c68-cbad-46f4-9af3-f42ac54f827a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert 'standardized_score' to float\n",
    "cleaned_df['standardized_score'] = pd.to_numeric(cleaned_df['standardized_score'], errors='coerce')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5e7a8a2f-686f-4e43-b9d5-5cdf0f32c100",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Count the number of values greater than 100\n",
    "greater_100 = (cleaned_df['standardized_score'] > 100).sum()\n",
    "lesser_0 = (cleaned_df['standardized_score']< 0).sum()\n",
    "\n",
    "# Display the result\n",
    "print(\"Number of values greater than 100:\", greater_100)\n",
    "print(\"Number of values greater than 100:\", lesser_0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b4c09cf9-98b6-4d67-80ad-87029528dc04",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Remove rows with values greater than 100 or less than 0\n",
    "cleaned_df = cleaned_df[(cleaned_df['standardized_score'] <= 100) & (cleaned_df['standardized_score'] >= 0)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "359e9a53-ae06-48a0-8bb9-e7068f75bfb1",
   "metadata": {},
   "outputs": [],
   "source": [
    "with pd.option_context('display.float_format', '{:.2f}'.format):\n",
    "    print(cleaned_df['standardized_score'].describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f51e139-4e9f-4e67-84e7-167cbc59994e",
   "metadata": {},
   "source": [
    "# FEATURE ENGINEERING"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b43bb01d-0c67-4d15-b13c-eb6e25332133",
   "metadata": {},
   "outputs": [],
   "source": [
    "with pd.option_context('display.float_format', '{:.2f}'.format):\n",
    "    print(cleaned_df[['audienceScore','tomatoMeter','standardized_score']].describe())"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3f740178-61ae-4fd8-b4f8-5dc8bf1ff2f5",
   "metadata": {},
   "source": [
    "Since the values for these three attributes is from range 0-100, so that no need do Normalization(Min-Max Scaler) and Standardization(Z score)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e21fabfa-23fb-452a-8aa1-8894e1332ff3",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "55b8d2b8-0c8f-491d-9b90-9947f69bc094",
   "metadata": {},
   "outputs": [],
   "source": [
    "cleaned_df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e216ed2f-d182-4dff-8877-f307ea81cbd4",
   "metadata": {},
   "source": [
    "## Label Encoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
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
   "execution_count": null,
   "id": "f9e26922-855b-41a3-8161-f723eb2377a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "encoded_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
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
   "execution_count": null,
   "id": "b7e1b8c0-7488-4005-9c05-688a335d1f8d",
   "metadata": {},
   "outputs": [],
   "source": [
    "final_encoded_df_cleaned.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
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
   "execution_count": null,
   "id": "6773bf04-a030-4847-b287-cd3da376120b",
   "metadata": {},
   "outputs": [],
   "source": [
    "content_df = df.copy(deep=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1b2285b7-7d0d-45b3-ad48-ab20d9ef1ee6",
   "metadata": {},
   "outputs": [],
   "source": [
    "columns_remove = ['audienceScore','userName', 'reviewId', 'reviewText','scoreSentiment', 'reviewState','review_text_flag', 'year_flag',\n",
    "    'standardized_score'] \n",
    "content_df = content_df.drop(columns=columns_remove)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc0c026b-7e13-450d-8ff5-95ff58518abf",
   "metadata": {},
   "outputs": [],
   "source": [
    "content_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
   "id": "6f6a0bf9-0287-42ec-9726-0d01580e9e46",
   "metadata": {},
   "outputs": [],
   "source": [
    "content_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
   "id": "ae1d6bcf-7109-4d87-8c13-28a56a54fe5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "collaborative_df = df.copy(deep=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "83c47368-f18e-4a5b-8cff-6e40c80fd05a",
   "metadata": {},
   "outputs": [],
   "source": [
    "collaborative_df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "34ffbd1d-4e8a-46be-8b35-ee2f00b80959",
   "metadata": {},
   "outputs": [],
   "source": [
    "genre_columns = [\n",
    "    'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime',\n",
    "    'Documentary', 'Drama', 'Entertainment', 'Faith & spirituality',\n",
    "    'Fantasy', 'Health & wellness', 'History', 'Horror', 'Kids & family',\n",
    "    'LGBTQ+', 'Music', 'Nature', 'Other', 'Reality', 'Romance', 'Sci-fi', 'Sports',\n",
    "    'Variety Show', 'War', 'Western'\n",
    "]\n",
    "\n",
    "columns_remove = ['movie_info', 'year', 'director','originalLanguage', 'runtimeMinutes','reviewId','reviewText',\n",
    "                  'scoreSentiment','reviewState','review_text_flag', 'year_flag', 'movie_info_flag'] + genre_columns \n",
    "collaborative_df = collaborative_df.drop(columns=columns_remove)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
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
   "execution_count": null,
   "id": "4f05b23d-a785-4d85-9c25-39aab7c11490",
   "metadata": {},
   "outputs": [],
   "source": [
    "nlp_df = content_df.copy(deep=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
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
   "execution_count": null,
   "id": "aedcd193-fc71-4a3b-91a6-3570c4aeff30",
   "metadata": {},
   "outputs": [],
   "source": [
    "content_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "df9c2181-d82e-43b8-b1d5-9463d6aeca73",
   "metadata": {},
   "outputs": [],
   "source": [
    "content_df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
   "id": "4c6c9a64-ff74-4f4b-a1cd-2ea5a4da9df2",
   "metadata": {},
   "outputs": [],
   "source": [
    "feature_cols = [\n",
    "    'runtimeMinutes', 'director', 'originalLanguage',\n",
    "    'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime',\n",
    "    'Documentary', 'Drama', 'Entertainment', 'Faith & spirituality',\n",
    "    'Fantasy', 'Health & wellness', 'History', 'Horror', 'Kids & family',\n",
    "    'LGBTQ+', 'Music', 'Nature','Other', 'Reality', 'Romance', 'Sci-fi', 'Sports',\n",
    "    'Variety Show', 'War', 'Western'\n",
    "]\n",
    "\n",
    "content_features = content_df[feature_cols]\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2f01d0fd-acf3-45b8-bf13-973d03db1b55",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "def sort_by_tomatoMeter(df, similarities, top_n=5):\n",
    "    matched = [(idx, df.iloc[idx], similarities[idx]) for idx in similarities.argsort()[::-1]]\n",
    "    matched.sort(key=lambda x: float(x[1].get('tomatoMeter', 0) or 0), reverse=True)\n",
    "    return [idx for idx, _, _ in matched[:top_n]]\n",
    "\n",
    "# Recommender function\n",
    "def content_recommender(user_input, df, feature_df, top_n=5, input_type='title'):\n",
    "    df = df.reset_index(drop=True)\n",
    "    feature_df = feature_df.reset_index(drop=True)\n",
    "\n",
    "    explanation = {\n",
    "        'movie_title': user_input,\n",
    "        'movie_details': None,\n",
    "        'recommendations': [],\n",
    "        'input_type': input_type,\n",
    "        'user_input': user_input\n",
    "    }\n",
    "\n",
    "    if input_type == 'title':\n",
    "        try:\n",
    "            movie_idx = df[df['title'].str.lower() == user_input.lower()].index[0]\n",
    "        except IndexError:\n",
    "            return f\"Movie '{user_input}' not found.\"\n",
    "\n",
    "        selected_movie = df.iloc[movie_idx]\n",
    "        target_vector = feature_df.iloc[movie_idx].values.reshape(1, -1)\n",
    "        all_vectors = feature_df.values\n",
    "        similarities = cosine_similarity(target_vector, all_vectors)[0]\n",
    "\n",
    "        selected_director = selected_movie['director']\n",
    "        selected_language = selected_movie['originalLanguage']\n",
    "        selected_genres = [col for col in feature_df.columns if selected_movie.get(col, 0) == 1]\n",
    "\n",
    "        for idx in range(len(similarities)):\n",
    "            movie = df.iloc[idx]\n",
    "            if idx == movie_idx:\n",
    "                similarities[idx] = -1  # Exclude the same movie\n",
    "                continue\n",
    "            if movie['director'] == selected_director:\n",
    "                similarities[idx] += 0.1\n",
    "            movie_genres = [col for col in feature_df.columns if movie.get(col, 0) == 1]\n",
    "            genre_overlap = len(set(selected_genres).intersection(set(movie_genres)))\n",
    "            similarities[idx] += 0.05 * genre_overlap\n",
    "            if movie['originalLanguage'] == selected_language:\n",
    "                similarities[idx] += 0.03\n",
    "\n",
    "        top_indices = similarities.argsort()[::-1][:top_n]\n",
    "\n",
    "        explanation['movie_details'] = {\n",
    "            'title': selected_movie['title'],\n",
    "            'genres': selected_genres,\n",
    "            'director': director_decoder.get(selected_director, 'Unknown'),\n",
    "            'original_language': language_decoder.get(selected_language, 'Unknown'),\n",
    "            'runtime_minutes': selected_movie['runtimeMinutes'],\n",
    "            'tomatoMeter': selected_movie.get('tomatoMeter', 'N/A'),\n",
    "            'year': selected_movie.get('year', 'N/A')\n",
    "        }\n",
    "\n",
    "        explanation['reasoning'] = (\n",
    "            f\"Since you liked '{selected_movie['title']}', we're recommending movies that share similar \"\n",
    "            f\"genres ({', '.join(selected_genres)}), the same director \"\n",
    "            f\"({director_decoder.get(selected_director, 'Unknown')}), or are in the same language \"\n",
    "            f\"({language_decoder.get(selected_language, 'Unknown')}).\"\n",
    "        )\n",
    "\n",
    "    elif input_type == 'genre':\n",
    "        if user_input not in feature_df.columns:\n",
    "            return f\"Genre '{user_input}' not found.\"\n",
    "        \n",
    "        # Filter to include only movies with this genre\n",
    "        matching_indices = feature_df[feature_df[user_input] == 1].index\n",
    "        if len(matching_indices) == 0:\n",
    "            return f\"No movies found with genre '{user_input}'.\"\n",
    "        \n",
    "        # Use the filtered dataframes\n",
    "        selected_df = df.loc[matching_indices].reset_index(drop=True)\n",
    "        selected_features = feature_df.loc[matching_indices].reset_index(drop=True)\n",
    "        \n",
    "        target_vector = selected_features.mean().values.reshape(1, -1)\n",
    "        all_vectors = selected_features.values\n",
    "        similarities = cosine_similarity(target_vector, all_vectors)[0]\n",
    "        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)\n",
    "        \n",
    "        df = selected_df\n",
    "        feature_df = selected_features\n",
    "\n",
    "    elif input_type == 'language':\n",
    "        rev_language_decoder = {v.lower(): k for k, v in language_decoder.items()}\n",
    "        lang_code = rev_language_decoder.get(user_input.lower())\n",
    "        if lang_code is None:\n",
    "            return f\"Language '{user_input}' not found.\"\n",
    "\n",
    "        matching_indices = df[df['originalLanguage'] == lang_code].index\n",
    "        if matching_indices.empty:\n",
    "            return f\"No movies found for language '{user_input}'.\"\n",
    "\n",
    "        target_vector = feature_df.loc[matching_indices].mean().values.reshape(1, -1)\n",
    "        selected_df = df.loc[matching_indices]\n",
    "        selected_features = feature_df.loc[matching_indices]\n",
    "        all_vectors = selected_features.values\n",
    "        similarities = cosine_similarity(target_vector, all_vectors)[0]\n",
    "        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)\n",
    "\n",
    "    elif input_type == 'year':\n",
    "        try:\n",
    "            year = int(user_input)\n",
    "        except ValueError:\n",
    "            return \"Invalid year format. Please enter a number.\"\n",
    "        matching_indices = df[df['year'] == year].index\n",
    "        if matching_indices.empty:\n",
    "            return f\"No movies found for year '{user_input}'.\"\n",
    "\n",
    "        target_vector = feature_df.loc[matching_indices].mean().values.reshape(1, -1)\n",
    "        selected_df = df.loc[matching_indices]\n",
    "        selected_features = feature_df.loc[matching_indices]\n",
    "        all_vectors = selected_features.values\n",
    "        similarities = cosine_similarity(target_vector, all_vectors)[0]\n",
    "        top_indices = sort_by_tomatoMeter(selected_df, similarities, top_n)\n",
    "\n",
    "    else:\n",
    "        return f\"Unsupported input type: {input_type}\"\n",
    "\n",
    "    for idx in top_indices:\n",
    "        movie_data = df.iloc[idx] if input_type == 'title' else (selected_df.iloc[idx] if input_type in ['language', 'year'] else df.iloc[idx])\n",
    "        similarity_score = similarities[idx]\n",
    "        genres = [\n",
    "            col for col in feature_df.columns\n",
    "            if col not in ['runtimeMinutes', 'director', 'originalLanguage'] and movie_data.get(col, 0) == 1\n",
    "        ]\n",
    "        director = director_decoder.get(movie_data['director'], 'Unknown')\n",
    "        language = language_decoder.get(movie_data['originalLanguage'], 'Unknown')\n",
    "\n",
    "        movie_features = {\n",
    "            'title': movie_data['title'],\n",
    "            'similarity_score': similarity_score,\n",
    "            'genres': genres,\n",
    "            'year': movie_data.get('year', 'N/A'),\n",
    "            'director': director,\n",
    "            'original_language': language,\n",
    "            'runtime_minutes': movie_data['runtimeMinutes'],\n",
    "            'tomatoMeter': movie_data.get('tomatoMeter', 'N/A')\n",
    "        }\n",
    "\n",
    "        explanation['recommendations'].append(movie_features)\n",
    "\n",
    "    return explanation\n",
    "\n",
    "# Evaluation function for Precision\n",
    "def evaluate_precision(result, input_type, user_input):\n",
    "    if isinstance(result, str):\n",
    "        return None\n",
    "    recommendations = result['recommendations']\n",
    "    top_n = len(recommendations)\n",
    "    if input_type == 'genre':\n",
    "        return sum([1 if user_input in rec['genres'] else 0 for rec in recommendations]) / top_n\n",
    "    elif input_type == 'language':\n",
    "        return sum([1 if rec['original_language'].lower() == user_input.lower() else 0 for rec in recommendations]) / top_n\n",
    "    elif input_type == 'year':\n",
    "        try:\n",
    "            year = int(user_input)\n",
    "            return sum([1 if rec['year'] == year else 0 for rec in recommendations]) / top_n\n",
    "        except ValueError:\n",
    "            return None\n",
    "    return None\n",
    "\n",
    "def show_recommendations(result, df):\n",
    "    if isinstance(result, str):\n",
    "        print(result)\n",
    "    else:\n",
    "        if result['movie_details']:\n",
    "            details = result['movie_details']\n",
    "            print(f\"\\nSelected Movie: {details['title']}\")\n",
    "            print(f\"   Genres: {', '.join(details['genres'])}\")\n",
    "            print(f\"   Year: {(details['year'])}\")\n",
    "            print(f\"   Director: {details['director']}\")\n",
    "            print(f\"   Original Language: {details['original_language']}\")\n",
    "            print(f\"   Runtime: {details['runtime_minutes']} minutes\")\n",
    "            print(\"----------------------------------------------------\")\n",
    "\n",
    "        print(f\"\\nRecommendations for '{result['movie_title']}':\")\n",
    "        for idx, recommendation in enumerate(result['recommendations'], 1):\n",
    "            print(f\"{idx}. Movie: {recommendation['title']}\")\n",
    "            print(f\"   Genres: {', '.join(recommendation['genres'])}\")\n",
    "            print(f\"   Year: {recommendation['year']}\")\n",
    "            print(f\"   Director: {recommendation['director']}\")\n",
    "            print(f\"   Language: {recommendation['original_language']}\")\n",
    "            print(f\"   Runtime: {recommendation['runtime_minutes']} minutes\")\n",
    "            print(f\"   TomatoMeter: {recommendation['tomatoMeter']}%\")\n",
    "            print(\"---\")\n",
    "\n",
    "        input_type = result['input_type']\n",
    "        user_input = result['user_input']\n",
    "\n",
    "        if input_type in ['genre', 'language', 'year']:\n",
    "            precision_val = evaluate_precision(result, input_type, user_input)\n",
    "\n",
    "            print(\"\\n--- Evaluation Metrics ---\")\n",
    "            if precision_val is not None:\n",
    "                print(f\"Precision@{len(result['recommendations'])}: {precision_val:.2f}\")\n",
    "\n",
    "def menu(df, feature_df):\n",
    "    print(\"Choose your input type:\")\n",
    "    print(\"1. Movie Title\")\n",
    "    print(\"2. Genre\")\n",
    "    print(\"3. Language\")\n",
    "    print(\"4. Release Year\")\n",
    "\n",
    "    choice = input(\"Enter your choice (1-4): \")\n",
    "\n",
    "    input_type_map = {\n",
    "        '1': 'title',\n",
    "        '2': 'genre',\n",
    "        '3': 'language',\n",
    "        '4': 'year'\n",
    "    }\n",
    "\n",
    "    if choice not in input_type_map:\n",
    "        print(\"Invalid choice.\")\n",
    "        return\n",
    "\n",
    "    input_type = input_type_map[choice]\n",
    "\n",
    "    if input_type == 'genre':\n",
    "        available_genres = [col for col in feature_df.columns if col not in ['runtimeMinutes', 'director', 'originalLanguage']]\n",
    "        print(\"\\nAvailable genres:\")\n",
    "        for i, genre in enumerate(available_genres, 1):\n",
    "            print(f\"{i}. {genre}\")\n",
    "        genre_index = int(input(\"\\nEnter genre number: \")) - 1\n",
    "        if genre_index < 0 or genre_index >= len(available_genres):\n",
    "            print(\"Invalid genre selection.\")\n",
    "            return\n",
    "        user_input = available_genres[genre_index]\n",
    "\n",
    "    elif input_type == 'language':\n",
    "        languages = sorted(set(language_decoder.values()))\n",
    "        print(\"\\nAvailable languages:\")\n",
    "        for i, lang in enumerate(languages, 1):\n",
    "            print(f\"{i}. {lang}\")\n",
    "        lang_index = int(input(\"\\nEnter language number: \")) - 1\n",
    "        if lang_index < 0 or lang_index >= len(languages):\n",
    "            print(\"Invalid language selection.\")\n",
    "            return\n",
    "        user_input = languages[lang_index]\n",
    "\n",
    "    elif input_type == 'year':\n",
    "        user_input = input(\"Enter the release year (e.g., 2020): \")\n",
    "        try:\n",
    "            year = int(user_input)\n",
    "        except ValueError:\n",
    "            print(\"Invalid year format. Please enter a valid year (e.g., 2020).\")\n",
    "            return\n",
    "        if df[df['year'] == year].empty:\n",
    "            print(f\"No movies found for the year '{year}'.\")\n",
    "            return\n",
    "\n",
    "    else:\n",
    "        user_input = input(f\"Enter {input_type}: \")\n",
    "\n",
    "    result = content_recommender(user_input, df, feature_df, top_n=5, input_type=input_type)\n",
    "    show_recommendations(result, df)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
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
    "    min_year, max_year = int(year_range[0]), int(year_range[1])\n",
    "elif len(year_range) == 1:\n",
    "    min_year = max_year = int(year_range[0])\n",
    "\n",
    "# Extract all known languages from decoder\n",
    "all_languages = [v.lower() for v in language_decoder.values()]\n",
    "for lang in all_languages:\n",
    "    if lang in user_query.lower():\n",
    "        query_language = lang\n",
    "        break\n",
    "\n",
    "# Genre extraction from list of known genres\n",
    "genre_columns = [\n",
    "    'Action', 'Adventure', 'Animation', 'Anime', 'Comedy', 'Crime', 'Documentary',\n",
    "    'Drama', 'Entertainment', 'Faith & spirituality', 'Fantasy', 'Health & wellness',\n",
    "    'History', 'Horror', 'Kids & family', 'LGBTQ+', 'Music', 'Nature', 'Other',\n",
    "    'Reality', 'Romance', 'Sci-fi', 'Sports', 'Variety Show', 'War', 'Western'\n",
    "]\n",
    "for genre in genre_columns:\n",
    "    if genre.lower() in user_query.lower():\n",
    "        query_genre = genre\n",
    "        break\n",
    "\n",
    "# Extract tomatoMeter threshold\n",
    "score_match = re.search(r'tomato.*?(\\d+)', user_query.lower())\n",
    "if score_match:\n",
    "    min_score = int(score_match.group(1))\n",
    "\n",
    "# Extract full director name\n",
    "director_match = re.search(r'director(?: is| named)? ([A-Za-z ]+)', user_query.lower())\n",
    "if director_match:\n",
    "    director_name = director_match.group(1).strip().lower()\n",
    "\n",
    "# clean query for better TF-IDF\n",
    "clean_query = user_query\n",
    "for lang in all_languages:\n",
    "    clean_query = clean_query.replace(lang, '')\n",
    "if query_genre:\n",
    "    clean_query = clean_query.replace(query_genre, '')\n",
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
    "    tokens = nltk.word_tokenize(text.lower())\n",
    "    vectors = [model[word] for word in tokens if word in model]\n",
    "    return np.mean(vectors, axis=0) if vectors else np.zeros(vector_size)\n",
    "\n",
    "content_df['w2v_vector'] = content_df['movie_info'].apply(lambda x: get_average_vector(x, w2v_model))\n",
    "query_w2v_vector = get_average_vector(user_query, w2v_model)\n",
    "\n",
    "w2v_similarities = content_df['w2v_vector'].apply(\n",
    "    lambda x: cosine_similarity([x], [query_w2v_vector])[0][0]\n",
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
    "    filtered_df = filtered_df[filtered_df['year'].between(min_year, max_year if max_year else min_year)]\n",
    "\n",
    "# Language filter\n",
    "if query_language:\n",
    "    matching_lang_ids = [k for k, v in language_decoder.items() if v.lower() == query_language]\n",
    "    filtered_df = filtered_df[filtered_df['originalLanguage'].isin(matching_lang_ids)]\n",
    "\n",
    "# Genre filter\n",
    "if query_genre:\n",
    "    filtered_df = filtered_df[filtered_df[query_genre] == 1]\n",
    "\n",
    "# Director filter\n",
    "if director_name:\n",
    "    matching_director_ids = [\n",
    "        k for k, v in director_decoder.items() if director_name in v.lower()\n",
    "    ]\n",
    "    if matching_director_ids:\n",
    "        filtered_df = filtered_df[filtered_df['director'].isin(matching_director_ids)]\n",
    "\n",
    "# Rotten Tomatoes score filter\n",
    "if min_score is not None:\n",
    "    filtered_df = filtered_df[filtered_df['tomatoMeter'] >= min_score]\n",
    "\n",
    "# --- Final result ---\n",
    "top_recommendations = filtered_df.sort_values(by='similarity', ascending=False).head(10)\n",
    "\n",
    "# --- Display results ---\n",
    "print(\"\\nHere are 10 movies that match your description:\\n\")\n",
    "for i, (_, row) in enumerate(top_recommendations.iterrows(), 1):\n",
    "    genres = [genre for genre in genre_columns if row.get(genre, 0) == 1]\n",
    "    genre_str = \", \".join(genres) if genres else \"N/A\"\n",
    "    language = language_decoder.get(row['originalLanguage'], 'Unknown')\n",
    "    director = director_decoder.get(row['director'], 'Unknown')\n",
    "    runtime = f\"{row['runtimeMinutes']} minutes\" if not pd.isna(row.get('runtimeMinutes')) else \"N/A\"\n",
    "    tomato_score = f\"{row['tomatoMeter']}%\" if not pd.isna(row.get('tomatoMeter')) else \"N/A\"\n",
    "\n",
    "    print(f\"{i}. Movie: {row['title']}\")\n",
    "    print(f\"   Genres: {genre_str}\")\n",
    "    print(f\"   Year: {int(row['year']) if not pd.isna(row['year']) else 'N/A'}\")\n",
    "    print(f\"   Director: {director}\")\n",
    "    print(f\"   Language: {language}\")\n",
    "    print(f\"   Runtime: {runtime}\")\n",
    "    print(f\"   TomatoMeter: {tomato_score}\")\n",
    "    print()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
   "id": "32d7e081-6bfe-4929-8d20-11f6d0268fbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "collaborative_df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7ecc59c8-8b34-4b29-a3c1-6658dd69a6f8",
   "metadata": {},
   "outputs": [],
   "source": [
    "collaborative_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
   "id": "e746dae1-8be4-491e-a485-b8638cad3cf9",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_movies_collab_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
   "execution_count": null,
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
   "execution_count": null,
   "id": "af9b05b3-aed4-4136-a505-c3a1d9c31571",
   "metadata": {},
   "outputs": [],
   "source": [
    "top_movies_collab_df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "7d1b8685-3ecf-425e-b989-0245be0e0908",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.decomposition import TruncatedSVD\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "import numpy as np\n",
    "\n",
    "def collaborative_menu(top_movies_collab_df, top_n=10):\n",
    "    # Step 1: Create the user-item matrix\n",
    "    rating_matrix = top_movies_collab_df.pivot(index='userName', columns='title', values='standardized_score')\n",
    "    \n",
    "    # Step 2: Fill missing values with 0\n",
    "    rating_matrix = rating_matrix.fillna(0)\n",
    "    \n",
    "    # Step 3: Apply SVD for dimensionality reduction\n",
    "    svd = TruncatedSVD(n_components=20, random_state=42)\n",
    "    matrix_svd = svd.fit_transform(rating_matrix)\n",
    "    \n",
    "    # Step 4: Compute cosine similarity between users\n",
    "    similarity_matrix = cosine_similarity(matrix_svd)\n",
    "    \n",
    "    # Step 5: Prompt for user input\n",
    "    user_input = input(\"Please enter your userName: \").strip()\n",
    "    \n",
    "    if user_input:\n",
    "        try:\n",
    "            user_id = int(user_input)\n",
    "        except ValueError:\n",
    "            print(\"Invalid input. Please enter a valid numeric user ID.\")\n",
    "            return\n",
    "\n",
    "        if user_id not in rating_matrix.index:\n",
    "            print(f\"User '{user_id}' not found in the dataset.\")\n",
    "            return\n",
    "\n",
    "        user_idx = rating_matrix.index.get_loc(user_id)\n",
    "        user_similarity = similarity_matrix[user_idx]\n",
    "\n",
    "        # Show top 3 similar users\n",
    "        most_similar_users = [\n",
    "            (int(rating_matrix.index[i]), sim)\n",
    "            for i, sim in enumerate(user_similarity)\n",
    "            if i != user_idx\n",
    "        ]\n",
    "        most_similar_users = sorted(most_similar_users, key=lambda x: x[1], reverse=True)[:3]\n",
    "\n",
    "        print(\"\\nUsers with the most similar preferences to you:\")\n",
    "        for uid, score in most_similar_users:\n",
    "            print(f\"User {uid} (Similarity Score: {score:.4f})\")\n",
    "\n",
    "        # Recommend based on similar users\n",
    "        similar_users_idx = np.argsort(user_similarity)[::-1]\n",
    "        user_seen_movies = set(rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index)\n",
    "\n",
    "        recommendations = {}\n",
    "        for idx in similar_users_idx:\n",
    "            similar_user_ratings = rating_matrix.iloc[idx]\n",
    "            for movie, rating in similar_user_ratings.items():\n",
    "                if rating > 0 and movie not in user_seen_movies:\n",
    "                    recommendations[movie] = recommendations.get(movie, 0) + rating\n",
    "\n",
    "        if recommendations:\n",
    "            max_score = max(recommendations.values())\n",
    "            percent_recommendations = {\n",
    "                movie: (score / max_score) * 100 for movie, score in recommendations.items()\n",
    "            }\n",
    "            sorted_recommendations = sorted(percent_recommendations.items(), key=lambda x: x[1], reverse=True)\n",
    "        else:\n",
    "            sorted_recommendations = []\n",
    "\n",
    "        # Display recommendations\n",
    "        print(f\"\\nHere are {top_n} movies we think you'll enjoy, based on your preferences:\\n\")\n",
    "        for movie, percent in sorted_recommendations[:top_n]:\n",
    "            print(f\"{movie}: {percent:.1f}%\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3232541c-0208-43a6-8a6a-d0d3d506a6fc",
   "metadata": {},
   "outputs": [],
   "source": [
    "collaborative_menu(top_movies_collab_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fbc9b6bb-f7d2-4f9f-96d5-51f98c37d257",
   "metadata": {},
   "outputs": [],
   "source": [
    "def compare_multiple_users(user_ids, df):\n",
    "    # Pivot to get user-item matrix\n",
    "    rating_matrix = df.pivot(index='userName', columns='title', values='standardized_score')\n",
    "\n",
    "    # Check if all users exist\n",
    "    missing_users = [uid for uid in user_ids if uid not in rating_matrix.index]\n",
    "    if missing_users:\n",
    "        print(f\"User(s) not found in the dataset: {missing_users}\")\n",
    "        return\n",
    "\n",
    "    # Collect all user ratings\n",
    "    comparison = pd.DataFrame()\n",
    "    for uid in user_ids:\n",
    "        comparison[f'User {uid}'] = rating_matrix.loc[uid]\n",
    "\n",
    "    # Drop movies not rated by any of them\n",
    "    comparison = comparison.dropna(how='all')\n",
    "\n",
    "    # Print results\n",
    "    print(f\"\\nComparison of Users {', '.join(map(str, user_ids))}:\\n\")\n",
    "    print(comparison.sort_index())\n",
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
   "execution_count": null,
   "id": "50fcc850-0d93-4e25-913e-14424b4b50b9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "from sklearn.decomposition import TruncatedSVD\n",
    "from sklearn.metrics.pairwise import cosine_similarity\n",
    "\n",
    "def hybrid_recommender(user_id, movie_title, content_df, content_features, top_movies_collab_df, \n",
    "                       content_weight=0.5, collab_weight=0.5, top_n=10):\n",
    "    \"\"\"\n",
    "    Hybrid recommendation system that combines content-based and collaborative filtering\n",
    "    \"\"\"\n",
    "    # Normalize weights\n",
    "    total_weight = content_weight + collab_weight\n",
    "    content_weight = content_weight / total_weight\n",
    "    collab_weight = collab_weight / total_weight\n",
    "    \n",
    "    # 1. Get content-based recommendations\n",
    "    content_result = content_recommender(movie_title, content_df, content_features, \n",
    "                                         top_n=top_n*2, input_type='title')\n",
    "    \n",
    "    # Process content recommendations\n",
    "    content_recommendations = {}\n",
    "    if not isinstance(content_result, str):\n",
    "        content_recommendations = {\n",
    "            rec['title']: rec['similarity_score'] \n",
    "            for rec in content_result['recommendations']\n",
    "        }\n",
    "        # Normalize scores\n",
    "        if content_recommendations:\n",
    "            max_content_score = max(content_recommendations.values())\n",
    "            content_recommendations = {\n",
    "                movie: (score / max_content_score) * 100 \n",
    "                for movie, score in content_recommendations.items()\n",
    "            }\n",
    "    \n",
    "    # 2. Get collaborative recommendations\n",
    "    collab_recommendations = {}\n",
    "    \n",
    "    # Create the user-item matrix\n",
    "    rating_matrix = top_movies_collab_df.pivot(index='userName', columns='title', values='standardized_score')\n",
    "    rating_matrix = rating_matrix.fillna(0)\n",
    "    \n",
    "    # Apply SVD\n",
    "    svd = TruncatedSVD(n_components=20, random_state=42)\n",
    "    matrix_svd = svd.fit_transform(rating_matrix)\n",
    "    similarity_matrix = cosine_similarity(matrix_svd)\n",
    "    \n",
    "    # Check if user exists\n",
    "    if user_id in rating_matrix.index:\n",
    "        user_idx = rating_matrix.index.get_loc(user_id)\n",
    "        user_similarity = similarity_matrix[user_idx]\n",
    "        similar_users_idx = np.argsort(user_similarity)[::-1]\n",
    "        user_seen_movies = set(rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index)\n",
    "        \n",
    "        # Get recommendations from similar users\n",
    "        for idx in similar_users_idx:\n",
    "            if idx == user_idx:\n",
    "                continue\n",
    "            similar_user_ratings = rating_matrix.iloc[idx]\n",
    "            for movie, rating in similar_user_ratings.items():\n",
    "                if rating > 0 and movie not in user_seen_movies:\n",
    "                    collab_recommendations[movie] = collab_recommendations.get(movie, 0) + rating\n",
    "        \n",
    "        # Normalize collaborative scores\n",
    "        if collab_recommendations:\n",
    "            max_collab_score = max(collab_recommendations.values())\n",
    "            collab_recommendations = {\n",
    "                movie: (score / max_collab_score) * 100 \n",
    "                for movie, score in collab_recommendations.items()\n",
    "            }\n",
    "    \n",
    "    # 3. Combine recommendations with weighted scores\n",
    "    hybrid_scores = {}\n",
    "    \n",
    "    # Add content-based scores with weight\n",
    "    for movie, score in content_recommendations.items():\n",
    "        hybrid_scores[movie] = content_weight * score\n",
    "    \n",
    "    # Add collaborative scores with weight\n",
    "    for movie, score in collab_recommendations.items():\n",
    "        if movie in hybrid_scores:\n",
    "            hybrid_scores[movie] += collab_weight * score\n",
    "        else:\n",
    "            hybrid_scores[movie] = collab_weight * score\n",
    "    \n",
    "    # Sort and return top recommendations\n",
    "    sorted_recommendations = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)\n",
    "    \n",
    "    return sorted_recommendations[:top_n]\n",
    "\n",
    "def show_hybrid_recommendations(recommendations, content_df, content_features, user_movie):\n",
    "    # First, display details about the input movie\n",
    "    movie_details = content_df[content_df['title'] == user_movie]\n",
    "    if not movie_details.empty:\n",
    "        details = movie_details.iloc[0]\n",
    "        print(\"\\n===== Movie You Enjoyed =====\")\n",
    "        print(f\"Title: {details['title']}\")\n",
    "        print(f\"Year: {details.get('year', 'N/A')}\")\n",
    "        \n",
    "        genres = [col for col in content_features.columns \n",
    "                 if col not in ['runtimeMinutes', 'director', 'originalLanguage'] \n",
    "                 and details.get(col, 0) == 1]\n",
    "        print(f\"Genres: {', '.join(genres)}\")\n",
    "        \n",
    "        director_id = details.get('director', 'Unknown')\n",
    "        director_name = director_decoder.get(director_id, 'Unknown')\n",
    "        print(f\"Director: {director_name}\")\n",
    "        \n",
    "        language_id = details.get('originalLanguage', 'Unknown')\n",
    "        language_name = language_decoder.get(language_id, 'Unknown')\n",
    "        print(f\"Language: {language_name}\")\n",
    "        \n",
    "        print(f\"Runtime: {details.get('runtimeMinutes', 'N/A')} minutes\")\n",
    "        print(f\"TomatoMeter: {details.get('tomatoMeter', 'N/A')}%\")\n",
    "        print(\"=\" * 30)\n",
    "    \n",
    "    # Then show recommendations\n",
    "    print(\"\\n===== Hybrid Recommendations =====\")\n",
    "    for i, (movie, score) in enumerate(recommendations, 1):\n",
    "        # Find movie details in content_df\n",
    "        movie_details = content_df[content_df['title'] == movie]\n",
    "        \n",
    "        if not movie_details.empty:\n",
    "            details = movie_details.iloc[0]\n",
    "            print(f\"{i}. {movie} (Score: {score:.1f}%)\")\n",
    "            print(f\"   Year: {details.get('year', 'N/A')}\")\n",
    "            \n",
    "            genres = [col for col in content_features.columns \n",
    "                     if col not in ['runtimeMinutes', 'director', 'originalLanguage'] \n",
    "                     and details.get(col, 0) == 1]\n",
    "            print(f\"   Genres: {', '.join(genres)}\")\n",
    "            \n",
    "            director_id = details.get('director', 'Unknown')\n",
    "            director_name = director_decoder.get(director_id, 'Unknown')\n",
    "            print(f\"   Director: {director_name}\")\n",
    "            \n",
    "            language_id = details.get('originalLanguage', 'Unknown')\n",
    "            language_name = language_decoder.get(language_id, 'Unknown')\n",
    "            print(f\"   Language: {language_name}\")\n",
    "            \n",
    "            print(f\"   Runtime: {details.get('runtimeMinutes', 'N/A')} minutes\")\n",
    "            print(f\"   TomatoMeter: {details.get('tomatoMeter', 'N/A')}%\")\n",
    "        else:\n",
    "            print(f\"{i}. {movie} (Score: {score:.1f}%)\")\n",
    "        print(\"---\")\n",
    "\n",
    "def evaluate_hybrid_recommender(recommendations, user_id, movie_title, content_df, content_features, k=20000):\n",
    "    \"\"\"\n",
    "    Evaluate hybrid recommender system using precision, recall, and F1 score at K\n",
    "    \"\"\"\n",
    "    # Get actual genres of the original movie\n",
    "    movie_details = content_df[content_df['title'] == movie_title]\n",
    "    if movie_details.empty:\n",
    "        print(f\"Cannot find details for movie '{movie_title}'\")\n",
    "        return None\n",
    "    \n",
    "    original_movie = movie_details.iloc[0]\n",
    "    \n",
    "    # Get genres of the original movie\n",
    "    original_genres = [col for col in content_features.columns \n",
    "                       if col not in ['runtimeMinutes', 'director', 'originalLanguage'] \n",
    "                       and original_movie.get(col, 0) == 1]\n",
    "    \n",
    "    # Get director of the original movie\n",
    "    original_director = original_movie['director']\n",
    "    \n",
    "    # Get relevant movies (same genres or director)\n",
    "    relevant_movies = set()\n",
    "    for _, row in content_df.iterrows():\n",
    "        # Check if shares any genre\n",
    "        movie_genres = [col for col in content_features.columns \n",
    "                       if col not in ['runtimeMinutes', 'director', 'originalLanguage'] \n",
    "                       and row.get(col, 0) == 1]\n",
    "        \n",
    "        if any(genre in original_genres for genre in movie_genres) or row['director'] == original_director:\n",
    "            relevant_movies.add(row['title'])\n",
    "    \n",
    "    # Remove the original movie from the relevant set\n",
    "    if movie_title in relevant_movies:\n",
    "        relevant_movies.remove(movie_title)\n",
    "    \n",
    "    # Get recommended movies\n",
    "    recommended_movies = [movie for movie, _ in recommendations[:k]]\n",
    "    \n",
    "    # Calculate metrics\n",
    "    relevant_recommended = set(recommended_movies) & relevant_movies\n",
    "    \n",
    "    precision = len(relevant_recommended) / len(recommended_movies) if recommended_movies else 0\n",
    "    recall = len(relevant_recommended) / len(relevant_movies) if relevant_movies else 0\n",
    "    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0\n",
    "    \n",
    "    metrics = {\n",
    "        f'precision@{k}': precision,\n",
    "        f'recall@{k}': recall,\n",
    "        f'f1@{k}': f1\n",
    "    }\n",
    "    \n",
    "    return metrics\n",
    "\n",
    "def hybrid_menu(content_df, content_features, top_movies_collab_df):\n",
    "    print(\"Welcome to the Hybrid Movie Recommender!\")\n",
    "    \n",
    "    # Get user ID\n",
    "    user_id = input(\"Please enter your userName: \").strip()\n",
    "    try:\n",
    "        user_id = int(user_id)\n",
    "    except ValueError:\n",
    "        print(\"Invalid user ID. Please enter a numeric value.\")\n",
    "        return\n",
    "    \n",
    "    # Get a movie the user likes\n",
    "    movie_title = input(\"Enter a movie you enjoyed: \").strip()\n",
    "    \n",
    "    # Define weights\n",
    "    content_weight = 0.5\n",
    "    collab_weight = 0.5\n",
    "    \n",
    "    # Get recommendations\n",
    "    recommendations = hybrid_recommender(\n",
    "        user_id, movie_title, content_df, content_features, top_movies_collab_df, \n",
    "        content_weight, collab_weight\n",
    "    )\n",
    "    \n",
    "    # Show recommendations\n",
    "    show_hybrid_recommendations(recommendations, content_df, content_features, movie_title)\n",
    "    \n",
    "    # Evaluate the recommendations\n",
    "    metrics = evaluate_hybrid_recommender(\n",
    "        recommendations, user_id, movie_title, content_df, content_features\n",
    "    )\n",
    "    \n",
    "    # Display evaluation metrics\n",
    "    if metrics:\n",
    "        print(\"\\n===== Evaluation Metrics =====\")\n",
    "        for metric_name, value in metrics.items():\n",
    "            print(f\"{metric_name}: {value:.4f}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1435e678-e586-4187-a66e-b7d3143cdfff",
   "metadata": {},
   "outputs": [],
   "source": [
    "hybrid_menu(content_df, content_features, top_movies_collab_df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
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
