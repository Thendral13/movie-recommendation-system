import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('ratings.csv')
print(movies['title'].head(20))


# Remove missing values
movies['genres'] = movies['genres'].fillna('')

# Convert to string first (prevents list error)
movies['genres'] = movies['genres'].astype(str)

# Now split
movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))

# Get unique genres
all_genres = list(set(g for sublist in movies['genres'] for g in sublist if g != ''))

# Create one-hot columns
for genre in all_genres:
    movies[genre] = movies['genres'].apply(lambda x: int(genre in x))

user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
movie_features = movies[all_genres]
cosine_sim = cosine_similarity(movie_features, movie_features)
def recommend_movies(movie_title, top_n=10):
    idx = movies[movies['title'] == movie_title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_n+1]
    movie_indices = [i[0] for i in sim_scores]
    return movies['title'].iloc[movie_indices]
recommendations = recommend_movies("The Dark Knight Rises")
#recommendations = recommend_movies("The Dark Knight Rises")

# print(recommendations)
