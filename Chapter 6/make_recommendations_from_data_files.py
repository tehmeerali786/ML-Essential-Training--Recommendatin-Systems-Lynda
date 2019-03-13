import numpy as np
import pandas as pd
import scipy
import matrix_factorization_utilities
import pickle

# Load prediction rules from data files
movies_df = pd.read_csv('movies.csv', index_col='movie_id')

print("Enter a user_id to get recommendations (Between 1 and 100):")
user_id_to_search = int(input())

print("Movies we will recommend:")

user_ratings = predicted_ratings[user_id_to_search - 1]
movies_df['rating'] = user_ratings
movies_df = movies_df.sort_values(by=['rating'], ascending=False)

print(movies_df[['title', 'genre', 'rating']].head(5))