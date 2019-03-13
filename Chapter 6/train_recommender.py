import numpy as np
import pandas as pd
import scipy
import matrix_factorization_utilities
import pickle

# Load user ratings
raw_dataset_df = pd.read_csv('movie_ratings_data_set.csv')

# Convert the running list of user ratings into a matrix
ratings_df = pd.pivot_table(raw_dataset_df, index='user_id', columns='movie_id', aggfunc=np.max)

# Apply matrix factorization to find the latent features
U, M = matrix_factorization_utilities.low_rank_matrix_factorization(ratings_training_df.as_matrix(), 
                                                                          num_features=15, regularization_amount=0.1)

# Find all predicted ratings by multiplying U and M
predicted_ratings = np.matmul(U, M)

# Save features and predicted ratings by multiplying U and M
pickle.dump(U, open("user_features.dat", "wb"))
pickle.dump(M, open("product_features.dat", "wb"))
pickle.dump(predicted_ratings, open("predicted_ratings.dat", "wb"))