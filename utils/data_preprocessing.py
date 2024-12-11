import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split

# Normalize the column of the user-movie matrix (per user)
def z_score_column_normalization(column):

    non_null_ratings = column[~np.isnan(column)]

    if len(non_null_ratings) > 1:
        mean = np.mean(non_null_ratings)
        std = np.std(non_null_ratings)
        if std == 0:
            std = 1

    else:
        mean = np.mean(non_null_ratings) if len(non_null_ratings) > 0 else 0
        std = 1

    normalized_column = (column - mean) / std

    return normalized_column, mean, std

def load_movielens_data(file_path, mask_percentage=0.1, seed=42):
    np.random.seed(seed)

    ratings = pd.read_csv(file_path)
    user_movie_matrix = ratings.pivot(index='movieId', columns='userId', values='rating')

    # bitmask for the available ratings (1) and the missing ratings (0)
    B0 = user_movie_matrix.notnull().astype(int).to_numpy()

    # Now, we will mask a percentage of the known ratings so to test them later.
    # B0 - total known ratings;
    # B1 - total known ratings without the ones we will be using for testing.

    mask = (B0 == 1)
    total_known = np.sum(mask)
    num_to_mask = int(mask_percentage * total_known)

    indices_to_mask = np.random.choice(np.arange(total_known), size=num_to_mask, replace=False)

    flat_matrix = user_movie_matrix.to_numpy().flatten()
    know_rating_indices = np.where(~np.isnan(flat_matrix))[0]
    flat_matrix[know_rating_indices[indices_to_mask]] = np.nan
    masked_matrix = flat_matrix.reshape(user_movie_matrix.shape)

    B1 = 1 - np.isnan(masked_matrix).astype(int)

    return user_movie_matrix.to_numpy(), masked_matrix, B0, B1


def normalize_and_fill_user_movie_matrix(masked_matrix):
    # apply - axis=0 means that we will normalize each column.
    # normalized_matrix = pd.DataFrame(masked_matrix).apply(z_score_column_normalization, axis=0)

    # fill the missing values with 0.
    df_masked_matrix = pd.DataFrame(masked_matrix)
    normalized_matrix = pd.DataFrame(index=df_masked_matrix.index, columns=df_masked_matrix.columns)
    user_mean = []
    user_std = []

    for col in df_masked_matrix.columns:
        normalized_column, mean, std = z_score_column_normalization(df_masked_matrix[col])
        normalized_matrix[col] = normalized_column
        user_mean.append(mean)
        user_std.append(std)

    normalized_matrix_filled = normalized_matrix.fillna(0)

    return normalized_matrix_filled, user_mean, user_std

