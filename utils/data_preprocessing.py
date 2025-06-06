import sys
# prepend the path to the parent directory
sys.path.append('../')


import pandas as pd
import numpy as np
import torch
import sklearn

from utils.covariance_utils import nanstd
from sklearn.model_selection import train_test_split

def load_movielens_data(file_path):

    ratings = pd.read_csv(file_path)
    user_movie_matrix = ratings.pivot(index='movieId', columns='userId', values='rating')

    # bitmask for the available ratings (1) and the missing ratings (0)
    B0 = user_movie_matrix.notnull().astype(int).to_numpy()

    return user_movie_matrix.to_numpy(), B0

def split_test_set(X0, B0, mask_percentage=0.1, seed=42):
    # Now, we will mask a percentage of the known ratings so to test them later.
    # B0 - total known ratings;
    # B1 - total known ratings without the ones we will be using for testing.

    np.random.seed(seed)

    total_known = np.sum(B0)
    num_to_mask = int(mask_percentage * total_known)

    indices_to_mask = np.random.choice(np.arange(total_known), size=num_to_mask, replace=False)

    X1 = X0.flatten()
    known_rating_indices = np.where(~np.isnan(X1))[0]

    X1[known_rating_indices[indices_to_mask]] = np.nan
    X1 = X1.reshape(X0.shape)                   # this is Train & Validation set.
    B1 = 1 - np.isnan(X1).astype(int)

    B_test = B0 - B1
    X_test = X0 * B_test

    return X1, B1, X_test, B_test

def split_val_set(X1, B1, mask_percentage=1/9, seed=42):
    # Mask the eval from the training set.
    # X1 - Train & Validation set

    # X_train - Train set
    # X_val - Validation set

    np.random.seed(seed)

    total_ratings_cnt = np.sum(B1)
    masked_ratings_cnt = int(mask_percentage * total_ratings_cnt)

    indices = np.where(B1.flatten() == 1)[0]
    np.random.shuffle(indices)

    B_val = np.zeros_like(B1)
    X_val = np.copy(X1)

    for i in indices[:masked_ratings_cnt]:
        B_val[i // B1.shape[1], i % B1.shape[1]] = 1

    X_val = X_val * B_val
    B_train = B1 - B_val
    X_train = X1 * B_train

    return X_train, B_train, X_val, B_val


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

    return normalized_matrix_filled, np.array(user_mean), np.array(user_std)


def normalize_and_fill_set(X_set, user_means, user_stds):
    Z_val = (X_set - user_means) / user_stds

    return np.nan_to_num(Z_val)


def get_pytorch_normalized_inputs_and_targets(
        X0_movies_x_users_tensor: torch.Tensor,  # Original full ratings tensor
        # B0_mask_movies_x_users_tensor: torch.Tensor,  # Original full mask
        train_mask_movies_x_users_tensor: torch.Tensor,  # Mask for ratings in this specific (train/val) set
        user_means_for_norm: torch.Tensor = None,  # Optional: For val/test
        user_stds_for_norm: torch.Tensor = None  # Optional: For val/test
):
    # X0 contains actual ratings or NaNs
    # train_mask indicates which of X0 are part of the current set's *known* ratings

    # 1. Create the context/feature matrix (Z)
    #    - For training Z, use only train_mask ratings to calculate means/stds if not provided
    #    - For val/test Z, use provided train_means/stds
    #    - Normalize, then fill all NaNs (original or from masking) with 0

    ratings_for_norm_stats = X0_movies_x_users_tensor * train_mask_movies_x_users_tensor
    # Make unmasked NaNs for stat calculation truly NaN, not 0*NaN
    ratings_for_norm_stats = torch.where(train_mask_movies_x_users_tensor == 1, X0_movies_x_users_tensor, torch.nan)

    if user_means_for_norm is None:  # Calculate for training
        user_means_for_norm = torch.nanmean(ratings_for_norm_stats, dim=0, keepdim=True)
        user_stds_for_norm = nanstd(ratings_for_norm_stats, dim=0, keepdim=True, correction=0)  # Use population std
        user_stds_for_norm = torch.where(user_stds_for_norm == 0,
                                         torch.tensor(1.0, device=X0_movies_x_users_tensor.device),
                                         user_stds_for_norm)  # Avoid div by zero

    Z_features_movies_x_users = (ratings_for_norm_stats - user_means_for_norm) / (user_stds_for_norm + 1e-8)
    Z_features_movies_x_users = torch.nan_to_num(Z_features_movies_x_users, nan=0.0)  # Fill all NaNs

    # 2. Create the target matrix (Y) and loss mask (B_loss)
    #    - Y should be the normalized *actual known* ratings for this set
    #    - B_loss is just train_mask_movies_x_users_tensor

    # Normalize only the actual known ratings for the target
    actual_known_ratings_in_set = ratings_for_norm_stats * train_mask_movies_x_users_tensor
    # Make non-target NaNs for normalization, then fill
    actual_known_ratings_in_set_for_norm = torch.where(train_mask_movies_x_users_tensor == 1, ratings_for_norm_stats,
                                                       torch.nan)

    Y_targets_norm_movies_x_users = (actual_known_ratings_in_set_for_norm - user_means_for_norm) / (
                user_stds_for_norm + 1e-8)
    Y_targets_norm_movies_x_users = torch.nan_to_num(Y_targets_norm_movies_x_users, nan=0.0)
    # Ensure only target items contribute to Y by re-masking (nan_to_num might fill non-targets if X0 had NaNs there)
    Y_targets_norm_movies_x_users = Y_targets_norm_movies_x_users * train_mask_movies_x_users_tensor

    B_loss_mask_movies_x_users = train_mask_movies_x_users_tensor

    # Transpose to users_x_movies for GNN
    Z_features_users_x_movies = Z_features_movies_x_users.T
    Y_targets_norm_users_x_movies = Y_targets_norm_movies_x_users.T
    B_loss_mask_users_x_movies = B_loss_mask_movies_x_users.T

    # Also return means/stds if calculated for training
    if user_means_for_norm is not None and user_means_for_norm.ndim > 1:  # Squeezing if it was keepdim=True
        user_means_for_norm = user_means_for_norm.squeeze(0)
        user_stds_for_norm = user_stds_for_norm.squeeze(0)

    return  Z_features_users_x_movies, \
            Y_targets_norm_users_x_movies, \
            B_loss_mask_users_x_movies, \
            user_means_for_norm, \
            user_stds_for_norm

