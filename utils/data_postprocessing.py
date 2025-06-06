import numpy as np


# denormalize the prediction matrix
def denormalize_ratings_user_x_movie(prediction_matrix_norm_users_x_movies, # (num_users, num_movies)
                                     user_means_1d,  # 1D array of shape (num_users,)
                                     user_stds_1d):  # 1D array of shape (num_users,)

    means_expanded = user_means_1d[:, np.newaxis]
    stds_expanded = user_stds_1d[:, np.newaxis]

    denormalized_matrix = (prediction_matrix_norm_users_x_movies * stds_expanded) + means_expanded
    return denormalized_matrix
