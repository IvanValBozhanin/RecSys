import numpy as np


# denormalize the prediction matrix
def denormalize_ratings(prediction_matrix, user_means, user_stds):
    means_expanded = np.expand_dims(user_means, axis=0)
    stds_expanded = np.expand_dims(user_stds, axis=0)

    denormalized_matrix = (prediction_matrix * stds_expanded) + means_expanded

    return denormalized_matrix
