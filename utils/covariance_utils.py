import numpy as np
import torch


def compute_covariance_matrix(Z1):

    # (GSO) is the covariance matrix based on the user-movie matrix.
    # [N_users x N_users]

    N_items = Z1.shape[0]

    C = (1 / N_items) * np.matmul(Z1.T, Z1)

    return C
