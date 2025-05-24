import numpy as np
import torch


def nanvar(
        tensor: torch.Tensor,
        dim: int = None,
        keepdim: bool = False,
        correction: int = 1
) -> torch.Tensor:
    # ... (your implementation) ...
    count = (~torch.isnan(tensor)).sum(dim=dim, keepdim=keepdim)
    mean = torch.nanmean(tensor, dim=dim, keepdim=True)
    sq_diff = (tensor - mean).pow(2)
    # Ensure NaNs that were part of (tensor - mean) don't contribute if original was NaN
    sq_diff = torch.where(torch.isnan(tensor), torch.zeros_like(sq_diff), sq_diff)
    sq_diff_filled = torch.nan_to_num(sq_diff, nan=0.0)  # Fill any other NaNs from computation with 0

    sum_sq_diff = sq_diff_filled.sum(dim=dim, keepdim=keepdim)
    divisor = (count - correction).clamp(min=1)
    return sum_sq_diff / divisor.to(sum_sq_diff.dtype)


def nanstd(
        tensor: torch.Tensor,
        dim: int = None,
        keepdim: bool = False,
        correction: int = 1
) -> torch.Tensor:
    return nanvar(tensor, dim=dim, keepdim=keepdim, correction=correction).sqrt()


def compute_user_user_covariance_torch(
        Z_features_movies_x_users: torch.Tensor  # Normalized, zero-filled features movies x users
) -> torch.Tensor:
    # Z_features_movies_x_users: (|I| x |U|) - already normalized and 0-filled where originally NaN
    # For covariance, we treat users as variables and items as samples/observations.
    # Or, treat items as variables and users as samples.
    # If Z is (items x users), and we want user-user covariance:
    # Standard covariance: (X - mu).T @ (X - mu) / (n-1) where X is (samples x features)
    # Here, items are "samples/observations", users are "features".
    # So we want covariance between columns of Z_features_movies_x_users.
    # Z_centered = Z_features_movies_x_users - Z_features_movies_x_users.mean(dim=0, keepdim=True) # Center per user
    # user_user_cov = (Z_centered.T @ Z_centered) / (Z_features_movies_x_users.shape[0] -1)

    # Your function's approach:
    # r : (|I| x |U|)
    # means = torch.nanmean(r, dim=0) # Mean rating per user
    # stds = nanstd(r, dim=0, correction=0) # Std rating per user
    # R_z = (r - means) / (stds + 1e-8) # Normalize per user
    # R_z = torch.nan_to_num(R_z, nan=0.0) # Fill NaNs with 0 (user's mean normalized rating)
    # n_items = R_z.shape[0]
    # C = (R_z.t() @ R_z) / n_items
    # This `R_z` is essentially your `Z_features_movies_x_users`.
    # So if the input to this function is already normalized and zero-filled (as Z_features will be),
    # then we can directly compute the covariance.

    n_items = Z_features_movies_x_users.shape[0]
    # If Z_features_movies_x_users is already centered around 0 (due to z-score with 0 fill),
    # then mean subtraction is not strictly necessary here again.
    # The division by n_items makes it more like a "second moment matrix" if not centered,
    # or scaled covariance if centered. This is a common GSO in GCNs.
    C = (Z_features_movies_x_users.t() @ Z_features_movies_x_users) / n_items
    return C