import numpy as np
import torch

def sparsify_covariance(C, cov_type, thr=0.0, p=0.1, sparse_tensor=False, sparsification_value = 1e-3):
    print(f"Sparsifying covariance matrix with type: {cov_type}, threshold: {thr}, probability: {p}")
    device = C.device
    if cov_type == "standard":
        C_sparse = C
    elif cov_type == "RCV": 
        # Generate probability values
        sigma = min((1-p)/3, p/3)
        lim_prob = np.linspace(0,1,1000)
        distr_prob = 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((lim_prob-p)/sigma)**2)
        distr_prob = distr_prob / distr_prob.sum()
        prob_values = np.random.choice(lim_prob, p=distr_prob, size=C.shape[0] ** 2)
        prob_values = torch.tensor(np.sort(prob_values), dtype=torch.float32, device=device)

        # Assign probability values 
        sorted_idx = torch.argsort(C.abs().flatten())
        prob = torch.zeros([C.shape[0] ** 2,], device=device).float().scatter_(0, sorted_idx, prob_values)
        prob = prob.reshape(C.shape)
        prob[torch.eye(prob.shape[0], device=device).long()] = 1 # no removal on the diagonal
        
        # Drop edges symmetrically
        mask = torch.rand(C.shape, device=device) <= prob
        triu = torch.triu(torch.ones(C.shape, device=device), diagonal=0).bool()
        mask = mask * triu + mask.t() * ~triu # make resulting matrix symmetric
        C_sparse = torch.where(mask, C, sparsification_value)

    elif cov_type == "ACV":
        prob = C.abs() / C.abs().max()
        prob[torch.eye(prob.shape[0], device=device).long()] = 1 # no removal on the diagonal
        mask = torch.rand(C.shape, device=device) <= prob
        triu = torch.triu(torch.ones(C.shape, device=device), diagonal=0).bool()
        mask = mask * triu + mask.t() * ~triu # make resulting matrix symmetric
        C_sparse = torch.where(mask, C, sparsification_value)

    elif cov_type == "hard_thr":
        C_sparse = torch.where(C.abs() > thr, C, sparsification_value)
    elif cov_type == "soft_thr":
        C_sparse = torch.where(C.abs() > thr, C - (C>0).float()*thr, sparsification_value)

    if sparse_tensor:
        return C_sparse.to_sparse()
    
    return C_sparse

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
        Z_features_movies_x_users: torch.Tensor,  # Normalized, zero-filled features movies x users
        cov_type: str = "standard",  # Options: "standard", "RCV", "ACV", "hard_thr", "soft_thr"
        thr: float = 0.0,  # Threshold for hard/soft thresholding
        p: float = 0.0,  # Probability for RCV/ACV
        sparsification_value: float = 1e-3  # Value to fill in for sparsification
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
    C_sparse = sparsify_covariance(C, cov_type, thr, p, sparsification_value=sparsification_value)
    return C_sparse