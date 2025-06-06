import numpy as np
import torch
import pandas as pd
import scipy.stats as stats
from typing import List, Callable, Dict, Tuple, Set


def nanvar(
        tensor: torch.Tensor,
        dim: int=None,
        keepdim: bool=False,
        correction: int=1
) -> torch.Tensor:
    count = (~torch.isnan(tensor)).sum(dim=dim, keepdim=keepdim)

    mean = torch.nanmean(tensor, dim=dim, keepdim=True)

    sq_diff = (tensor - mean).pow(2)
    sq_diff = torch.where(
        torch.isnan(sq_diff),
        torch.zeros_like(sq_diff),
        sq_diff)

    sum_sq_diff = sq_diff.sum(dim=dim, keepdim=keepdim)

    divisor = (count - correction).clamp(min=1)

    return sum_sq_diff / divisor.to(sum_sq_diff.dtype)

def nanstd(
        tensor: torch.Tensor,
        dim: int=None,
        keepdim: bool=False,
        correction: int=1
) -> torch.Tensor:

    return nanvar(tensor, dim=dim, keepdim=keepdim, correction=correction).sqrt()


def user_user_covariance_torch(
        r: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # r : (|I| x |U|)
    means = torch.nanmean(r, dim=0)
    stds = nanstd(r, dim=0, correction=0)

    R_z = (r - means) / (stds + 1e-8)
    R_z = torch.nan_to_num(R_z, nan=0.0)

    n_items = R_z.shape[0]
    C = (R_z.t() @ R_z) / n_items
    return (C, means, stds + 1e-8)



