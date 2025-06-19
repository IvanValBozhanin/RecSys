import numpy as np
import torch
from constants import forward_ratio


def create_user_batches(
        num_total_users,
        batch_size_of_users,
        shuffle=True,
        device=None):

    user_idx = torch.arange(num_total_users, dtype=torch.long,
                            device=device or torch.device('cpu'))
    if shuffle:
        user_idx = user_idx[torch.randperm(num_total_users, device=user_idx.device)]
    # Split into chunks of size batch_size_of_users
    return list(user_idx.split(batch_size_of_users))


def random_training_split(
        mask_train_UxM,
        forward_ratio
):
    """
    Randomly split the binary training mask into forward/backward masks.
    Each epoch, a new random split ensures the GNN sees different input subsets.
    """

    U, M = mask_train_UxM.shape

    flat = mask_train_UxM.reshape(-1)  # (U*M,)

    idx = torch.nonzero(flat, as_tuple=False).squeeze(1)  # known-rating indices
    perm = torch.randperm(idx.numel(), device=flat.device)

    n_fwd = int(idx.numel() * forward_ratio)
    fwd_idx, bwd_idx = idx[perm[:n_fwd]], idx[perm[n_fwd:]]

    fwd_flat = torch.zeros_like(flat)
    fwd_flat[fwd_idx] = 1

    bwd_flat = torch.zeros_like(flat)
    bwd_flat[bwd_idx] = 1
    return fwd_flat.reshape(U, M), bwd_flat.reshape(U, M)


def train_epoch(model,
                optimizer,
                X_features_all_users_UxM,
                Y_targets_all_users_norm_UxM,
                B_loss_mask_all_users_UxM,
                loss_fn,
                batch_size_of_users,
                device):
    """
    Perform one epoch of GNN training using a random forward/backward split.

    1) Split training mask into forward/backward subsets.
    2) Mask inputs by forward subset and do a single full-graph forward pass.
    3) Accumulate loss over backward subset in user-batches.
    4) Backpropagate once on the aggregated loss.
    """
    model.train()

    fwd_np, bwd_np = random_training_split(B_loss_mask_all_users_UxM, forward_ratio)

    fwd_mask = torch.tensor(fwd_np, dtype=torch.int, device=device)
    bwd_mask = torch.tensor(bwd_np, dtype=torch.int, device=device)

    y_hat_all_users = model(X_features_all_users_UxM * fwd_mask)  # (U, M)

    total_loss_sum = torch.tensor(0.0, device=device)
    total_count = 0

    num_users = X_features_all_users_UxM.shape[0]
    user_batches = create_user_batches(num_users, batch_size_of_users, shuffle=False, device=device)

    for user_idx_batch in user_batches:
        preds_batch = y_hat_all_users[user_idx_batch, :]                  # (u_batch, M)
        targets_batch = Y_targets_all_users_norm_UxM[user_idx_batch, :]   # (u_batch, M)
        mask_bwd_batch = bwd_mask[user_idx_batch, :]                      # (u_batch, M)

        masked_preds = preds_batch * mask_bwd_batch
        masked_tgts  = targets_batch * mask_bwd_batch
        batch_loss_sum = loss_fn(masked_preds, masked_tgts)  # scalar

        count_bwd = int(mask_bwd_batch.sum().item())

        if count_bwd > 0:
            total_loss_sum += batch_loss_sum
            total_count += count_bwd

    if total_count > 0:
        epoch_loss = total_loss_sum / total_count
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        return epoch_loss.item()
    else:
        return 0.0
