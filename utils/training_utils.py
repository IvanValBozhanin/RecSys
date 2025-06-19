import numpy as np
import torch
from constants import forward_ratio


def create_user_batches(
        num_total_users,
        batch_size_of_users,
        shuffle=True):

    user_indices = np.arange(num_total_users)
    if shuffle:
        np.random.shuffle(user_indices)

    batches_of_user_indices = []
    for start in range(0, num_total_users, batch_size_of_users):
        end = start + batch_size_of_users
        batches_of_user_indices.append(user_indices[start:end])
    return batches_of_user_indices


def random_training_split(
        mask_train_UxM,
        forward_ratio
):
    """
    Randomly split the binary training mask into forward/backward masks.
    Each epoch, a new random split ensures the GNN sees different input subsets.
    """

    if torch.is_tensor(mask_train_UxM):
        mask_np = mask_train_UxM.cpu().numpy()
    else:
        mask_np = mask_train_UxM

    flat_idx = np.where(mask_np.flatten() == 1)[0]
    np.random.shuffle(flat_idx)

    n_total = flat_idx.size
    n_forward = int(n_total * forward_ratio)

    fwd_idx = flat_idx[:n_forward]
    bwd_idx = flat_idx[n_forward:]

    fwd_flat = np.zeros(mask_np.size, dtype=np.int8)
    bwd_flat = np.zeros(mask_np.size, dtype=np.int8)
    fwd_flat[fwd_idx] = 1
    bwd_flat[bwd_idx] = 1

    fwd_mask_UxM = fwd_flat.reshape(mask_np.shape)
    bwd_mask_UxM = bwd_flat.reshape(mask_np.shape)
    return fwd_mask_UxM, bwd_mask_UxM


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

    X_in = X_features_all_users_UxM * fwd_mask

    y_hat_all_users = model(X_in)  # (U, M)

    total_loss_sum = torch.tensor(0.0, device=device)
    total_count = 0

    num_users = X_features_all_users_UxM.shape[0]
    user_batches = create_user_batches(num_users, batch_size_of_users, shuffle=False)

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
