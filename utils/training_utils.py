import numpy as np
import torch
from constants import forward_ratio


def create_user_batches(
        num_total_users,
        batch_size_of_users,
        shuffle=True,
        device=None):
    """Create batches of user indices for batch processing."""
    user_idx = torch.arange(num_total_users, dtype=torch.long,
                            device=device or torch.device('cpu'))
    if shuffle:
        user_idx = user_idx[torch.randperm(num_total_users, device=user_idx.device)]
    # Split into chunks of size batch_size_of_users
    return list(user_idx.split(batch_size_of_users))


def random_training_split(mask_train_UxM, forward_ratio):
    """
    Randomly split the binary training mask into forward/backward masks.
    Each epoch, a new random split ensures the GNN sees different input subsets.

    Args:
        mask_train_UxM: Binary mask of shape (num_users, num_movies)
        forward_ratio: Fraction of training data to use for forward pass

    Returns:
        fwd_mask_UxM: Forward pass mask (num_users, num_movies)
        bwd_mask_UxM: Backward pass mask (num_users, num_movies)
    """
    U, M = mask_train_UxM.shape

    # Flatten mask to work with indices
    flat = mask_train_UxM.reshape(-1)  # (U*M,)

    # Get indices of known ratings
    idx = torch.nonzero(flat, as_tuple=False).squeeze(1)  # known-rating indices
    perm = torch.randperm(idx.numel(), device=flat.device)

    # Split into forward and backward sets
    n_fwd = int(idx.numel() * forward_ratio)
    fwd_idx, bwd_idx = idx[perm[:n_fwd]], idx[perm[n_fwd:]]

    # Create forward mask
    fwd_flat = torch.zeros_like(flat)
    fwd_flat[fwd_idx] = 1

    # Create backward mask
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
    Perform one epoch of SelectionGNN training using a random forward/backward split.

    The training process:
    1) Split training mask into forward/backward subsets
    2) Use forward subset to mask inputs for GNN forward pass
    3) Compute loss only on backward subset predictions
    4) Backpropagate and update parameters

    Args:
        model: SelectionGNN model
        optimizer: PyTorch optimizer
        X_features_all_users_UxM: Input features (num_users, num_movies)
        Y_targets_all_users_norm_UxM: Target ratings normalized (num_users, num_movies)
        B_loss_mask_all_users_UxM: Training mask (num_users, num_movies)
        loss_fn: Loss function (e.g., MSELoss)
        batch_size_of_users: Batch size for users (not used in SelectionGNN forward pass)
        device: PyTorch device

    Returns:
        Average loss for the epoch
    """
    model.train()

    # Split training mask into forward/backward subsets
    fwd_mask_UxM, bwd_mask_UxM = random_training_split(B_loss_mask_all_users_UxM, forward_ratio)

    # Mask inputs with forward mask
    masked_features_UxM = X_features_all_users_UxM * fwd_mask_UxM

    # Prepare input for SelectionGNN
    # SelectionGNN expects (batch_size, num_features, num_nodes)
    # We have (num_users, num_movies), need to transpose to (num_movies, num_users)
    # and add batch dimension: (1, num_movies, num_users)
    gnn_input = masked_features_UxM.T.unsqueeze(0)  # (1, num_movies, num_users)

    # Forward pass through SelectionGNN
    # Output shape: (1, num_users * num_movies) since average=False
    gnn_output = model(gnn_input)  # (1, num_users * num_movies)

    # Reshape output back to (num_users, num_movies)
    num_users, num_movies = X_features_all_users_UxM.shape
    y_hat_all_users_UxM = gnn_output.reshape(num_users, num_movies)

    # Apply backward mask to predictions and targets for loss computation
    masked_preds = y_hat_all_users_UxM * bwd_mask_UxM
    masked_targets = Y_targets_all_users_norm_UxM * bwd_mask_UxM

    # Compute loss only on backward mask
    total_loss_sum = loss_fn(masked_preds, masked_targets)  # scalar sum
    count_bwd = int(bwd_mask_UxM.sum().item())

    # Backpropagate if we have any training samples
    if count_bwd > 0:
        epoch_loss = total_loss_sum / count_bwd  # Average loss
        optimizer.zero_grad()
        epoch_loss.backward()
        optimizer.step()
        return epoch_loss.item()
    else:
        print("Warning: No training samples in this epoch!")
        return 0.0


def validate_epoch(model,
                   X_features_all_users_UxM,
                   Y_targets_all_users_norm_UxM,
                   B_val_mask_all_users_UxM,
                   loss_fn,
                   batch_size_of_users,
                   device):
    """
    Perform validation epoch using SelectionGNN.

    Args:
        model: SelectionGNN model
        X_features_all_users_UxM: Input features (num_users, num_movies)
        Y_targets_all_users_norm_UxM: Target ratings normalized (num_users, num_movies)
        B_val_mask_all_users_UxM: Validation mask (num_users, num_movies)
        loss_fn: Loss function
        batch_size_of_users: Batch size for users (not used in SelectionGNN forward pass)
        device: PyTorch device

    Returns:
        Average validation loss
    """
    model.eval()

    with torch.no_grad():
        # Prepare input for SelectionGNN
        # Use all available training data for features (no forward/backward split in validation)
        gnn_input = X_features_all_users_UxM.T.unsqueeze(0)  # (1, num_movies, num_users)

        # Forward pass
        gnn_output = model(gnn_input)  # (1, num_users * num_movies)

        # Reshape output back to (num_users, num_movies)
        num_users, num_movies = X_features_all_users_UxM.shape
        y_hat_all_users_UxM = gnn_output.reshape(num_users, num_movies)

        # Apply validation mask
        masked_preds = y_hat_all_users_UxM * B_val_mask_all_users_UxM
        masked_targets = Y_targets_all_users_norm_UxM * B_val_mask_all_users_UxM

        # Compute validation loss
        total_loss_sum = loss_fn(masked_preds, masked_targets)
        count_val = int(B_val_mask_all_users_UxM.sum().item())

        if count_val > 0:
            return (total_loss_sum / count_val).item()
        else:
            return 0.0