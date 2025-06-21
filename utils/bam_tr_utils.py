import numpy as np
import torch
from constants import forward_ratio
from utils.metrics import calc_popularity, calc_collab_dissimilarity, calc_genre_dissimilarity, calc_hybrid_dissimilarity


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
                device,
                p_scores_pt=None,
                C_hybrid_pt=None,
                lambda_rmse=1.0,
                lambda_novelty=0.0,
                lambda_diversity=0.0,
                N_train=10):
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

    # Compute RMSE loss only on backward mask
    masked_preds = y_hat_all_users_UxM * bwd_mask_UxM
    masked_targets = Y_targets_all_users_norm_UxM * bwd_mask_UxM
    rmse_loss = loss_fn(masked_preds, masked_targets)
    count_bwd = int(bwd_mask_UxM.sum().item())

    if count_bwd > 0:
        rmse_loss = rmse_loss / count_bwd

        # Compute multi-objective loss
        if lambda_novelty > 0 or lambda_diversity > 0:
            novelty_loss, diversity_loss = compute_bam_losses(
                y_hat_all_users_UxM, bwd_mask_UxM, p_scores_pt, C_hybrid_pt, N_train, device
            )

            total_loss = (lambda_rmse * rmse_loss +
                          lambda_novelty * novelty_loss +
                          lambda_diversity * diversity_loss)

            if (lambda_novelty > 0 or lambda_diversity > 0) and torch.rand(1).item() < 0.1:  # Log 10% of batches
                print(f"Loss components - RMSE: {rmse_loss.item():.4f}, "
                      f"Novelty: {novelty_loss.item():.4f}, Diversity: {diversity_loss.item():.4f}")
        else:
            total_loss = rmse_loss

        # Backpropagate
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        return total_loss.item()
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


def compute_bam_losses(predictions_UxM, mask_UxM, p_scores_pt, C_hybrid_pt, N, device):
    """
    Compute novelty and diversity losses for multi-objective training.

    Args:
        predictions_UxM: Model predictions (U, M)
        mask_UxM: Training mask for this batch (U, M)
        p_scores_pt: Popularity scores tensor (M,)
        C_hybrid_pt: Hybrid dissimilarity matrix (M, M)
        N: Top-N cutoff for recommendations
        device: PyTorch device

    Returns:
        novelty_loss: Tensor scalar (lower novelty = higher loss)
        diversity_loss: Tensor scalar (lower diversity = higher loss)
    """
    U, M = predictions_UxM.shape

    novelty_losses = []
    diversity_losses = []

    for u in range(U):
        # Get user's predictions and mask
        user_preds = predictions_UxM[u]  # (M,)
        user_mask = mask_UxM[u]  # (M,)

        # Only consider items in the training mask for this user
        if user_mask.sum() == 0:
            continue

        # Get top-N predictions for items in mask
        masked_preds = torch.where(user_mask > 0, user_preds, torch.tensor(-float('inf'), device=device))
        _, top_indices = torch.topk(masked_preds, min(N, int(user_mask.sum())), largest=True)

        if len(top_indices) < 2:
            continue

        # Compute novelty loss (we want to maximize novelty, so minimize negative novelty)
        top_popularities = p_scores_pt[top_indices]
        novelty = torch.mean(1.0 - top_popularities)  # Average of (1 - popularity)
        novelty_loss = -novelty  # Negative because we want to maximize novelty

        # Compute diversity loss (we want to maximize diversity, so minimize negative diversity)
        if len(top_indices) >= 2:
            diversity_sum = 0.0
            pair_count = 0
            for i in range(len(top_indices)):
                for j in range(i + 1, len(top_indices)):
                    diversity_sum += C_hybrid_pt[top_indices[i], top_indices[j]]
                    pair_count += 1
            diversity = diversity_sum / pair_count if pair_count > 0 else 0.0
            diversity_loss = -diversity  # Negative because we want to maximize diversity
        else:
            diversity_loss = torch.tensor(0.0, device=device)

        novelty_losses.append(novelty_loss)
        diversity_losses.append(diversity_loss)

    # Average across users
    if novelty_losses:
        avg_novelty_loss = torch.stack(novelty_losses).mean()
        avg_diversity_loss = torch.stack(diversity_losses).mean()
    else:
        avg_novelty_loss = torch.tensor(0.0, device=device)
        avg_diversity_loss = torch.tensor(0.0, device=device)

    return avg_novelty_loss, avg_diversity_loss