import torch
from utils.training_utils import validate_epoch


def validate_model(model,
                   X_features_all_users_UxM,
                   Y_targets_all_users_norm_UxM,
                   B_val_mask_all_users_UxM,
                   loss_fn,
                   batch_size_of_users,
                   device):
    """
    Validate the model on the validation set.

    This is a wrapper around validate_epoch to maintain consistency
    with the existing codebase.

    Args:
        model: SelectionGNN model
        X_features_all_users_UxM: Input features (num_users, num_movies)
        Y_targets_all_users_norm_UxM: Normalized target ratings (num_users, num_movies)
        B_val_mask_all_users_UxM: Validation mask (num_users, num_movies)
        loss_fn: Loss function
        batch_size_of_users: Batch size for processing users
        device: PyTorch device

    Returns:
        Average validation loss
    """
    return validate_epoch(
        model,
        X_features_all_users_UxM,
        Y_targets_all_users_norm_UxM,
        B_val_mask_all_users_UxM,
        loss_fn,
        batch_size_of_users,
        device
    )