import torch
import numpy as np


def create_user_batches_val(num_total_users,
                            batch_size_of_users):
    user_indices = np.arange(num_total_users)
    batches_of_user_indices = []
    for i in range(0, num_total_users, batch_size_of_users):
        batches_of_user_indices.append(user_indices[i: i + batch_size_of_users])
    return batches_of_user_indices


def validate_model(model,
                   X_features_val_users,  # (num_val_users, num_movies_features) - context
                   Y_targets_val_users_norm,  # (num_val_users, num_movies_targets) - ground truth
                   B_loss_mask_val_users,  # (num_val_users, num_movies_mask)
                   loss_fn,
                   batch_size_of_users,
                   device):
    model.eval()
    total_val_loss = 0.0
    total_masked_elements = 0

    with torch.no_grad():

        y_hat_all_val_users = model(X_features_val_users)

        batches_user_indices = create_user_batches_val(X_features_val_users.shape[0], batch_size_of_users)

        for user_idx_batch in batches_user_indices:
            y_hat_batch = y_hat_all_val_users[user_idx_batch, :]
            targets_batch = Y_targets_val_users_norm[user_idx_batch, :]
            mask_batch = B_loss_mask_val_users[user_idx_batch, :]

            predictions_masked = y_hat_batch * mask_batch
            targets_masked = targets_batch * mask_batch

            batch_loss = loss_fn(predictions_masked, targets_masked)

            num_elements_in_loss = mask_batch.sum().item()
            if num_elements_in_loss > 0:
                total_val_loss += batch_loss.item()
                total_masked_elements += num_elements_in_loss

    if total_masked_elements == 0:
        return float('inf')
    return total_val_loss / total_masked_elements
