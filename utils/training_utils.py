# training_utils.py

import numpy as np
import torch


def create_user_batches(num_total_users, batch_size_of_users, shuffle=True):
    # ... (same as before)
    user_indices = np.arange(num_total_users)
    if shuffle:
        np.random.shuffle(user_indices)
    batches_of_user_indices = []
    for i in range(0, num_total_users, batch_size_of_users):
        batches_of_user_indices.append(user_indices[i: i + batch_size_of_users])
    return batches_of_user_indices


def train_epoch_full_graph(model, optimizer,
                           X_features_all_users,
                           Y_targets_all_users_norm,
                           B_loss_mask_all_users,
                           loss_fn, batch_size_of_users,
                           device):  # batch_size_of_users here is for iterating, not SGD batch
    model.train()

    optimizer.zero_grad()  # Zero gradients at the start of the epoch

    # --- Perform ONE forward pass for all users ---
    y_hat_all_users = model(X_features_all_users)  # Shape: (num_users, num_movies_predictions)

    # Accumulate loss across all users/batches
    total_loss_for_epoch_sum_reduction = torch.tensor(0.0, device=device)  # Ensure it's a tensor on the right device
    total_masked_elements_in_epoch = 0

    # Iterate through user batches just to manage potential memory for loss calculation sum,
    # but the loss is accumulated into a single tensor.
    batches_user_indices = create_user_batches(X_features_all_users.shape[0], batch_size_of_users,
                                               shuffle=False)  # No shuffle needed if one backward

    for user_idx_batch in batches_user_indices:
        y_hat_batch = y_hat_all_users[user_idx_batch, :]
        targets_batch = Y_targets_all_users_norm[user_idx_batch, :]
        mask_batch = B_loss_mask_all_users[user_idx_batch, :]

        predictions_masked = y_hat_batch * mask_batch
        targets_masked = targets_batch * mask_batch

        # loss_fn is nn.MSELoss(reduction='sum')
        current_batch_loss_sum = loss_fn(predictions_masked, targets_masked)

        num_elements_in_batch_loss = mask_batch.sum().item()

        if num_elements_in_batch_loss > 0:
            total_loss_for_epoch_sum_reduction = total_loss_for_epoch_sum_reduction + current_batch_loss_sum
            total_masked_elements_in_epoch += num_elements_in_batch_loss

    if total_masked_elements_in_epoch > 0:
        final_epoch_loss_normalized = total_loss_for_epoch_sum_reduction / total_masked_elements_in_epoch

        # Perform ONE backward pass for the entire epoch's accumulated loss
        final_epoch_loss_normalized.backward()
        optimizer.step()

        return final_epoch_loss_normalized.item()
    else:
        print("Warning: No elements to calculate loss in the entire epoch for training.")
        return 0.0  # Or handle as NaN or raise error