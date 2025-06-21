import numpy as np
import torch

from utils.data_postprocessing import denormalize_ratings_user_x_movie
from utils.plot_utils import plot_predictions_vs_actuals


def create_user_batches_test(num_total_users,
                             batch_size_of_users):
    user_indices = np.arange(num_total_users)
    batches_of_user_indices = []
    for i in range(0, num_total_users, batch_size_of_users):
        batches_of_user_indices.append(user_indices[i: i + batch_size_of_users])
    return batches_of_user_indices


def test_model(model,
               X_features_test_users,  # (num_users, num_movies) - test context
               X_targets_test_users_original,  # (num_users, num_movies) - original scale targets
               B_mask_test_users,  # (num_users, num_movies) - test mask
               user_means_np,  # (num_users,) - user means for denormalization
               user_stds_np,  # (num_users,) - user stds for denormalization
               device):
    """
    Test the SelectionGNN model on the test set.

    Args:
        model: SelectionGNN model
        X_features_test_users: Test features (num_users, num_movies)
        X_targets_test_users_original: Original scale test targets (num_users, num_movies)
        B_mask_test_users: Test mask indicating which ratings to evaluate (num_users, num_movies)
        user_means_np: User means for denormalization (num_users,)
        user_stds_np: User stds for denormalization (num_users,)
        device: PyTorch device

    Returns:
        RMSE on test set
    """
    model.eval()

    all_predictions_denorm_list = []
    all_actuals_orig_list = []

    with torch.no_grad():
        # Prepare input for SelectionGNN
        # SelectionGNN expects (batch_size, num_features, num_nodes)
        # We have (num_users, num_movies), need to transpose to (num_movies, num_users)
        # and add batch dimension: (1, num_movies, num_users)
        gnn_input = X_features_test_users.T.unsqueeze(0)  # (1, num_movies, num_users)

        # Forward pass through SelectionGNN
        # Output shape: (1, num_users * num_movies) since average=False
        gnn_output = model(gnn_input)  # (1, num_users * num_movies)

        # Reshape output back to (num_users, num_movies)
        num_users, num_movies = X_features_test_users.shape
        y_hat_test_users_norm = gnn_output.reshape(num_users, num_movies)

        # Denormalize predictions to original scale
        predictions_denorm = denormalize_ratings_user_x_movie(
            y_hat_test_users_norm.cpu().numpy(),
            user_means_np,  # (num_users,)
            user_stds_np  # (num_users,)
        )

        # Extract predictions and actuals only for test items (where mask == 1)
        test_mask_np = B_mask_test_users.cpu().numpy()
        test_targets_np = X_targets_test_users_original.cpu().numpy()

        # Get indices where test mask is 1
        test_indices = np.where(test_mask_np == 1)

        # Extract corresponding predictions and actuals
        test_predictions_flat = predictions_denorm[test_indices]
        test_actuals_flat = test_targets_np[test_indices]

    # Calculate RMSE only on test samples
    if len(test_actuals_flat) == 0:
        print("No test items to evaluate!")
        return float('nan')

    rmse = np.sqrt(np.mean(np.square(test_predictions_flat - test_actuals_flat)))

    print(f"Test RMSE on 1-5 scale: {rmse:.4f}")
    print(f"Number of test samples: {len(test_actuals_flat)}")
    print(f"test_rmse {rmse:.4f}")

    # Plot predictions vs actuals
    plot_predictions_vs_actuals(test_predictions_flat, test_actuals_flat)

    return rmse