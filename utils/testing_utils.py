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
               X_features_test_users,  # (num_test_users, num_movies_features) - context
               X_targets_test_users_original,  # (num_test_users, num_movies_targets) - original scale
               B_mask_test_users,  # (num_test_users, num_movies_mask)
               user_means_np,
               user_stds_np,
               device):  # Make sure these are 1D numpy arrays of length num_users
    model.eval()

    all_predictions_denorm_list = []
    all_actuals_orig_list = []

    with torch.no_grad():
        # Model processes all test users' features
        y_hat_all_test_users_norm = model(X_features_test_users)  # Output: (num_test_users, num_movies)

        # Denormalize predictions
        # denormalize_ratings_user_x_movie needs to handle (num_users, num_movies)
        # and user_means/stds being (num_users,)
        predictions_denorm = denormalize_ratings_user_x_movie(
            y_hat_all_test_users_norm.cpu().numpy(),
            user_means_np,  # Should be (num_users,)
            user_stds_np  # Should be (num_users,)
        )

        # Iterate through users to apply mask (or vectorize if careful)
        for i in range(X_targets_test_users_original.shape[0]):  # Iterate over users
            user_preds_denorm = predictions_denorm[i, :]
            user_actuals_orig = X_targets_test_users_original[i, :].cpu().numpy()  # if tensor
            user_mask = B_mask_test_users[i, :].cpu().numpy()  # if tensor

            relevant_preds = user_preds_denorm[user_mask == 1]
            relevant_actuals = user_actuals_orig[user_mask == 1]

            all_predictions_denorm_list.extend(relevant_preds.tolist())
            all_actuals_orig_list.extend(relevant_actuals.tolist())

    test_predictions_flat = np.array(all_predictions_denorm_list)
    test_actuals_flat = np.array(all_actuals_orig_list)

    if len(test_actuals_flat) == 0:
        print("No test items to evaluate!")
        return float('nan')

    rmse = np.sqrt(np.mean(np.square(test_predictions_flat - test_actuals_flat)))

    print(f"Test RMSE on 1-5 scale: {rmse:.4f}")
    print(f"test_rmse {rmse:.4f}" )
    # print(f"Predictions (sample): {test_predictions_flat[:20]}")
    # print(f"Actuals (sample): {test_actuals_flat[:20]}")

    plot_predictions_vs_actuals(test_predictions_flat, test_actuals_flat)
    return rmse