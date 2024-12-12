import numpy as np
import torch

from utils.data_postprocessing import denormalize_ratings
from utils.plot_utils import plot_predictions_vs_actuals


def create_testing_batches(X_train, X_test, b, batch_size):
    n = X_train.shape[0]
    batches = []

    for i in range(0, n, batch_size):
        X_train_batch = X_train[i: i + batch_size, :]
        X_test_batch = X_test[i: i + batch_size, :]
        b_batch = b[i: i + batch_size, :]
        batches.append((X_train_batch, X_test_batch, b_batch))

    return batches


def test_model(model, Z_train, X_test, B_test, loss_fn, batch_size, user_means, user_stds, device):
    model.eval()

    all_predictions = []
    all_actuals = []

    B_test_tensor = torch.tensor(B_test, dtype=torch.int).to(device)

    batches = create_testing_batches(Z_train, X_test, B_test_tensor, batch_size)

    with torch.no_grad():
        for Z_train_batch, X_test_batch, B_test_batch in batches:
            y_hat = model(Z_train_batch.unsqueeze(1)).squeeze(1)

            y_hat_masked = (y_hat * B_test_batch).cpu().numpy()
            actual_masked = (X_test_batch * B_test_batch).cpu().numpy()

            all_predictions.append(y_hat_masked)
            all_actuals.append(actual_masked)

    all_predictions = np.vstack(all_predictions)
    all_actuals = np.vstack(all_actuals)
    denormalize_predictions = denormalize_ratings(all_predictions, user_means, user_stds)

    positions_non_zero_non_nan = np.where((all_actuals != 0.0) & (~np.isnan(all_actuals)))
    test_predictions = denormalize_predictions[positions_non_zero_non_nan]
    test_actuals = all_actuals[positions_non_zero_non_nan]


    # rmse = loss_fn_denorm(torch.tensor(denormalize_predictions), torch.tensor(denormalize_actuals)).item()
    rmse = np.sqrt(np.mean(np.square(test_predictions - test_actuals)))

    print(f"Test RMSE on 1-5 scale: {rmse}")
    print(f"Predictions: {test_predictions}")
    print(f"Actuals: {test_actuals}")

    plot_predictions_vs_actuals(test_predictions, test_actuals)

