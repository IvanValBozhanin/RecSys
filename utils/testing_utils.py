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


def test_model(model, X_train, X_test, B_eval, loss_fn, batch_size, user_means, user_stds, device):
    model.eval()
    test_loss = 0.0

    all_predictions = []
    all_actuals = []

    batches = create_testing_batches(X_train, X_test, B_eval, batch_size)

    with torch.no_grad():
        for X_train_batch, X_test_batch, B_batch in batches:
            X_batch_shaped = X_train_batch.unsqueeze(1)
            y_hat = model(X_batch_shaped).squeeze(1)

            y_hat_masked = (y_hat * B_batch).cpu().numpy()

            actual_masked = (X_test_batch * B_batch).cpu().numpy()

            # batch_loss = loss_fn(y_hat_masked, actual_masked)
            # test_loss += batch_loss.item()

            all_predictions.append(y_hat_masked)
            all_actuals.append(actual_masked)

            # print(f"y_hat_masked: {y_hat_masked}")
            # print(f"actual_masked: {actual_masked}")

    all_predictions = np.vstack(all_predictions)
    all_actuals = np.vstack(all_actuals)

    positions_non_zero_non_nan = np.where((all_actuals != 0.0) & (~np.isnan(all_actuals)))

    denormalize_predictions = denormalize_ratings(all_predictions, user_means, user_stds)

    test_predictions = denormalize_predictions[positions_non_zero_non_nan]
    test_actuals = all_actuals[positions_non_zero_non_nan]


    # rmse = loss_fn_denorm(torch.tensor(denormalize_predictions), torch.tensor(denormalize_actuals)).item()
    rmse = np.sqrt(np.mean(np.square(test_predictions - test_actuals)))

    print(f"Test Loss (MSE) z-scored: {test_loss}")
    print(f"Test RMSE on 1-5 scale: {rmse}")
    print(f"Predictions: {test_predictions}")
    print(f"Actuals: {test_actuals}")

    plot_predictions_vs_actuals(test_predictions, test_actuals)

