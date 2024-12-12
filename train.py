import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils.data_preprocessing import load_movielens_data, normalize_and_fill_user_movie_matrix, split_test_set, \
    split_val_set, normalize_and_fill_set
from utils.covariance_utils import compute_covariance_matrix
from models.gnn_model import MovieLensGNN
from constants import *
from utils.testing_utils import test_model
from utils.val_utils import validate_model
from utils.training_utils import train_epoch
from utils.plot_utils import plot_training_validation_performance
from itertools import product

#TODO: set the seed for torch for replicability. But RUN WITHOUT SEED!
np.random.seed(seed)
torch.manual_seed(seed)


file_path = 'ml-latest-small/ratings.csv'

# Load the data:
# X0 - all ratings;
# X1 - ratings with a percentage of them masked for testing;
# B0 - bitmask for the available ratings (1) and the missing ratings (0);
# B1 - bitmask for the available ratings without the ones we will be using for testing.
X0, B0 = load_movielens_data(file_path)



X1, B1, X_test, B_test = split_test_set(X0, B0, mask_percentage=mask_percentage, seed=seed)
X_train, B_train, X_val, B_val = split_val_set(X1, B1, mask_percentage=mask_percentage, seed=seed)


# Normalize and fill the user-movie matrix.
Z_train, user_means, user_stds = normalize_and_fill_user_movie_matrix(X_train) # returns pd.DataFrame + array + array
Z_train = Z_train.to_numpy()

Z_val = normalize_and_fill_set(X_val, user_means, user_stds)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}.")

#GSO gpu!
C = compute_covariance_matrix(Z_train)
C = torch.tensor(C, dtype=torch.float32).to(device)


input_dim = Z_train.shape[1]


Z_train = torch.tensor(Z_train, dtype=torch.float32).to(device)
# B_train = torch.tensor(B1, dtype=torch.int).to(device)

Z_val = torch.tensor(Z_val, dtype=torch.float32).to(device)
# B_val = torch.tensor(B_val, dtype=torch.int).to(device)

X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
# B_test = torch.tensor(B_test, dtype=torch.int).to(device)

dimNodeSignals_options = [[1, 16, 1], [1, 32, 1], [1, 64, 1]]
nTaps_options = [2, 3]
dimLayersMLP_options = [[1, 1], [1, 16, 1], [1, 64, 1]]
best_hyperparameters = None
best_val_loss = float('inf')

for dimNodeSignals, nTaps, dimLayersMLP in product(dimNodeSignals_options, nTaps_options, dimLayersMLP_options):

    # gpu!
    gnn_model = MovieLensGNN(C, input_dim, dimNodeSignals, nTaps, dimLayersMLP).to(device)
    optimizer = optim.Adam(gnn_model.parameters(), lr=lr) # TODO: try some different lr (0.01) | NO need to grid search.
    loss_fn = nn.MSELoss()

    # train_losses = []
    # val_losses = []
    train_loss, val_loss = 0, 0
    print(f"Training with hyperparameters: {dimNodeSignals}, {nTaps}, {dimLayersMLP}")

    for epoch in range(n_epochs):
        train_loss = train_epoch(gnn_model, optimizer, Z_train, B_train, loss_fn, batch_size, device)
        # train_losses.append(train_loss)

        val_loss = validate_model(gnn_model, Z_train, B_train, Z_val, B_val, loss_fn, batch_size, device)
        # val_losses.append(val_loss)

        print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_hyperparams = (dimNodeSignals, nTaps, dimLayersMLP)
        print(f"Best hyperparameters: {best_hyperparams} with val loss: {best_val_loss}")

    # plot_training_validation_performance(train_losses, val_losses, n_epochs)

    # test_model(gnn_model, Z_train, X_test, B_test, loss_fn, batch_size, user_means, user_stds, device)


