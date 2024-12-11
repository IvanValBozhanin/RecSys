import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np

from utils.data_preprocessing import load_movielens_data, normalize_and_fill_user_movie_matrix
from utils.covariance_utils import compute_covariance_matrix
from models.gnn_model import MovieLensGNN
from constants import *
from utils.testing_utils import test_model
from utils.eval_utils import evaluate_model
from utils.training_utils import train_epoch
from utils.plot_utils import plot_training_validation_performance

#TODO: set the seed for torch for replicability. But RUN WITHOUT SEED!
np.random.seed(seed)
torch.manual_seed(seed)


file_path = 'ml-latest-small/ratings.csv'

# Load the data:
# X0 - all ratings;
# X1 - ratings with a percentage of them masked for testing;
# B0 - bitmask for the available ratings (1) and the missing ratings (0);
# B1 - bitmask for the available ratings without the ones we will be using for testing.
X0, B0 = load_movielens_data(file_path, mask_percentage=mask_percentage, seed=seed)


X0, X1, B0, B1 = load_movielens_data(file_path, mask_percentage=mask_percentage, seed=seed)
# print(X0)
# Normalize and fill the user-movie matrix.
Z1, user_means, user_stds = normalize_and_fill_user_movie_matrix(X1)

Z1 = Z1.to_numpy()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}.")


#GSO gpu!
C = compute_covariance_matrix(Z1)
C = torch.tensor(C, dtype=torch.float32).to(device)


input_dim = Z1.shape[1]
# gpu!
gnn_model = MovieLensGNN(C, input_dim).to(device)

X_train = torch.tensor(Z1, dtype=torch.float32).to(device)
B_train = torch.tensor(B1, dtype=torch.int).to(device)

X_test = torch.tensor(X0, dtype=torch.float32).to(device)
B_eval = torch.tensor(B0 - B1, dtype=torch.int).to(device)

optimizer = optim.Adam(gnn_model.parameters(), lr=lr) # TODO: try some different lr (0.01) | NO need to grid search.
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []

for epoch in range(n_epochs):
    train_loss = train_epoch(gnn_model, optimizer, X_train, B1, loss_fn, batch_size, device)
    train_losses.append(train_loss)

    eval_loss = evaluate_model(gnn_model, X_train, B1, loss_fn, batch_size, device)
    val_losses.append(eval_loss)

    print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss}, Eval Loss: {eval_loss}')

plot_training_validation_performance(train_losses, val_losses, n_epochs)

test_model(gnn_model, X_train, X_test, B_eval, loss_fn, batch_size, user_means, user_stds, device)


