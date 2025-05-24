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



def cross_validate_model(X1, B1, n_folds, best_hyperparams, loss_fn, batch_size, n_epochs, device):

    fold_metrics = []

    for fold in range(n_folds):
        print(f"Starting : Fold {fold + 1}/{n_folds}")

        X_train, B_train, X_val, B_val = split_val_set(X1, B1, 1/n_folds, seed=fold)

        Z_train, user_means, user_stds = normalize_and_fill_user_movie_matrix(X_train)
        Z_train = Z_train.to_numpy()
        Z_val = normalize_and_fill_set(X_val, user_means, user_stds)

        C = compute_covariance_matrix(Z_train)
        C = torch.tensor(C, dtype=torch.float32).to(device)

        input_dim = Z_train.shape[1]

        Z_train = torch.tensor(Z_train, dtype=torch.float32).to(device)
        Z_val = torch.tensor(Z_val, dtype=torch.float32).to(device)

        gnn_model = MovieLensGNN().to(device)

        optimizer = optim.Adam(gnn_model.parameters(), lr=lr)
        train_losses = []
        val_losses = []

        for epoch in range(n_epochs):
            train_loss = train_epoch(gnn_model, optimizer, Z_train, B_train, loss_fn, batch_size, device)
            train_losses.append(train_loss)

            val_loss = validate_model(gnn_model, Z_train, B_train, Z_val, B_val, loss_fn, batch_size, device)
            val_losses.append(val_loss)

            print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}')

        fold_metrics.append((train_losses, val_losses))