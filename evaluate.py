import torch
from utils.data_preprocessing import load_movielens_data, split_data_by_items
from utils.covariance_utils import compute_covariance_matrix
from models.gnn_model import MovieLensGNN
import torch.nn as nn


# TODO: pass the done model into this file

# file_path = 'data/movielens.csv'
# user_movie_matrix_filled = load_movielens_data(file_path)
# X_train, X_val, X_test = split_data_by_items(user_movie_matrix_filled)
# cov_matrix = compute_covariance_matrix(X_train)
# input_dim = X_train.shape[1]
# gnn_model = MovieLensGNN(cov_matrix, input_dim)
# loss_fn = nn.MSELoss()
#
# gnn_model.eval()
# with torch.no_grad():
#     y_val_hat = gnn_model(X_val)
#     val_loss = loss_fn(y_val_hat, X_val)
#     print(f'Validation Loss: {val_loss.item()}')
#
# # Evaluation on test set
# with torch.no_grad():
#     y_test_hat = gnn_model(X_test)
#     test_loss = loss_fn(y_test_hat, X_test)
#     print(f'Test Loss: {test_loss.item()}')
