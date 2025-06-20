import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd

from Utils.miscTools import parse_args
from utils.data_preprocessing import load_movielens_data, normalize_and_fill_user_movie_matrix, split_test_set, \
    split_val_set, normalize_and_fill_set, get_pytorch_normalized_inputs_and_targets
from utils.covariance_utils import compute_user_user_covariance_torch
from models.gnn_model import MovieLensGNN
from constants import *
from utils.testing_utils import test_model
from utils.val_utils import validate_model
from utils.training_utils import train_epoch
from utils.plot_utils import plot_training_validation_performance
from itertools import product
from utils.metrics_evaluation_utils import evaluate_beyond_accuracy
import Modules.architectures as archit
import Utils.graphML as gml


args = parse_args()

#TODO: set the seed for torch for replicability. But RUN WITHOUT SEED!
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}.")

# file_path = 'ml-latest-small/ratings.csv'
file_path = 'ml-latest-small/ml-100k/u_csv.csv'


cov_type = args.cov_type
tau = args.tau

ratings_full_MxU, mask_full_MxU  = load_movielens_data(file_path)

# Data Splitting (NumPy)
ratings_trval_MxU, mask_trval_MxU, ratings_test_MxU, mask_test_MxU = split_test_set(
    ratings_full_MxU, mask_full_MxU, mask_percentage=1 / 10, seed=seed
)
ratings_train_MxU, mask_train_MxU, ratings_val_MxU, mask_val_MxU = split_val_set(
    ratings_trval_MxU, mask_trval_MxU, mask_percentage=1 / 9, seed=seed
)

# Calculate nTrain - number of samples in training set
# M = number of movies
# U = number of users
nTrain = mask_train_MxU.sum()
M, U = ratings_full_MxU.shape
print(f"Number of users: {U}, Number of movies: {M}")

# Convert to PyTorch Tensors
ratings_full_pt_MxU = torch.tensor(ratings_full_MxU, dtype=torch.float32, device=device)
mask_full_pt_MxU = torch.tensor(mask_full_MxU, dtype=torch.int, device=device)

mask_train_pt_MxU = torch.tensor(mask_train_MxU, dtype=torch.int, device=device)
mask_val_pt_MxU = torch.tensor(mask_val_MxU, dtype=torch.int, device=device)
mask_test_pt_MxU = torch.tensor(mask_test_MxU, dtype=torch.int, device=device)

visible_ratings_train_pt_MxU = ratings_full_pt_MxU.clone()
visible_ratings_train_pt_MxU[mask_train_pt_MxU == 0] = 0

visible_ratings_val_pt_MxU = ratings_full_pt_MxU.clone()
visible_ratings_val_pt_MxU[mask_val_pt_MxU == 0] = 0

visible_ratings_test_pt_MxU = ratings_full_pt_MxU.clone()
visible_ratings_test_pt_MxU[mask_test_pt_MxU == 0] = 0

# --- Prepare Normalized Data using PyTorch function ---
# For training features and targets:
features_train_pt_UxM, targets_train_norm_pt_UxM, mask_train_loss_pt_UxM, \
train_user_means, train_user_stds = get_pytorch_normalized_inputs_and_targets(
    visible_ratings_train_pt_MxU,
    train_mask_movies_x_users_tensor=mask_train_pt_MxU
)

# For validation:
features_val_pt_UxM, targets_val_norm_pt_UxM, mask_val_loss_pt_UxM, \
_, _ = get_pytorch_normalized_inputs_and_targets(
    visible_ratings_val_pt_MxU,
    train_mask_movies_x_users_tensor=mask_val_pt_MxU,
    user_means_for_norm=train_user_means,
    user_stds_for_norm=train_user_stds
)

# For testing:
featuers_test_pt_UxM, _, mask_test_pt_UxM, \
_, _ = get_pytorch_normalized_inputs_and_targets(
    visible_ratings_test_pt_MxU,
    train_mask_movies_x_users_tensor=mask_test_pt_MxU,
    user_means_for_norm=train_user_means,
    user_stds_for_norm=train_user_stds
)

targets_test_orig_pt_UxM = torch.tensor(ratings_test_MxU.T, dtype=torch.float32, device=device)


# --- GSO Calculation ---
# Use Z_train_feat_users_x_movies (already users x movies, normalized, 0-filled)
# but compute_user_user_covariance_torch expects movies x users input.
C_user_user_pt_UxU = compute_user_user_covariance_torch(features_train_pt_UxM.T, cov_type, thr =tau * torch.tensor(np.sqrt(np.log(U) / nTrain)), p = args.p).to(device)
# C = torch.full_like(C, 0)

threshold_value = tau * torch.tensor(np.sqrt(np.log(U) / nTrain))
sparsity = (C_user_user_pt_UxU == 0).sum().item() / C_user_user_pt_UxU.numel()
print(C_user_user_pt_UxU.max(), C_user_user_pt_UxU.min(), C_user_user_pt_UxU.mean(), C_user_user_pt_UxU.std())

print(f"Computed GSO with threshold {threshold_value:.4f}, sparsity: {sparsity:.4%}")

# --- Hyperparameter Search Loop (largely unchanged from previous good version) ---
GNN_dimNodeSignals_options = [
    # [num_movies, 256, 512],
    # [num_movies, 128, 512],
    [M, 1024, 1024]
]
GNN_numTaps_options = [[2, 2]]
MLP_layerDims_options = [[1024, 1024, M]]

best_hyperparams_tuple = (GNN_dimNodeSignals_options[0], GNN_numTaps_options[0], MLP_layerDims_options[0])
best_val_loss = float('inf')
train_losses_for_best_model, val_losses_for_best_model = [], []

for dimNodeSignals, GNN_numTaps, MLP_layerDims in product(GNN_dimNodeSignals_options, GNN_numTaps_options, MLP_layerDims_options):
    current_epoch_train_losses, current_epoch_val_losses = [], []
    print(f"Training with hyperparameters: GNN Layers: {dimNodeSignals}, Taps: {GNN_numTaps}, MLP Layers: {MLP_layerDims}")

    # gnn_model = MovieLensGNN(C_user_user_pt_UxU, M, dimNodeSignals, GNN_numTaps, MLP_layerDims, U).to(device)
    gnn_model = archit.SelectionGNN(dimNodeSignals,
                                    GNN_numTaps,
                                    True,
                                    nn.LeakyReLU,
                                    [U] * len(GNN_numTaps),
                                    gml.NoPool,
                                    [1] * len(GNN_numTaps),
                                    MLP_layerDims,
                                    C_user_user_pt_UxU
                                    )
    optimizer = optim.Adam(gnn_model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction='sum')

    final_epoch_val_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_train_loss = train_epoch(
                                        gnn_model,
                                        optimizer,
                                        features_train_pt_UxM,
                                        targets_train_norm_pt_UxM,
                                        mask_train_loss_pt_UxM,
                                        loss_fn, batch_size, device
                                        )
        current_epoch_train_losses.append(epoch_train_loss)

        final_epoch_val_loss = validate_model(
                                    gnn_model,
                                    features_val_pt_UxM,
                                    targets_val_norm_pt_UxM,
                                    mask_val_loss_pt_UxM,
                                    loss_fn, batch_size, device
                                    )
        current_epoch_val_losses.append(final_epoch_val_loss)
        print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {final_epoch_val_loss:.4f}')

    if final_epoch_val_loss < best_val_loss:
        best_val_loss = final_epoch_val_loss
        best_hyperparams_tuple = (dimNodeSignals, GNN_numTaps, MLP_layerDims)
        torch.save(gnn_model.state_dict(), 'best_model.pth')
        train_losses_for_best_model = current_epoch_train_losses
        val_losses_for_best_model = current_epoch_val_losses
        print(f"NEW BEST: Hyperparameters: {best_hyperparams_tuple} with Val Loss: {best_val_loss:.4f}")

if train_losses_for_best_model:
    plot_training_validation_performance(train_losses_for_best_model, val_losses_for_best_model, len(train_losses_for_best_model))

print(f"\nLoading best model with hyperparameters: {best_hyperparams_tuple}")
gnn_model_best = MovieLensGNN(C_user_user_pt_UxU, M,
                              best_hyperparams_tuple[0],
                              best_hyperparams_tuple[1],
                              best_hyperparams_tuple[2],
                              U).to(device)
gnn_model_best.load_state_dict(torch.load('best_model.pth'))

test_model(gnn_model_best,
           featuers_test_pt_UxM,
           targets_test_orig_pt_UxM,
           mask_test_pt_UxM,
           train_user_means.cpu().numpy(),
           train_user_stds.cpu().numpy(),
           device)

# --- Beyond-Accuracy Metric Evaluation ---
print("\n--- Preparing for Beyond-Accuracy Evaluation ---")
# Create mappings from original MovieLens IDs to array indices and vice-versa
# user_movie_matrix_df_orig is movies_as_rows (index), users_as_columns from load_movielens_data

#user_movie_matrix_df_orig is the original dataframe
user_movie_matrix_df_orig = pd.DataFrame(ratings_full_MxU, columns=range(ratings_full_MxU.shape[1]))


# Movie ID mappings:
# These are the original MovieLens movieIDs present in your ratings data
movie_ids_in_ratings_df = sorted(user_movie_matrix_df_orig.index.tolist())
# Create a continuous 0-based index for movies based on their order in your matrix
# This internal_idx is what your model's output columns correspond to.
internal_idx_to_original_movie_id_map = {i: mid for i, mid in enumerate(movie_ids_in_ratings_df)}
# original_movie_id_to_internal_idx_map = {mid: i for i, mid in enumerate(movie_ids_in_ratings_df)} # If needed

# User ID mappings (assuming user_array_idx used in evaluate_beyond_accuracy
# directly corresponds to columns of original movies_x_users matrices)
# user_ids_in_ratings_df = sorted(user_movie_matrix_df_orig.columns.tolist())
# internal_user_idx_to_original_user_id_map = {i: uid for i, uid in enumerate(user_ids_in_ratings_df)}

eval_user_array_indices = np.where(mask_test_MxU.sum(axis=0) > 0)[0]

if len(eval_user_array_indices) == 0:
    print("No users found with items in the test set for beyond-accuracy evaluation.")
else:
    evaluate_beyond_accuracy(
        model=gnn_model_best,
        X_features_all_users_pt=Z_train_feat_users_x_movies.to(device),
        eval_user_array_indices=eval_user_array_indices,
        B_train_history_mask_movies_x_users_np=B_train_known_movies_x_users_np,
        X_eval_targets_orig_movies_x_users_np=X_test_orig_movies_x_users_np,
        B_eval_target_mask_movies_x_users_np=B_test_movies_x_users_np,
        idx_to_movie_id_map=internal_idx_to_original_movie_id_map, # USE THIS
        top_n=TOP_N_RECOMMENDATIONS,
        device=device
    )