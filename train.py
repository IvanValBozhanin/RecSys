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
from utils.training_utils import train_epoch_full_graph
from utils.plot_utils import plot_training_validation_performance
from itertools import product
from utils.metrics_evaluation_utils import evaluate_beyond_accuracy

args = parse_args()

#TODO: set the seed for torch for replicability. But RUN WITHOUT SEED!
# np.random.seed(seed)
# torch.manual_seed(seed)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}.")

file_path = 'ml-latest-small/ratings.csv'

cov_type = args.cov_type
tau = args.tau

X0_movies_x_users_np, B0_movies_x_users_np = load_movielens_data(file_path)

# Data Splitting (NumPy)
X1_movies_x_users_np, B1_movies_x_users_np, X_test_orig_movies_x_users_np, B_test_movies_x_users_np = split_test_set(
    X0_movies_x_users_np, B0_movies_x_users_np, mask_percentage=1/10, seed=seed
)
X_train_known_movies_x_users_np, B_train_known_movies_x_users_np, X_val_orig_movies_x_users_np, B_val_movies_x_users_np = split_val_set(
    X1_movies_x_users_np, B1_movies_x_users_np, mask_percentage=1/9, seed=seed
)

# Calculate nTrain - number of samples in training set
nTrain = B_train_known_movies_x_users_np.sum()
m = X0_movies_x_users_np.shape[1]  # Number of users

# Convert to PyTorch Tensors
X0_full_pt = torch.tensor(X0_movies_x_users_np, dtype=torch.float32, device=device)
B0_full_mask_pt = torch.tensor(B0_movies_x_users_np, dtype=torch.int, device=device)

B_train_active_mask_pt = torch.tensor(B_train_known_movies_x_users_np, dtype=torch.int, device=device)
B_val_active_mask_pt = torch.tensor(B_val_movies_x_users_np, dtype=torch.int, device=device)
B_test_active_mask_pt = torch.tensor(B_test_movies_x_users_np, dtype=torch.int, device=device)

X_only_train_visible = X0_full_pt.clone()
X_only_train_visible[B_train_active_mask_pt == 0] = 0

X_only_val_visible = X0_full_pt.clone()
X_only_val_visible[B_val_active_mask_pt == 0] = 0

X_only_test_visible = X0_full_pt.clone()
X_only_test_visible[B_test_active_mask_pt == 0] = 0

# --- Prepare Normalized Data using PyTorch function ---
# For training features and targets:
Z_train_feat_users_x_movies, Y_train_targets_users_x_movies, B_train_loss_users_x_movies, \
train_user_means, train_user_stds = get_pytorch_normalized_inputs_and_targets(
    X_only_train_visible, train_mask_movies_x_users_tensor=B_train_active_mask_pt # Use B_train_active_mask_pt to define known for stats
)

# For validation:
Z_val_feat_users_x_movies, Y_val_targets_users_x_movies, B_val_loss_users_x_movies, \
_, _ = get_pytorch_normalized_inputs_and_targets(
    X_only_val_visible, train_mask_movies_x_users_tensor=B_val_active_mask_pt, # Use B_val_active_mask_pt for its known ratings
    user_means_for_norm=train_user_means, user_stds_for_norm=train_user_stds
)

# For testing:
Z_test_feat_users_x_movies, _, B_test_loss_users_x_movies, \
_, _ = get_pytorch_normalized_inputs_and_targets(
    X_only_test_visible, train_mask_movies_x_users_tensor=B_test_active_mask_pt, # Use B_test_active_mask_pt for its known ratings
    user_means_for_norm=train_user_means, user_stds_for_norm=train_user_stds
)
# Test targets remain in original scale (from X_test_orig_movies_x_users_np)
X_test_targets_orig_users_x_movies = torch.tensor(X_test_orig_movies_x_users_np.T, dtype=torch.float32, device=device)

sparsification_value = 1e-8  # Value to fill in for sparsification

# --- GSO Calculation ---
# Use Z_train_feat_users_x_movies (already users x movies, normalized, 0-filled)
# but compute_user_user_covariance_torch expects movies x users input.
C = compute_user_user_covariance_torch(Z_train_feat_users_x_movies.T, cov_type, 
                                       thr = tau * torch.tensor(np.sqrt(np.log(m) / nTrain)), 
                                       p = args.p, 
                                       sparsification_value=sparsification_value).to(device)
# C = torch.full_like(C, 0)

threshold_value = tau * torch.tensor(np.sqrt(np.log(m) / nTrain))
sparsity = (torch.abs(C) <= sparsification_value).sum().item() / C.numel()
print(f"C stats - Max: {C.max().item():.4f}, Min: {C.min().item():.4f}, Mean: {C.mean().item():.4f}, Std: {C.std().item():.4f}")

print(f"Computed GSO with threshold {threshold_value:.4f}, sparsity: {sparsity:.4%}")
print(f"sparcity: {sparsity:.4}")

num_users, num_movies = Z_train_feat_users_x_movies.shape
print(f"Number of users: {num_users}, Number of movies: {num_movies}")

# --- Hyperparameter Search Loop (largely unchanged from previous good version) ---
dimNodeSignals_options = [
    # [num_movies, 256, 512],
    # [num_movies, 128, 512],
    [num_movies, 512, 512]
]
nTaps_options = [2]
dimLayersMLP_options = [[512, 1024, num_movies]]

best_hyperparams_tuple = (dimNodeSignals_options[0], nTaps_options[0], dimLayersMLP_options[0])
best_val_loss = float('inf')
train_losses_for_best_model, val_losses_for_best_model = [], []

for dimNodeSignals, nTaps, dimLayersMLP in product(dimNodeSignals_options, nTaps_options, dimLayersMLP_options):
    current_epoch_train_losses, current_epoch_val_losses = [], []
    print(f"Training with hyperparameters: GNN Layers: {dimNodeSignals}, Taps: {nTaps}, MLP Layers: {dimLayersMLP}")

    gnn_model = MovieLensGNN(C, num_movies, dimNodeSignals, nTaps, dimLayersMLP, num_users).to(device)
    optimizer = optim.Adam(gnn_model.parameters(), lr=lr)
    class RMSELoss(nn.Module):
        def __init__(self, reduction='sum'):
            super().__init__()
            self.mse = nn.MSELoss(reduction=reduction)
            
        def forward(self, yhat, y):
            return torch.sqrt(self.mse(yhat, y))

    loss_fn = RMSELoss(reduction='sum')
    # loss_fn = nn.MSELoss(reduction='sum')


    final_epoch_val_loss = float('inf')

    for epoch in range(n_epochs):
        epoch_train_loss = train_epoch_full_graph(
                                        gnn_model, optimizer,
                                        Z_train_feat_users_x_movies,
                                        Y_train_targets_users_x_movies,
                                        B_train_loss_users_x_movies, # This is the mask of *known training ratings*
                                        loss_fn, batch_size, device
                                        )
        current_epoch_train_losses.append(epoch_train_loss)

        final_epoch_val_loss = validate_model(
                                    gnn_model,
                                    Z_val_feat_users_x_movies, # Input features for val users
                                    Y_val_targets_users_x_movies, # Ground truth for val users
                                    B_val_loss_users_x_movies, # Mask of *known validation ratings*
                                    loss_fn, batch_size, device
                                    )
        current_epoch_val_losses.append(final_epoch_val_loss)
        print(f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {final_epoch_val_loss:.4f}')

    if final_epoch_val_loss < best_val_loss:
        best_val_loss = final_epoch_val_loss
        best_hyperparams_tuple = (dimNodeSignals, nTaps, dimLayersMLP)
        torch.save(gnn_model.state_dict(), 'best_model.pth')
        train_losses_for_best_model = current_epoch_train_losses
        val_losses_for_best_model = current_epoch_val_losses
        print(f"NEW BEST: Hyperparameters: {best_hyperparams_tuple} with Val Loss: {best_val_loss:.4f}")

if train_losses_for_best_model:
    plot_training_validation_performance(train_losses_for_best_model, val_losses_for_best_model, len(train_losses_for_best_model))

print(f"\nLoading best model with hyperparameters: {best_hyperparams_tuple}")
gnn_model_best = MovieLensGNN(C, num_movies,
                              best_hyperparams_tuple[0],
                              best_hyperparams_tuple[1],
                              best_hyperparams_tuple[2],
                              num_users).to(device)
gnn_model_best.load_state_dict(torch.load('best_model.pth'))

test_model(gnn_model_best,
           Z_test_feat_users_x_movies, # Input features for test (normalized train context)
           X_test_targets_orig_users_x_movies, # Original scale targets
           B_test_loss_users_x_movies, # Mask of *known test ratings*
           train_user_means.cpu().numpy(), # Ensure numpy for denormalization
           train_user_stds.cpu().numpy(),  # Ensure numpy for denormalization
           device)

# --- Beyond-Accuracy Metric Evaluation ---
print("\n--- Preparing for Beyond-Accuracy Evaluation ---")
# Create mappings from original MovieLens IDs to array indices and vice-versa
# user_movie_matrix_df_orig is movies_as_rows (index), users_as_columns from load_movielens_data

#user_movie_matrix_df_orig is the original dataframe
user_movie_matrix_df_orig = pd.DataFrame(X0_movies_x_users_np, columns=range(X0_movies_x_users_np.shape[1]))


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

eval_user_array_indices = np.where(B_test_movies_x_users_np.sum(axis=0) > 0)[0]

if len(eval_user_array_indices) == 0:
    print("No users found with items in the test set for beyond-accuracy evaluation.")
# else:
    # evaluate_beyond_accuracy(
    #     model=gnn_model_best,
    #     X_features_all_users_pt=Z_train_feat_users_x_movies.to(device),
    #     eval_user_array_indices=eval_user_array_indices,
    #     B_train_history_mask_movies_x_users_np=B_train_known_movies_x_users_np,
    #     X_eval_targets_orig_movies_x_users_np=X_test_orig_movies_x_users_np,
    #     B_eval_target_mask_movies_x_users_np=B_test_movies_x_users_np,
    #     idx_to_movie_id_map=internal_idx_to_original_movie_id_map, # USE THIS
    #     top_n=TOP_N_RECOMMENDATIONS,
    #     device=device
    # )