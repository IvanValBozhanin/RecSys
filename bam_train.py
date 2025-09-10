import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
import time
import random
import itertools

from Utils.miscTools import parse_args
from utils.data_preprocessing import load_movielens_data, normalize_and_fill_user_movie_matrix, split_test_set, \
    split_val_set, normalize_and_fill_set, get_pytorch_normalized_inputs_and_targets
from utils.covariance_utils import compute_user_user_covariance_torch, compute_user_user_precision_torch
import Modules.architectures as archit
import Utils.graphML as gml
from constants import *
from utils.testing_utils import test_model
from utils.val_utils import validate_model
from utils.bam_tr_utils import train_epoch
from utils.plot_utils import plot_training_validation_performance
from utils.metrics_evaluation_utils import load_genre_matrix_for_evaluation, create_movie_id_mapping
from utils.metrics import calc_popularity, calc_collab_dissimilarity, calc_genre_dissimilarity, calc_hybrid_dissimilarity
from itertools import product
from utils.metrics import (
    calc_popularity,
    calc_collab_dissimilarity,
    calc_genre_dissimilarity,
    calc_hybrid_dissimilarity
)
from utils.metrics_evaluation_utils import (
    evaluate_beyond_accuracy,
    create_movie_id_mapping,
    load_genre_matrix_for_evaluation,
    validate_dimensions
)


seeds        = [10]#, 16, 42, 2025, 12345]     # 5 independent runs
gso_types    = ["cov"]#, "prec"]                   # covariance or precision
sparsif_opts = ['standard']#, "hard_thr", "soft_thr"] # dense vs hard thr vs soft thr

grid = itertools.product(seeds, gso_types, sparsif_opts)
all_results = []

args = parse_args()

# Set seeds for reproducibility
np.random.seed(seed)
torch.manual_seed(seed)

# Multi-objective training hyperparameters
lambda_rmse = 0.4    # Weight for RMSE loss
lambda_novelty = 0.3  # Weight for novelty loss
lambda_diversity = 0.3  # Weight for diversity loss
# Note: lambda_rmse + lambda_novelty + lambda_diversity should = 1.0

# Recommendation cutoff for BAM computation during training
N_train = 10  # Top-N for computing novelty/diversity during training

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}.")

# Load data
file_path = 'ml-latest-small/ml-100k/u_csv.csv'
movies_csv_path = 'ml-latest-small/ml-100k/movies_transformed.csv'
cov_type = args.cov_type
tau = args.tau


# Load and split data (ratings are MxU: Movies x Users)
ratings_full_MxU, mask_full_MxU = load_movielens_data(file_path)

# Data Splitting
ratings_trval_MxU, mask_trval_MxU, ratings_test_MxU, mask_test_MxU = split_test_set(
    ratings_full_MxU, mask_full_MxU, mask_percentage=1 / 10, seed=seed
)
ratings_train_MxU, mask_train_MxU, ratings_val_MxU, mask_val_MxU = split_val_set(
    ratings_trval_MxU, mask_trval_MxU, mask_percentage=1 / 9, seed=seed
)

# Calculate training set size and dimensions
nTrain = mask_train_MxU.sum()
M, U = ratings_full_MxU.shape  # M = num_movies, U = num_users
print(f"Number of users: {U}, Number of movies: {M}")
print(f"Training samples: {nTrain}")

# Convert to PyTorch tensors
ratings_full_pt_MxU = torch.tensor(ratings_full_MxU, dtype=torch.float32, device=device)
mask_full_pt_MxU = torch.tensor(mask_full_MxU, dtype=torch.int, device=device)
mask_train_pt_MxU = torch.tensor(mask_train_MxU, dtype=torch.int, device=device)
mask_val_pt_MxU = torch.tensor(mask_val_MxU, dtype=torch.int, device=device)
mask_test_pt_MxU = torch.tensor(mask_test_MxU, dtype=torch.int, device=device)

# Create visible ratings (zero out unknown ratings)
visible_ratings_train_pt_MxU = ratings_full_pt_MxU.clone()
visible_ratings_train_pt_MxU[mask_train_pt_MxU == 0] = 0

visible_ratings_val_pt_MxU = ratings_full_pt_MxU.clone()
visible_ratings_val_pt_MxU[mask_val_pt_MxU == 0] = 0

visible_ratings_test_pt_MxU = ratings_full_pt_MxU.clone()
visible_ratings_test_pt_MxU[mask_test_pt_MxU == 0] = 0

# Prepare normalized data
# For training: get normalized features and targets (both UxM: Users x Movies)
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
features_test_pt_UxM, _, mask_test_loss_pt_UxM, \
    _, _ = get_pytorch_normalized_inputs_and_targets(
    visible_ratings_test_pt_MxU,
    train_mask_movies_x_users_tensor=mask_test_pt_MxU,
    user_means_for_norm=train_user_means,
    user_stds_for_norm=train_user_stds
)

# Original test targets (not normalized)
targets_test_orig_pt_UxM = torch.tensor(ratings_test_MxU.T, dtype=torch.float32, device=device)

# Precompute metrics needed for multi-objective training
print("Precomputing popularity and dissimilarity matrices for multi-objective training...")

# Compute popularity scores for training
ratings_train_np_MxU = visible_ratings_train_pt_MxU.cpu().numpy()
p_scores_train = calc_popularity(ratings_train_np_MxU, method='user_norm')
p_scores_train_pt = torch.tensor(p_scores_train, device=device, dtype=torch.float32)

# Compute collaborative dissimilarity
Z_train_feat_MxU = features_train_pt_UxM.T.cpu().numpy()
C_collab_train = calc_collab_dissimilarity(Z_train_feat_MxU)


ratings_csv_path = file_path
internal_idx_to_movie_id_map = create_movie_id_mapping(ratings_csv_path, M)

# Compute genre dissimilarity (you can use your existing genre loading code)
try:
    genre_matrix = load_genre_matrix_for_evaluation(movies_csv_path, internal_idx_to_movie_id_map, M)
    C_genre_train = calc_genre_dissimilarity(genre_matrix, method='cosine')
except:
    print("Using random genre matrix for training")
    np.random.seed(42)
    C_genre_train = np.random.rand(M, M).astype(np.float32)
    C_genre_train = (C_genre_train + C_genre_train.T) / 2
    np.fill_diagonal(C_genre_train, 0)

# Compute hybrid dissimilarity
C_hybrid_train = calc_hybrid_dissimilarity(C_collab_train, C_genre_train, alpha=0.5)
C_hybrid_train_pt = torch.tensor(C_hybrid_train, device=device, dtype=torch.float32)





print("Precomputation complete.")

# ============================================================================
# hyperparameter grid search over the configurations.
for seed, gso, sparse_type in grid:
    print(f"\nRunning with Seed: {seed}, GSO Type: {gso}, Sparsity Type: {sparse_type}")
    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

    start_time = time.time()

    # Compute user-user covariance matrix (UxU)
    # Input: features_train_pt_UxM.T = MxU (Movies x Users) as expected by covariance function
    C_user_user_pt_UxU = compute_user_user_covariance_torch(
        features_train_pt_UxM.T,
        sparse_type,
        thr=tau * torch.tensor(np.sqrt(np.log(U) / nTrain)),
        p=args.p
    ).to(device)


    # print(C_user_user_pt_UxU)
    if gso == "prec":
        C_user_user_pt_UxU = compute_user_user_precision_torch(C_user_user_pt_UxU, U)
    # print(P_user_user_pt_UxU)

    threshold_value = tau * torch.tensor(np.sqrt(np.log(U) / nTrain))
    sparsity = (C_user_user_pt_UxU == 0).sum().item() / C_user_user_pt_UxU.numel()
    print(f"GSO stats - Max: {C_user_user_pt_UxU.max():.4f}, Min: {C_user_user_pt_UxU.min():.4f}")
    print(f"GSO stats - Mean: {C_user_user_pt_UxU.mean():.4f}, Std: {C_user_user_pt_UxU.std():.4f}")
    print(f"Computed GSO with threshold {threshold_value:.4f}, sparsity: {sparsity:.4%}")

    # Hyperparameter search options
    # Note: For SelectionGNN, dimNodeSignals[0] should match the number of input features per node
    # In our case, each user (node) has M movie features, so dimNodeSignals[0] = M
    GNN_dimNodeSignals_options = [
        # [M, 512, 256],  # [num_movies, hidden1, hidden2]
        # [M, 256, 128],
        [M, 1]
    ]
    GNN_numTaps_options = [2]
    MLP_layerDims_options = [
        # [256, M],
        # [512, 512, M],  # [hidden1, hidden2, num_movies]
        [1, 1024, 1024, M]
    ]

    best_hyperparams_tuple = None
    best_val_loss = float('inf')
    train_losses_for_best_model, val_losses_for_best_model = [], []
    best_model_state = None

    for dimNodeSignals, GNN_numTaps, MLP_layerDims in product(
            GNN_dimNodeSignals_options, GNN_numTaps_options, MLP_layerDims_options
    ):
        current_epoch_train_losses, current_epoch_val_losses = [], []
        print(f"\nTraining with hyperparameters:")
        print(f"  GNN Layers: {dimNodeSignals}")
        print(f"  Filter Taps: {GNN_numTaps}")
        print(f"  MLP Layers: {MLP_layerDims}")

        # Create SelectionGNN
        # For GNN layers: we have len(dimNodeSignals) - 1 GNN layers
        nFilterTaps_list = [GNN_numTaps] * (len(dimNodeSignals) - 1)
        nSelectedNodes_list = [U] * len(nFilterTaps_list)  # Keep all users at each layer
        poolingSize_list = [1] * len(nFilterTaps_list)  # No pooling

        gnn_model = archit.SelectionGNN(
            dimNodeSignals=dimNodeSignals,  # [M, hidden1, hidden2, ...] - features per node
            nFilterTaps=nFilterTaps_list,  # [K, K, ...] for each GNN layer
            bias=True,
            nonlinearity=nn.LeakyReLU,
            nSelectedNodes=nSelectedNodes_list,  # [U, U, ...] - keep all users
            poolingFunction=gml.NoPool,
            poolingSize=poolingSize_list,  # [1, 1, ...] - no pooling
            dimLayersMLP=MLP_layerDims,  # [hidden, num_movies] - final MLP layers
            GSO=C_user_user_pt_UxU,  # User-user covariance matrix (U x U)
            average=False  # We want per-node (per-user) output
        )

        optimizer = optim.Adam(gnn_model.parameters(), lr=lr)
        loss_fn = nn.MSELoss(reduction='sum')
        gnn_model.to(device)
        final_epoch_val_loss = float('inf')

        for epoch in range(n_epochs):
            # Training epoch
            epoch_train_loss = train_epoch(
                gnn_model,
                optimizer,
                features_train_pt_UxM,
                targets_train_norm_pt_UxM,
                mask_train_loss_pt_UxM,
                loss_fn,
                batch_size,
                device,
                p_scores_train_pt,
                C_hybrid_train_pt,
                lambda_rmse,
                lambda_novelty,
                lambda_diversity,
                N_train
            )
            current_epoch_train_losses.append(epoch_train_loss)

            # Validation epoch
            final_epoch_val_loss = validate_model(
                gnn_model,
                features_val_pt_UxM,
                targets_val_norm_pt_UxM,
                mask_val_loss_pt_UxM,
                loss_fn,
                batch_size,
                device
            )
            current_epoch_val_losses.append(final_epoch_val_loss)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f'Epoch {epoch + 1}/{n_epochs}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {final_epoch_val_loss:.4f}')

        # Check if this is the best model
        if final_epoch_val_loss < best_val_loss:
            best_val_loss = final_epoch_val_loss
            best_hyperparams_tuple = (dimNodeSignals, GNN_numTaps, MLP_layerDims)
            best_model_state = gnn_model.state_dict().copy()
            train_losses_for_best_model = current_epoch_train_losses
            val_losses_for_best_model = current_epoch_val_losses
            print(f"*** NEW BEST MODEL ***")
            print(f"Hyperparameters: {best_hyperparams_tuple}")
            print(f"Validation Loss: {best_val_loss:.4f}")

    # Plot training curves for best model
    if train_losses_for_best_model:
        plot_training_validation_performance(
            train_losses_for_best_model,
            val_losses_for_best_model,
            len(train_losses_for_best_model)
        )

    # Final evaluation on test set
    print(f"\n=== FINAL EVALUATION ===")
    print(f"Best hyperparameters: {best_hyperparams_tuple}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    # Recreate best model
    best_dimNodeSignals, best_GNN_numTaps, best_MLP_layerDims = best_hyperparams_tuple
    nFilterTaps_list = [best_GNN_numTaps] * (len(best_dimNodeSignals) - 1)
    nSelectedNodes_list = [U] * len(nFilterTaps_list)
    poolingSize_list = [1] * len(nFilterTaps_list)

    gnn_model_best = archit.SelectionGNN(
        dimNodeSignals=best_dimNodeSignals,
        nFilterTaps=nFilterTaps_list,
        bias=True,
        nonlinearity=nn.LeakyReLU,
        nSelectedNodes=nSelectedNodes_list,
        poolingFunction=gml.NoPool,
        poolingSize=poolingSize_list,
        dimLayersMLP=best_MLP_layerDims,
        GSO=C_user_user_pt_UxU,
        average=False
    )

    # Load best model weights
    gnn_model_best.load_state_dict(best_model_state)
    gnn_model_best.to(device)

    # Test the model
    test_rmse = test_model(
        gnn_model_best,
        features_test_pt_UxM,
        targets_test_orig_pt_UxM,
        mask_test_loss_pt_UxM,
        train_user_means.cpu().numpy(),
        train_user_stds.cpu().numpy(),
        device
    )

    # Beyond-accuracy evaluation
    # ============================================================================
    # BEYOND-ACCURACY METRICS EVALUATION
    # ============================================================================
    # This code should be appended to the end of train.py after the test RMSE calculation

    print("\n" + "=" * 60)
    print("BEYOND-ACCURACY METRICS EVALUATION")
    print("=" * 60)

    # Import the required functions
    from utils.metrics import (
        calc_popularity,
        calc_collab_dissimilarity,
        calc_genre_dissimilarity,
        calc_hybrid_dissimilarity
    )
    from utils.metrics_evaluation_utils import (
        evaluate_beyond_accuracy,
        load_genre_matrix_from_csv
    )

    # ============================================================================
    # STEP 1: COMPUTE ITEM POPULARITY
    # ============================================================================
    print("\n--- Computing Item Popularity ---")

    # Use training ratings matrix (M x U format)
    # ratings_train_MxU contains the training ratings with shape (M, U)
    p_scores = calc_popularity(ratings_train_MxU, method='user_norm')
    print(f"Popularity scores computed. Shape: {p_scores.shape}")
    print(f"Popularity stats - Min: {np.min(p_scores):.4f}, Max: {np.max(p_scores):.4f}, Mean: {np.mean(p_scores):.4f}")

    # ============================================================================
    # STEP 2: COMPUTE COLLABORATIVE DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Collaborative Dissimilarity ---")

    # We need z-scored features for collaborative dissimilarity
    # features_train_pt_UxM is (U, M), so we transpose to get (M, U)
    Z_train_feat_MxU = features_train_pt_UxM.T.cpu().numpy()  # Shape: (M, U)

    # Compute collaborative dissimilarity matrix
    C_collab = calc_collab_dissimilarity(Z_train_feat_MxU)
    print(f"Collaborative dissimilarity matrix computed. Shape: {C_collab.shape}")
    print(f"Collab dissim stats - Min: {np.min(C_collab):.4f}, Max: {np.max(C_collab):.4f}, Mean: {np.mean(C_collab):.4f}")

    # ============================================================================
    # STEP 3: LOAD GENRE DATA AND COMPUTE GENRE DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Genre Dissimilarity ---")

    # Load genre matrix from MovieLens movies.csv
    # Update this path to match your data location
    movies_csv_path = 'ml-latest-small/ml-100k/movies_transformed.csv'  # Update path as needed

    try:
        # Load genre matrix
        genre_matrix_full, genre_names, movie_ids_full = load_genre_matrix_from_csv(movies_csv_path)
        print(f"Loaded genre data: {len(movie_ids_full)} movies, {len(genre_names)} genres")

        # Create mapping from internal indices to movie IDs
        internal_idx_to_original_movie_id_map = create_movie_id_mapping(ratings_csv_path, M)
        ratings_movie_ids = [internal_idx_to_original_movie_id_map[i] for i in range(M)]

        # Filter genre matrix to match movies in our ratings matrix
        movie_id_to_genre_idx = {movie_id: idx for idx, movie_id in enumerate(movie_ids_full)}

        # Create genre matrix for our subset of movies
        genre_matrix = np.zeros((M, len(genre_names)), dtype=int)
        missing_movies = []

        for internal_idx, movie_id in enumerate(ratings_movie_ids):
            if movie_id in movie_id_to_genre_idx:
                genre_idx = movie_id_to_genre_idx[movie_id]
                genre_matrix[internal_idx] = genre_matrix_full[genre_idx]
            else:
                missing_movies.append(movie_id)
                # For missing movies, assign no genres (all zeros)

        if missing_movies:
            print(f"Warning: {len(missing_movies)} movies not found in genre data")

        print(f"Genre matrix for ratings subset: {genre_matrix.shape}")
        print(f"Average genres per movie: {np.mean(np.sum(genre_matrix, axis=1)):.2f}")

        # Compute genre dissimilarity matrix
        C_genre = calc_genre_dissimilarity(genre_matrix, method='cosine')
        print(f"Genre dissimilarity matrix computed. Shape: {C_genre.shape}")
        print(f"Genre dissim stats - Min: {np.min(C_genre):.4f}, Max: {np.max(C_genre):.4f}, Mean: {np.mean(C_genre):.4f}")

    except FileNotFoundError:
        print(f"Warning: Could not find genre file at {movies_csv_path}")
        print("Using random genre dissimilarity matrix as fallback")
        # Create random symmetric dissimilarity matrix as fallback
        np.random.seed(42)
        C_genre = np.random.rand(M, M)
        C_genre = (C_genre + C_genre.T) / 2  # Make symmetric
        np.fill_diagonal(C_genre, 0)  # Diagonal should be 0

    except Exception as e:
        print(f"Error loading genre data: {e}")
        print("Using random genre dissimilarity matrix as fallback")
        np.random.seed(42)
        C_genre = np.random.rand(M, M)
        C_genre = (C_genre + C_genre.T) / 2
        np.fill_diagonal(C_genre, 0)

    # ============================================================================
    # STEP 4: COMPUTE HYBRID DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Hybrid Dissimilarity ---")

    # Combine collaborative and genre dissimilarities
    alpha = 0.5  # Equal weighting between collaborative and genre
    C_hybrid = calc_hybrid_dissimilarity(C_collab, C_genre, alpha=alpha)
    print(f"Hybrid dissimilarity matrix computed with alpha={alpha}")
    print(f"Hybrid dissim stats - Min: {np.min(C_hybrid):.4f}, Max: {np.max(C_hybrid):.4f}, Mean: {np.mean(C_hybrid):.4f}")

    # ============================================================================
    # STEP 5: EVALUATE BEYOND-ACCURACY METRICS
    # ============================================================================
    # ============================================================================
    # BEYOND-ACCURACY METRICS EVALUATION
    # ============================================================================
    # This code should be appended to the end of train.py after the test RMSE calculation

    print("\n" + "=" * 60)
    print("BEYOND-ACCURACY METRICS EVALUATION")
    print("=" * 60)

    # Import the required functions


    # ============================================================================
    # STEP 1: CREATE MOVIE ID MAPPING
    # ============================================================================
    print("\n--- Creating Movie ID Mapping ---")

    # Create mapping from internal indices to original MovieLens IDs
    # This is needed for genre data alignment


    # ============================================================================
    # STEP 2: COMPUTE ITEM POPULARITY
    # ============================================================================
    print("\n--- Computing Item Popularity ---")

    # Use training ratings matrix (M x U format)
    # Convert from PyTorch tensor to numpy if needed
    if isinstance(visible_ratings_train_pt_MxU, torch.Tensor):
        ratings_train_np_MxU = visible_ratings_train_pt_MxU.cpu().numpy()
    else:
        ratings_train_np_MxU = visible_ratings_train_pt_MxU

    p_scores = calc_popularity(ratings_train_np_MxU, method='user_norm')
    print(f"Popularity scores computed. Shape: {p_scores.shape}")
    print(f"Popularity stats - Min: {np.min(p_scores):.4f}, Max: {np.max(p_scores):.4f}, Mean: {np.mean(p_scores):.4f}")

    # ============================================================================
    # STEP 3: COMPUTE COLLABORATIVE DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Collaborative Dissimilarity ---")

    # We need z-scored features for collaborative dissimilarity
    # features_train_pt_UxM is (U, M), so we transpose to get (M, U)
    Z_train_feat_MxU = features_train_pt_UxM.T.cpu().numpy()  # Shape: (M, U)

    # Compute collaborative dissimilarity matrix
    C_collab = calc_collab_dissimilarity(Z_train_feat_MxU)
    print(f"Collaborative dissimilarity matrix computed. Shape: {C_collab.shape}")
    print(f"Collab dissim stats - Min: {np.min(C_collab):.4f}, Max: {np.max(C_collab):.4f}, Mean: {np.mean(C_collab):.4f}")

    # ============================================================================
    # STEP 4: LOAD GENRE DATA AND COMPUTE GENRE DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Genre Dissimilarity ---")

    # Load genre matrix from MovieLens movies.csv

    try:
        genre_matrix = load_genre_matrix_for_evaluation(
            movies_csv_path,
            internal_idx_to_movie_id_map,
            M
        )

        # Compute genre dissimilarity matrix
        C_genre = calc_genre_dissimilarity(genre_matrix, method='cosine')
        print(f"Genre dissimilarity matrix computed. Shape: {C_genre.shape}")
        print(f"Genre dissim stats - Min: {np.min(C_genre):.4f}, Max: {np.max(C_genre):.4f}, Mean: {np.mean(C_genre):.4f}")

    except Exception as e:
        print(f"Error loading genre data: {e}")
        print("Using random genre dissimilarity matrix as fallback")
        # Create random symmetric dissimilarity matrix as fallback
        np.random.seed(42)
        C_genre = np.random.rand(M, M).astype(np.float32)
        C_genre = (C_genre + C_genre.T) / 2  # Make symmetric
        np.fill_diagonal(C_genre, 0)  # Diagonal should be 0

    # ============================================================================
    # STEP 5: COMPUTE HYBRID DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Hybrid Dissimilarity ---")

    # Combine collaborative and genre dissimilarities
    alpha = 0.5  # Equal weighting between collaborative and genre
    C_hybrid = calc_hybrid_dissimilarity(C_collab, C_genre, alpha=alpha)
    # ============================================================================
    # STEP 5 (cont’d): COMPUTE HYBRID DISSIMILARITY
    # ============================================================================
    print(f"Hybrid dissimilarity matrix computed. Shape: {C_hybrid.shape}")
    print(f"Hybrid dissim stats - Min: {np.min(C_hybrid):.4f}, Max: {np.max(C_hybrid):.4f}, Mean: {np.mean(C_hybrid):.4f}")

    # ============================================================================
    # STEP 6: VALIDATE INPUT DIMENSIONS
    # ============================================================================
    print("\n--- Validating Input Dimensions for Beyond-Accuracy Evaluation ---")
    if not validate_dimensions(features_test_pt_UxM, mask_test_pt_MxU, p_scores, C_hybrid):
        raise ValueError("Dimension mismatch: check features_test, test_mask, pop_scores or dissim_matrix")
    print("All input dimensions are consistent.")

    # ============================================================================
    # STEP 7: EVALUATE BEYOND-ACCURACY METRICS
    # ============================================================================
    print("\n--- Evaluating Beyond-Accuracy Metrics ---")
    # Choose your cutoff N (e.g., top-10 recommendations)
    N = 50

    per_user_diversities, per_user_novelties, mean_diversity, mean_novelty = evaluate_beyond_accuracy(
        gnn_model,                  # your trained SelectionGNN
        features_test_pt_UxM,       # test features (Users x Movies)
        mask_test_pt_MxU,           # test mask (Movies x Users)
        p_scores,                   # popularity scores (M,)
        C_hybrid,                   # hybrid dissimilarity matrix (M x M)
        N=N,                        # top-N cutoff
        device=device               # torch device
    )

    print(f"Successfully evaluated beyond-accuracy for {len(per_user_diversities)} users")
    print(f"Mean Diversity@{N}: {mean_diversity:.4f}")
    print(f"Mean Novelty@{N}:   {mean_novelty:.4f}")

    elapsed = time.time() - start_time

    result = {
        'seed': seed,
        'gso_type': gso,
        'sparsify': sparse_type,
        'rmse': test_rmse,  # from your test_model call
        'mean_diversity': mean_diversity,  # from evaluate_beyond_accuracy
        'mean_novelty': mean_novelty,  # from evaluate_beyond_accuracy
        'elapsed_sec': elapsed  # from your time.time() wrap
    }
    all_results.append(result)

    df = pd.DataFrame(all_results)
    ordered_cols = [
        'seed', 'gso_type', 'sparsify',
        'rmse', 'mean_diversity', 'mean_novelty',
        'elapsed_sec'
    ]
    df = df[[c for c in ordered_cols if c in df.columns]]
    df.to_csv('bam_grid_results.csv', index=False)