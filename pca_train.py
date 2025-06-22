import torch
import numpy as np
import pandas as pd
import itertools
import random
import time
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from Utils.miscTools import parse_args
from utils.data_preprocessing import load_movielens_data, normalize_and_fill_user_movie_matrix, split_test_set, \
    split_val_set, normalize_and_fill_set, get_pytorch_normalized_inputs_and_targets
from constants import *
from utils.metrics import (
    calc_popularity,
    calc_collab_dissimilarity,
    calc_genre_dissimilarity,
    calc_hybrid_dissimilarity
)
from utils.metrics_evaluation_utils import (
    create_movie_id_mapping,
    load_genre_matrix_for_evaluation,
    validate_dimensions
)


class PCARecommender:
    """PCA-based collaborative filtering recommender"""

    def __init__(self, n_components=50):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.user_means = None
        self.user_stds = None
        self.is_fitted = False

    def fit(self, ratings_matrix, mask_matrix):
        """
        Fit PCA on the ratings matrix

        Args:
            ratings_matrix: (M, U) Movies x Users rating matrix
            mask_matrix: (M, U) Binary mask (1 where ratings exist)
        """
        # Convert to Users x Movies for PCA
        ratings_UxM = ratings_matrix.T  # (U, M)
        mask_UxM = mask_matrix.T  # (U, M)

        # Normalize by user (z-score normalization)
        normalized_ratings = np.zeros_like(ratings_UxM)
        self.user_means = np.zeros(ratings_UxM.shape[0])
        self.user_stds = np.ones(ratings_UxM.shape[0])

        for u in range(ratings_UxM.shape[0]):
            user_ratings = ratings_UxM[u, mask_UxM[u] == 1]
            if len(user_ratings) > 0:
                self.user_means[u] = np.mean(user_ratings)
                self.user_stds[u] = np.std(user_ratings)
                if self.user_stds[u] == 0:
                    self.user_stds[u] = 1.0

                # Normalize user's ratings
                normalized_ratings[u, mask_UxM[u] == 1] = (
                        (ratings_UxM[u, mask_UxM[u] == 1] - self.user_means[u]) / self.user_stds[u]
                )

        # Fill missing values with 0 (mean after normalization)
        self.normalized_ratings_train = normalized_ratings

        # Fit PCA on normalized ratings
        self.pca.fit(normalized_ratings)
        self.is_fitted = True

        return self

    def predict(self, test_features_UxM, test_mask_UxM):
        """
        Make predictions using PCA reconstruction

        Args:
            test_features_UxM: (U, M) User features (normalized ratings with 0 for missing)
            test_mask_UxM: (U, M) Test mask

        Returns:
            predictions: (U, M) Predicted ratings (denormalized)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        # Project to PCA space and reconstruct
        pca_components = self.pca.transform(test_features_UxM)
        reconstructed = self.pca.inverse_transform(pca_components)

        # Denormalize predictions
        predictions = np.zeros_like(reconstructed)
        for u in range(reconstructed.shape[0]):
            predictions[u] = reconstructed[u] * self.user_stds[u] + self.user_means[u]

        return predictions

    def predict_top_n(self, test_features_UxM, test_mask_UxM, N=50):
        """
        Get top-N predictions for each user

        Args:
            test_features_UxM: (U, M) User features
            test_mask_UxM: (U, M) Test mask
            N: Number of top items to return

        Returns:
            top_n_items: List of arrays, each containing top-N movie indices for each user
        """
        predictions = self.predict(test_features_UxM, test_mask_UxM)

        top_n_items = []
        for u in range(predictions.shape[0]):
            # Get indices of top N predicted ratings for this user
            # Exclude items that were already rated (mask == 1)
            available_items = test_mask_UxM[u] == 0
            if available_items.sum() > 0:
                user_preds = predictions[u].copy()
                user_preds[~available_items] = -np.inf  # Exclude already rated items
                top_indices = np.argsort(user_preds)[-N:][::-1]  # Top N in descending order
                top_n_items.append(top_indices)
            else:
                # If no available items, return empty array
                top_n_items.append(np.array([]))

        return top_n_items


def evaluate_beyond_accuracy_pca(model, features_test_UxM, test_mask_MxU, pop_scores, dissim_matrix, N=50, device=None):
    """
    Evaluate beyond-accuracy metrics for PCA model

    Args:
        model: Fitted PCARecommender
        features_test_UxM: (U, M) Test features
        test_mask_MxU: (M, U) Test mask
        pop_scores: (M,) Popularity scores
        dissim_matrix: (M, M) Dissimilarity matrix
        N: Top-N cutoff
        device: Not used for PCA, kept for compatibility

    Returns:
        per_user_diversities, per_user_novelties, mean_diversity, mean_novelty
    """
    # Convert test mask to UxM format
    test_mask_UxM = test_mask_MxU.T

    # Get top-N recommendations for each user
    top_n_items = model.predict_top_n(features_test_UxM, test_mask_UxM, N=N)

    per_user_diversities = []
    per_user_novelties = []

    for u, user_top_items in enumerate(top_n_items):
        if len(user_top_items) == 0:
            continue

        # Calculate diversity (average pairwise dissimilarity)
        if len(user_top_items) > 1:
            diversity = 0.0
            count = 0
            for i in range(len(user_top_items)):
                for j in range(i + 1, len(user_top_items)):
                    diversity += dissim_matrix[user_top_items[i], user_top_items[j]]
                    count += 1
            diversity = diversity / count if count > 0 else 0.0
        else:
            diversity = 0.0

        # Calculate novelty (average negative log popularity)
        novelty = np.mean(-np.log(pop_scores[user_top_items] + 1e-10))/2

        per_user_diversities.append(diversity)
        per_user_novelties.append(novelty)

    mean_diversity = np.mean(per_user_diversities) if per_user_diversities else 0.0
    mean_novelty = np.mean(per_user_novelties) if per_user_novelties else 0.0

    return per_user_diversities, per_user_novelties, mean_diversity, mean_novelty


# Parse arguments (using dummy values for PCA-specific params)
args = parse_args()

seeds = [10, 16, 42, 2025, 12345]  # 5 independent runs
all_results = []

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}.")

# Load data
ratings_csv_file = 'ml-latest-small/ml-100k/u_csv.csv'
movies_csv_path = 'ml-latest-small/ml-100k/movies_transformed.csv'

print("Loading and preprocessing data...")

# ============================================================================
# HYPERPARAMETER GRID SEARCH OVER SEEDS
# ============================================================================
for seed in seeds:
    print(f"\nRunning PCA with Seed: {seed}")

    # Set seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    start_time = time.time()

    # Load and split data (ratings are MxU: Movies x Users)
    ratings_full_MxU, mask_full_MxU = load_movielens_data(ratings_csv_file)

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

    # Convert to PyTorch tensors for compatibility with existing functions
    ratings_full_pt_MxU = torch.tensor(ratings_full_MxU, dtype=torch.float32, device=device)
    mask_full_pt_MxU = torch.tensor(mask_full_MxU, dtype=torch.int, device=device)
    mask_train_pt_MxU = torch.tensor(mask_train_MxU, dtype=torch.int, device=device)
    mask_val_pt_MxU = torch.tensor(mask_val_MxU, dtype=torch.int, device=device)
    mask_test_pt_MxU = torch.tensor(mask_test_MxU, dtype=torch.int, device=device)

    # Create visible ratings (zero out unknown ratings)
    visible_ratings_train_pt_MxU = ratings_full_pt_MxU.clone()
    visible_ratings_train_pt_MxU[mask_train_pt_MxU == 0] = 0

    visible_ratings_test_pt_MxU = ratings_full_pt_MxU.clone()
    visible_ratings_test_pt_MxU[mask_test_pt_MxU == 0] = 0

    # Prepare normalized data for compatibility
    features_train_pt_UxM, targets_train_norm_pt_UxM, mask_train_loss_pt_UxM, \
        train_user_means, train_user_stds = get_pytorch_normalized_inputs_and_targets(
        visible_ratings_train_pt_MxU,
        train_mask_movies_x_users_tensor=mask_train_pt_MxU
    )

    features_test_pt_UxM, _, mask_test_loss_pt_UxM, \
        _, _ = get_pytorch_normalized_inputs_and_targets(
        visible_ratings_test_pt_MxU,
        train_mask_movies_x_users_tensor=mask_test_pt_MxU,
        user_means_for_norm=train_user_means,
        user_stds_for_norm=train_user_stds
    )

    # Original test targets (not normalized)
    targets_test_orig_pt_UxM = torch.tensor(ratings_test_MxU.T, dtype=torch.float32, device=device)

    # ============================================================================
    # TRAIN PCA MODEL
    # ============================================================================
    print("Training PCA model...")

    # Determine number of components (use min of users/movies or a reasonable default)
    n_components = 50
    pca_model = PCARecommender(n_components=50)

    # Fit PCA on training data
    pca_model.fit(visible_ratings_train_pt_MxU.cpu().numpy(), mask_train_pt_MxU.cpu().numpy())

    print(f"PCA fitted with {n_components} components")
    print(f"Explained variance ratio: {pca_model.pca.explained_variance_ratio_[:5]}")  # First 5 components

    # ============================================================================
    # EVALUATE ON TEST SET
    # ============================================================================
    print("Evaluating on test set...")

    # Make predictions
    features_test_np_UxM = features_test_pt_UxM.cpu().numpy()
    predictions_UxM = pca_model.predict(features_test_np_UxM, mask_test_pt_MxU.T.cpu().numpy())

    # Calculate RMSE
    mask_test_np_UxM = mask_test_loss_pt_UxM.cpu().numpy()
    targets_test_np_UxM = targets_test_orig_pt_UxM.cpu().numpy()

    # Only evaluate on test positions
    test_positions = mask_test_np_UxM == 1
    if test_positions.sum() > 0:
        test_rmse = np.sqrt(mean_squared_error(
            targets_test_np_UxM[test_positions],
            predictions_UxM[test_positions]
        ))
    else:
        test_rmse = float('inf')

    print(f"Test RMSE: {test_rmse:.6f}")

    # ============================================================================
    # BEYOND-ACCURACY METRICS EVALUATION
    # ============================================================================
    print("\n" + "=" * 60)
    print("BEYOND-ACCURACY METRICS EVALUATION")
    print("=" * 60)

    # ============================================================================
    # STEP 1: CREATE MOVIE ID MAPPING
    # ============================================================================
    print("\n--- Creating Movie ID Mapping ---")
    ratings_csv_path = 'ml-latest-small/ratings.csv'
    internal_idx_to_movie_id_map = create_movie_id_mapping(ratings_csv_path, M)

    # ============================================================================
    # STEP 2: COMPUTE ITEM POPULARITY
    # ============================================================================
    print("\n--- Computing Item Popularity ---")
    ratings_train_np_MxU = visible_ratings_train_pt_MxU.cpu().numpy()
    p_scores = calc_popularity(ratings_train_np_MxU, method='user_norm')
    print(f"Popularity scores computed. Shape: {p_scores.shape}")
    print(f"Popularity stats - Min: {np.min(p_scores):.4f}, Max: {np.max(p_scores):.4f}, Mean: {np.mean(p_scores):.4f}")

    # ============================================================================
    # STEP 3: COMPUTE COLLABORATIVE DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Collaborative Dissimilarity ---")
    Z_train_feat_MxU = features_train_pt_UxM.T.cpu().numpy()  # Shape: (M, U)
    C_collab = calc_collab_dissimilarity(Z_train_feat_MxU)
    print(f"Collaborative dissimilarity matrix computed. Shape: {C_collab.shape}")
    print(
        f"Collab dissim stats - Min: {np.min(C_collab):.4f}, Max: {np.max(C_collab):.4f}, Mean: {np.mean(C_collab):.4f}")

    # ============================================================================
    # STEP 4: LOAD GENRE DATA AND COMPUTE GENRE DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Genre Dissimilarity ---")
    movies_csv_path = 'ml-latest-small/movies.csv'

    try:
        genre_matrix = load_genre_matrix_for_evaluation(
            movies_csv_path,
            internal_idx_to_movie_id_map,
            M
        )
        C_genre = calc_genre_dissimilarity(genre_matrix, method='cosine')
        print(f"Genre dissimilarity matrix computed. Shape: {C_genre.shape}")
        print(
            f"Genre dissim stats - Min: {np.min(C_genre):.4f}, Max: {np.max(C_genre):.4f}, Mean: {np.mean(C_genre):.4f}")

    except Exception as e:
        print(f"Error loading genre data: {e}")
        print("Using random genre dissimilarity matrix as fallback")
        np.random.seed(42)
        C_genre = np.random.rand(M, M).astype(np.float32)
        C_genre = (C_genre + C_genre.T) / 2
        np.fill_diagonal(C_genre, 0)

    # ============================================================================
    # STEP 5: COMPUTE HYBRID DISSIMILARITY
    # ============================================================================
    print("\n--- Computing Hybrid Dissimilarity ---")
    alpha = 0.5
    C_hybrid = calc_hybrid_dissimilarity(C_collab, C_genre, alpha=alpha)
    print(f"Hybrid dissimilarity matrix computed. Shape: {C_hybrid.shape}")
    print(
        f"Hybrid dissim stats - Min: {np.min(C_hybrid):.4f}, Max: {np.max(C_hybrid):.4f}, Mean: {np.mean(C_hybrid):.4f}")

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
    N = 50

    per_user_diversities, per_user_novelties, mean_diversity, mean_novelty = evaluate_beyond_accuracy_pca(
        pca_model,  # PCA model
        features_test_np_UxM,  # test features (Users x Movies)
        mask_test_pt_MxU.cpu().numpy(),  # test mask (Movies x Users)
        p_scores,  # popularity scores (M,)
        C_hybrid,  # hybrid dissimilarity matrix (M x M)
        N=N,  # top-N cutoff
        device=device  # torch device (not used for PCA)
    )


    print(f"Successfully evaluated beyond-accuracy for {len(per_user_diversities)} users")
    print(f"Mean Diversity@{N}: {mean_diversity:.4f}")
    print(f"Mean Novelty@{N}:   {mean_novelty:.4f}")

    elapsed = time.time() - start_time

    # Store results
    result = {
        'seed': seed,
        'gso_type': '-',  # Not applicable for PCA
        'sparsify': '-',  # Not applicable for PCA
        'rmse': test_rmse,
        'mean_diversity': mean_diversity,
        'mean_novelty': mean_novelty,
        'elapsed_sec': elapsed
    }
    all_results.append(result)

    print(f"\nSeed {seed} completed in {elapsed:.2f} seconds")
    print(f"RMSE: {test_rmse:.6f}, Diversity: {mean_diversity:.4f}, Novelty: {mean_novelty:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "=" * 60)
print("SAVING RESULTS")
print("=" * 60)

df = pd.DataFrame(all_results)
ordered_cols = [
    'seed', 'gso_type', 'sparsify',
    'rmse', 'mean_diversity', 'mean_novelty',
    'elapsed_sec'
]
df = df[[c for c in ordered_cols if c in df.columns]]
df.to_csv('pca_grid_results.csv', index=False)

print("Results saved to 'pca_grid_results.csv'")
print("\nFinal Results Summary:")
print(df.to_string(index=False))

print(f"\nAverage across all seeds:")
print(f"RMSE: {df['rmse'].mean():.6f} ± {df['rmse'].std():.6f}")
print(f"Diversity: {df['mean_diversity'].mean():.4f} ± {df['mean_diversity'].std():.4f}")
print(f"Novelty: {df['mean_novelty'].mean():.4f} ± {df['mean_novelty'].std():.4f}")
print(f"Runtime: {df['elapsed_sec'].mean():.2f} ± {df['elapsed_sec'].std():.2f} seconds")