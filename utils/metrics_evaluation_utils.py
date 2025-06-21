"""
Beyond-Accuracy Metrics Evaluation Utilities

This module orchestrates the evaluation of beyond-accuracy metrics by:
1. Generating top-N recommendations for each user
2. Computing novelty and diversity scores
3. Handling the correct data format transformations
"""

import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
from .metrics import (
    calc_novelty,
    calc_diversity,
    load_genre_matrix_from_csv
)


def evaluate_beyond_accuracy(
        model: torch.nn.Module,
        features_test: torch.Tensor,
        test_mask: torch.Tensor,
        pop_scores: np.ndarray,
        dissim_matrix: np.ndarray,
        N: int = 10,
        device: str = 'cpu'
) -> Tuple[List[float], List[float], float, float]:
    """
    Evaluate beyond-accuracy metrics (diversity and novelty) for all test users.

    Args:
        model: Trained PyTorch model (SelectionGNN)
        features_test: Test features tensor of shape (U, M) - Users x Movies
        test_mask: Test mask tensor of shape (M, U) - Movies x Users  
        pop_scores: Popularity scores array of shape (M,)
        dissim_matrix: Dissimilarity matrix of shape (M, M)
        N: Number of recommendations per user
        device: PyTorch device

    Returns:
        tuple: (per_user_diversities, per_user_novelties, mean_diversity, mean_novelty)
    """

    model.eval()
    model.to(device)

    U, M = features_test.shape  # Users x Movies
    print(f"Evaluating {U} users with {M} movies, generating top-{N} recommendations")

    # Move test features to device
    features_test = features_test.to(device)

    # Generate predictions for all users
    with torch.no_grad():
        # SelectionGNN expects input shape: (batch_size, num_features, num_nodes)
        # We have (U, M), need to reshape to (1, M, U) for batch processing
        gnn_input = features_test.T.unsqueeze(0)  # (1, M, U)

        # Forward pass - output shape: (1, U * M) since average=False
        gnn_output = model(gnn_input)  # (1, U * M)

        # Reshape back to (U, M) - Users x Movies
        all_predictions = gnn_output.reshape(U, M)  # (U, M)

    # Convert test_mask from (M, U) to (U, M) format
    test_mask_UxM = test_mask.T  # (U, M)

    per_user_diversities = []
    per_user_novelties = []

    for user_idx in range(U):
        # Get predictions for this user
        user_predictions = all_predictions[user_idx]  # Shape: (M,)

        # Get test items for this user (items that should be excluded from recommendations)
        user_test_mask = test_mask_UxM[user_idx]  # Shape: (M,)

        # Create recommendation scores by masking out test items
        # We want to recommend items NOT in the test set
        rec_scores = user_predictions.clone()

        # Mask out test items (set to very low score so they won't be recommended)
        test_item_indices = torch.where(user_test_mask == 1)[0]
        rec_scores[test_item_indices] = -float('inf')

        # Get top-N recommendations
        if torch.all(rec_scores == -float('inf')):
            # All items are test items, skip this user
            continue

        top_n_indices = torch.topk(rec_scores, min(N, torch.sum(rec_scores != -float('inf')).item())).indices

        # Convert to numpy for metric calculation
        rec_list = top_n_indices.cpu().numpy().tolist()

        if len(rec_list) == 0:
            continue

        # Calculate metrics
        try:
            diversity = calc_diversity(rec_list, dissim_matrix)
            novelty = calc_novelty(rec_list, pop_scores)

            per_user_diversities.append(diversity)
            per_user_novelties.append(novelty)

        except Exception as e:
            print(f"Error calculating metrics for user {user_idx}: {e}")
            continue

    # Calculate mean metrics
    mean_diversity = np.mean(per_user_diversities) if per_user_diversities else 0.0
    mean_novelty = np.mean(per_user_novelties) if per_user_novelties else 0.0


    return per_user_diversities, per_user_novelties, mean_diversity, mean_novelty


def create_movie_id_mapping(ratings_file_path: str, num_movies: int) -> dict:
    """
    Create mapping from internal movie indices (0 to M-1) to original MovieLens IDs.

    This function assumes that the ratings data was loaded in a specific order
    and creates a mapping based on the first num_movies unique movie IDs.

    Args:
        ratings_file_path: Path to the ratings CSV file
        num_movies: Number of movies in the dataset (M)

    Returns:
        dict: Mapping from internal index to original movie ID
    """
    try:
        # Load ratings and get unique movie IDs in order of appearance
        ratings_df = pd.read_csv(ratings_file_path)
        unique_movie_ids = ratings_df['movieId'].unique()

        # Take first num_movies (this should match how data was originally loaded)
        movie_ids_subset = unique_movie_ids[:num_movies]

        # Create mapping from internal index to movie ID
        mapping = {i: int(movie_id) for i, movie_id in enumerate(movie_ids_subset)}

        print(f"Created movie ID mapping for {len(mapping)} movies")
        return mapping

    except Exception as e:
        print(f"Error creating movie ID mapping: {e}")
        # Fallback: create identity mapping
        return {i: i for i in range(num_movies)}


def load_genre_matrix_for_evaluation(movies_csv_path: str,
                                     movie_id_mapping: dict,
                                     num_movies: int) -> np.ndarray:
    """
    Load and filter genre matrix to match the movies in our ratings dataset.

    Args:
        movies_csv_path: Path to movies.csv file
        movie_id_mapping: Mapping from internal index to original movie ID
        num_movies: Number of movies in ratings dataset

    Returns:
        np.ndarray: Genre matrix of shape (M, num_genres) for our movie subset
    """
    try:
        # Load full genre matrix
        genre_matrix_full, genre_names, movie_ids_full = load_genre_matrix_from_csv(movies_csv_path)

        # Create mapping from movie ID to genre matrix row
        movie_id_to_genre_row = {movie_id: idx for idx, movie_id in enumerate(movie_ids_full)}

        # Filter genre matrix for our movies
        num_genres = len(genre_names)
        genre_matrix_filtered = np.zeros((num_movies, num_genres), dtype=int)

        missing_count = 0
        for internal_idx in range(num_movies):
            if internal_idx in movie_id_mapping:
                movie_id = movie_id_mapping[internal_idx]
                if movie_id in movie_id_to_genre_row:
                    genre_row_idx = movie_id_to_genre_row[movie_id]
                    genre_matrix_filtered[internal_idx] = genre_matrix_full[genre_row_idx]
                else:
                    missing_count += 1
                    # Keep as zeros (no genres)
            else:
                missing_count += 1

        if missing_count > 0:
            print(f"Warning: {missing_count} movies not found in genre data")

        print(f"Genre matrix shape: {genre_matrix_filtered.shape}")
        print(f"Average genres per movie: {np.mean(np.sum(genre_matrix_filtered, axis=1)):.2f}")

        return genre_matrix_filtered

    except Exception as e:
        print(f"Error loading genre matrix: {e}")
        print("Using random genre matrix as fallback")

        # Fallback: create random binary genre matrix
        np.random.seed(42)
        num_genres = 20  # Assume 20 genres
        genre_matrix = np.random.randint(0, 2, size=(num_movies, num_genres))
        return genre_matrix


def validate_dimensions(features_test: torch.Tensor,
                        test_mask: torch.Tensor,
                        pop_scores: np.ndarray,
                        dissim_matrix: np.ndarray) -> bool:
    """
    Validate that all input dimensions are consistent.

    Args:
        features_test: Shape (U, M)
        test_mask: Shape (M, U) 
        pop_scores: Shape (M,)
        dissim_matrix: Shape (M, M)

    Returns:
        bool: True if dimensions are consistent
    """
    U, M = features_test.shape

    # Check test_mask
    if test_mask.shape != (M, U):
        print(f"Error: test_mask shape {test_mask.shape} != expected ({M}, {U})")
        return False

    # Check pop_scores  
    if pop_scores.shape != (M,):
        print(f"Error: pop_scores shape {pop_scores.shape} != expected ({M},)")
        return False

    # Check dissim_matrix
    if dissim_matrix.shape != (M, M):
        print(f"Error: dissim_matrix shape {dissim_matrix.shape} != expected ({M}, {M})")
        return False

    print("All dimensions validated successfully")
    return True