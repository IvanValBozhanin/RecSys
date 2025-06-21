"""
Beyond-Accuracy Metrics Implementation

This module implements the exact mathematical definitions for:
1. Popularity: p(i) = #{u: r_{u,i} observed} / U  or  p(i) = #{u: r_{u,i} observed} / max_j #{u: r_{u,j}}
2. Collaborative dissimilarity: d_collab(i,j) = 1 - (sum_u r̃_{u,i} * r̃_{u,j}) / (||r̃_{·,i}|| * ||r̃_{·,j}||)
3. Genre dissimilarity: d_genre(i,j) = 1 - (g_i · g_j) / (||g_i|| * ||g_j||) or Jaccard
4. Hybrid distance: d_α(i,j) = α * d_collab(i,j) + (1-α) * d_genre(i,j)
5. Novelty: novelty(R) = (1/|R|) * sum_{i∈R} (1 - p(i))
6. Diversity: diversity(R) = (1/C(|R|,2)) * sum_{i<j} d_α(i,j)
"""

import numpy as np
import pandas as pd
from typing import Optional, Literal


def calc_popularity(rating_matrix: np.ndarray,
                    method: Literal['user_norm', 'max_norm'] = 'user_norm') -> np.ndarray:
    """
    Calculate item popularity scores.

    Mathematical definition:
    - user_norm: p(i) = #{u: r_{u,i} observed} / U
    - max_norm: p(i) = #{u: r_{u,i} observed} / max_j #{u: r_{u,j}}

    Args:
        rating_matrix: np.ndarray of shape (M, U) where M=movies, U=users
                      Non-zero entries indicate observed ratings
        method: 'user_norm' for normalization by total users, 
               'max_norm' for normalization by max item popularity

    Returns:
        np.ndarray of shape (M,) with popularity scores in [0, 1]
    """
    M, U = rating_matrix.shape

    # Count number of users who rated each item (non-zero entries)
    item_counts = np.count_nonzero(rating_matrix, axis=1)  # Shape: (M,)

    if method == 'user_norm':
        # Normalize by total number of users: p(i) = count(i) / U
        popularity_scores = item_counts / U
    elif method == 'max_norm':
        # Normalize by maximum item count: p(i) = count(i) / max_j count(j)
        max_count = np.max(item_counts) if np.max(item_counts) > 0 else 1
        popularity_scores = item_counts / max_count
    else:
        raise ValueError(f"Unknown method: {method}")

    return popularity_scores.astype(np.float32)


def calc_collab_dissimilarity(Z_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate collaborative dissimilarity matrix from z-scored ratings.

    Mathematical definition:
    d_collab(i,j) = 1 - (sum_u r̃_{u,i} * r̃_{u,j}) / (||r̃_{·,i}|| * ||r̃_{·,j}||)

    where r̃ are z-scored or mean-centered ratings.

    Args:
        Z_matrix: np.ndarray of shape (M, U) with z-scored ratings
                 Missing ratings should be 0 or NaN

    Returns:
        np.ndarray of shape (M, M) with collaborative dissimilarity matrix
        Values in [0, 2], where 0 = identical, 2 = completely opposite
    """
    M, U = Z_matrix.shape

    # Handle NaN values by setting them to 0
    Z_clean = np.nan_to_num(Z_matrix, nan=0.0)

    # Compute cosine similarity matrix
    # Numerator: Z @ Z.T gives sum_u r̃_{u,i} * r̃_{u,j} for all pairs (i,j)
    numerator = Z_clean @ Z_clean.T  # Shape: (M, M)

    # Denominator: ||r̃_{·,i}|| * ||r̃_{·,j}|| for all pairs
    norms = np.linalg.norm(Z_clean, axis=1)  # Shape: (M,)
    denominator = np.outer(norms, norms)  # Shape: (M, M)

    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-8, denominator)

    # Cosine similarity matrix
    cosine_sim = numerator / denominator

    # Clamp to [-1, 1] to handle numerical errors
    cosine_sim = np.clip(cosine_sim, -1.0, 1.0)

    # Convert to dissimilarity: d = 1 - sim
    dissimilarity = 1.0 - cosine_sim

    # Ensure diagonal is 0 (item is identical to itself)
    np.fill_diagonal(dissimilarity, 0.0)

    return dissimilarity.astype(np.float32)


def calc_genre_dissimilarity(genre_matrix: np.ndarray,
                             method: Literal['cosine', 'jaccard'] = 'cosine') -> np.ndarray:
    """
    Calculate genre-based dissimilarity matrix.

    Mathematical definitions:
    - cosine: d_genre(i,j) = 1 - (g_i · g_j) / (||g_i|| * ||g_j||)
    - jaccard: d_jaccard(i,j) = 1 - |g_i ∩ g_j| / |g_i ∪ g_j|

    Args:
        genre_matrix: np.ndarray of shape (M, G) where M=movies, G=genres
                     Binary matrix with 1 indicating movie has genre
        method: 'cosine' for cosine dissimilarity, 'jaccard' for Jaccard dissimilarity

    Returns:
        np.ndarray of shape (M, M) with genre dissimilarity matrix
        Values in [0, 1] where 0 = identical genres, 1 = no common genres
    """
    M, G = genre_matrix.shape

    if method == 'cosine':
        # Compute cosine similarity matrix
        numerator = genre_matrix @ genre_matrix.T  # Shape: (M, M)
        norms = np.linalg.norm(genre_matrix, axis=1)  # Shape: (M,)
        denominator = np.outer(norms, norms)  # Shape: (M, M)

        # Avoid division by zero
        denominator = np.where(denominator == 0, 1e-8, denominator)

        cosine_sim = numerator / denominator
        cosine_sim = np.clip(cosine_sim, 0.0, 1.0)  # Genre vectors are non-negative

        dissimilarity = 1.0 - cosine_sim

    elif method == 'jaccard':
        # Compute Jaccard dissimilarity
        dissimilarity = np.zeros((M, M), dtype=np.float32)

        for i in range(M):
            for j in range(i, M):
                g_i = genre_matrix[i]
                g_j = genre_matrix[j]

                # Intersection: |g_i ∩ g_j|
                intersection = np.sum(g_i * g_j)

                # Union: |g_i ∪ g_j|
                union = np.sum((g_i + g_j) > 0)

                if union == 0:
                    jaccard_sim = 0.0  # Both items have no genres
                else:
                    jaccard_sim = intersection / union

                jaccard_dissim = 1.0 - jaccard_sim
                dissimilarity[i, j] = jaccard_dissim
                dissimilarity[j, i] = jaccard_dissim  # Symmetric
    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure diagonal is 0
    np.fill_diagonal(dissimilarity, 0.0)

    return dissimilarity.astype(np.float32)


def calc_hybrid_dissimilarity(collab_dissim: np.ndarray,
                              genre_dissim: np.ndarray,
                              alpha: float) -> np.ndarray:
    """
    Calculate hybrid dissimilarity matrix combining collaborative and genre information.

    Mathematical definition:
    d_α(i,j) = α * d_collab(i,j) + (1-α) * d_genre(i,j)

    Args:
        collab_dissim: np.ndarray of shape (M, M) with collaborative dissimilarity
        genre_dissim: np.ndarray of shape (M, M) with genre dissimilarity  
        alpha: float in [0, 1] controlling the weight of collaborative vs genre
               α=1: pure collaborative, α=0: pure genre-based

    Returns:
        np.ndarray of shape (M, M) with hybrid dissimilarity matrix
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"Alpha must be in [0, 1], got {alpha}")

    if collab_dissim.shape != genre_dissim.shape:
        raise ValueError(f"Dissimilarity matrices must have same shape: "
                         f"{collab_dissim.shape} vs {genre_dissim.shape}")

    # Linear combination: d_α = α * d_collab + (1-α) * d_genre
    hybrid_dissim = alpha * collab_dissim + (1.0 - alpha) * genre_dissim

    # Ensure diagonal is 0
    np.fill_diagonal(hybrid_dissim, 0.0)

    return hybrid_dissim.astype(np.float32)


def calc_novelty(rec_list: list, pop_scores: np.ndarray) -> float:
    """
    Calculate novelty as the average popularity complement.

    Mathematical definition:
    novelty(R) = (1/|R|) * sum_{i∈R} (1 - p(i))

    Args:
        rec_list: List of recommended item indices
        pop_scores: np.ndarray of popularity scores for all items

    Returns:
        float: Novelty score in [0, 1] where 1 = maximum novelty (all unpopular items)
    """
    if not rec_list:
        return 0.0

    # Sum of (1 - popularity) for recommended items
    novelty_sum = sum(1.0 - pop_scores[item_idx] for item_idx in rec_list)

    return novelty_sum / len(rec_list)


def calc_diversity(rec_list: list, dissim_matrix: np.ndarray) -> float:
    """
    Calculate diversity as average pairwise dissimilarity.

    Mathematical definition:  
    diversity(R) = (1/C(|R|,2)) * sum_{i<j} d_α(i,j)

    where C(|R|,2) = |R| * (|R|-1) / 2 is the number of unique pairs.

    Args:
        rec_list: List of recommended item indices
        dissim_matrix: np.ndarray of shape (M, M) with dissimilarity values

    Returns:
        float: Diversity score ≥ 0, higher values indicate more diverse recommendations
    """
    n = len(rec_list)
    if n < 2:
        return 0.0

    # Calculate average pairwise dissimilarity
    total_dissim = 0.0
    pair_count = 0

    for i in range(n):
        for j in range(i + 1, n):
            item_i = rec_list[i]
            item_j = rec_list[j]
            total_dissim += dissim_matrix[item_i, item_j]
            pair_count += 1

    return total_dissim / pair_count if pair_count > 0 else 0.0


# Additional utility functions for loading data
def load_genre_matrix_from_csv(movies_csv_path: str) -> tuple:
    """
    Load genre matrix from MovieLens movies.csv file.

    Args:
        movies_csv_path: Path to movies.csv file

    Returns:
        tuple: (genre_matrix, genre_names, movie_ids)
            - genre_matrix: np.ndarray of shape (num_movies, num_genres)
            - genre_names: list of genre names
            - movie_ids: list of movie IDs
    """
    try:
        movies_df = pd.read_csv(movies_csv_path)

        # Handle missing genres
        movies_df['genres'] = movies_df['genres'].fillna('(no genres listed)')

        # Split genres and get all unique genres
        movies_df['genre_list'] = movies_df['genres'].str.split('|')
        all_genres = set()
        for genre_list in movies_df['genre_list']:
            all_genres.update(genre_list)

        # Remove empty strings and sort
        all_genres = sorted([g for g in all_genres if g.strip()])

        # Create binary genre matrix
        num_movies = len(movies_df)
        num_genres = len(all_genres)
        genre_matrix = np.zeros((num_movies, num_genres), dtype=int)

        for idx, genre_list in enumerate(movies_df['genre_list']):
            for genre in genre_list:
                if genre.strip() in all_genres:
                    genre_idx = all_genres.index(genre.strip())
                    genre_matrix[idx, genre_idx] = 1

        movie_ids = movies_df['movieId'].tolist()

        return genre_matrix, all_genres, movie_ids

    except Exception as e:
        raise RuntimeError(f"Error loading genre matrix from {movies_csv_path}: {e}")


