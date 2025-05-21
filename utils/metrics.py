import sys
sys.path.append('../')
import pandas as pd
import numpy as np

from typing import List, Callable, Dict, Tuple, Set


# Main Metrics methods

def calc_diversity(
        recommendation: List[int],
        dissimilarity: Callable[[int, int], float]
) -> float:
    n = len(recommendation)
    if n < 2: return 0.0
    total = 0
    count = 0
    for i in range(n-1):
        for j in range(i+1, n):
            total += dissimilarity(recommendation[i], recommendation[j])
            count += 1
    return 2 * total / count

def calc_novelty(
        recommendation: List[int],
        popularity_score: Callable[[int], float]
) -> float:
    if not recommendation: return 0.0
    return sum(1 - popularity_score(i) for i in recommendation) / len(recommendation)

def calc_serendipity(
    recommendation: List[int],
    liked_items: Set[int],
    user_history: Set[int],
    similarity: Callable[[int, int], float],
    threshold: float = 0.5
) -> float:
    if not recommendation: return 0.0
    ser = 0
    for i in recommendation:
        rel = (i in liked_items)
        unexp = all(similarity(i, h) < threshold for h in user_history)
        if rel and unexp:
            ser += 1
    return ser / len(recommendation)



# Helper methods
# Genre Based
def load_genre_vectors(
    movies_csv: str
) -> Dict[int, np.ndarray]:
    """
    Load MovieLens movies.csv and build a multi-hot genre vector for each movie.

    Parameters:
    - movies_csv: str
        Path to the MovieLens movies.csv file.

    Returns:
    - Dict[int, np.ndarray]
        Mapping from movieId to a binary vector of length G,
        where G is the number of unique genres.
    """

    movies = pd.read_csv(movies_csv)
    # Handle missing genres by filling with empty string
    movies['genre_list'] = movies['genres'].fillna('').str.split('|')
    all_genres = sorted({g for lst in movies['genre_list'] for g in lst if g})
    genre_to_idx = {genre: i for i, genre in enumerate(all_genres)}
    vecs = {}
    for _, r in movies.iterrows():
        vec = np.zeros(len(all_genres), dtype=float)
        for g in r['genre_list']:
            if g and g in genre_to_idx:
                vec[genre_to_idx[g]] = 1.0
        vecs[int(r['movieId'])] = vec
    return vecs

def jaccard_similarity(
        a: int,
        b: int,
        genre_vectors: Dict[int, np.ndarray]
) -> float:
    """
    Compute Jaccard similarity between two movies based on genre vectors.
    """
    vec_a = genre_vectors[a]
    vec_b = genre_vectors[b]
    intersection = np.logical_and(vec_a, vec_b).sum()
    union = np.logical_or(vec_a, vec_b).sum()
    if union == 0:
        return 0.0
    return intersection / union

def cosine_similarity_genre(
        a: int,
        b: int,
        genre_vectors: Dict[int, np.ndarray]
) -> float:
    """
    Compute Cosine similarity between two movies based on genre vectors.
    """
    vec_a = genre_vectors[a]
    vec_b = genre_vectors[b]
    num = float(np.dot(vec_a, vec_b))
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return num / denom

def dissimilarity_from_similarity(
    similarity_fn: Callable[[int, int], float]
) -> Callable[[int, int], float]:
    """
    Given a similarity function, return a dissimilarity function = 1 - similarity.
    """
    def diss(a: int, b: int) -> float:
        return 1.0 - similarity_fn(a, b)
    return diss


# Popularity Based
def load_ratings_popularity(
        ratings_csv: str
) -> Tuple[Dict[int, int], int]:
    """
    Load MovieLens ratings.csv and compute item popularity counts.

    Returns:
    - movie_counts: Dict[int, int] mapping movieId to number of unique users who rated it.
    - total_users: int total number of unique userWs in the dataset.
    """
    ratings = pd.read_csv(ratings_csv)
    total_users = ratings['userId'].nunique()
    # Count unique users per movie
    movie_counts = ratings.groupby('movieId')['userId'].nunique().to_dict()
    return movie_counts, total_users

def popularity_score_max_norm(
    movie_id: int,
    movie_counts: Dict[int, int]
) -> float:
    """
    Normalize popularity by the max count in the catalog.
    """
    max_count = max(movie_counts.values()) if movie_counts else 1
    return movie_counts.get(movie_id, 0) / max_count

def popularity_score_user_norm(
    movie_id: int,
    movie_counts: Dict[int, int],
    total_users: int
) -> float:
    """
    Normalize popularity by total number of users.
    """
    if total_users <= 0:
        return 0.0
    return movie_counts.get(movie_id, 0) / total_users

# Embedding Based
def cosine_similarity_embeddings(
    a: int,
    b: int,
    embeddings: Dict[int, np.ndarray]
) -> float:
    """
    Cosine similarity between precomputed item embeddings.

    Parameters:
    - embeddings: Dict[int, np.ndarray] mapping movieId to embedding vector.
    """
    vec_a = embeddings.get(a)
    vec_b = embeddings.get(b)
    if vec_a is None or vec_b is None:
        return 0.0
    num = float(np.dot(vec_a, vec_b))
    denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if denom == 0:
        return 0.0
    return num / denom
