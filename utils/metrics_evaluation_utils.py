import torch
import numpy as np


from utils.metrics import (
    calc_diversity, calc_novelty, calc_serendipity,
    load_genre_vectors, jaccard_similarity, cosine_similarity_genre, dissimilarity_from_similarity,
    load_ratings_popularity, popularity_score_max_norm
)

# Define constants here or import them
MOVIES_CSV_PATH = 'ml-latest-small/movies.csv'
RATINGS_CSV_PATH = 'ml-latest-small/ratings.csv'
TOP_N_RECOMMENDATIONS = 10
SERENDIPITY_SIMILARITY_THRESHOLD = 0.001
MIN_RATING_FOR_LIKED = 4.0


# In metrics_evaluation_utils.py

# ... (imports and constants remain the same) ...

def evaluate_beyond_accuracy(
        model,
        X_features_all_users_pt: torch.Tensor,
        eval_user_array_indices: np.ndarray,
        B_train_history_mask_movies_x_users_np: np.ndarray,
        X_eval_targets_orig_movies_x_users_np: np.ndarray,
        B_eval_target_mask_movies_x_users_np: np.ndarray,
        idx_to_movie_id_map: dict,  # This maps internal 0-N movie index to original MovieLens ID
        top_n: int = TOP_N_RECOMMENDATIONS,
        device = 'cpu'
):
    model.eval()
    model.to(device)

    print("\n--- Evaluating Beyond-Accuracy Metrics ---")

    genre_vectors = load_genre_vectors(MOVIES_CSV_PATH)
    movie_popularity_counts, _ = load_ratings_popularity(RATINGS_CSV_PATH)
    print(f"data of movie_popularity_counts: {len(movie_popularity_counts)}")
    print(f"head of movie_popularity_counts: {list(movie_popularity_counts.items())[:10]}")

    # Ensure similarity/dissimilarity functions handle missing movie_ids gracefully
    def robust_cosine_similarity_genre(m1_id, m2_id, g_vectors):
        if m1_id not in g_vectors or m2_id not in g_vectors:
            return 0.0  # Or handle as appropriate (e.g., skip pair, return avg similarity)
        return cosine_similarity_genre(m1_id, m2_id, g_vectors)

    cosine_genre_sim_fn = lambda m1, m2: robust_cosine_similarity_genre(m1, m2, genre_vectors)
    cosine_genre_dissim_fn = dissimilarity_from_similarity(cosine_genre_sim_fn)

    def robust_popularity_score(m_id, pop_counts):
        if m_id not in pop_counts:
            # This case should be rare if movie_popularity_counts is from the same ratings file
            # but if movies.csv has movies not in ratings.csv, this could happen.
            # For novelty, an unknown item is often considered maximally novel (pop_score = 0)
            return 0.0
        return popularity_score_max_norm(m_id, pop_counts)

    pop_score_fn = lambda mid: robust_popularity_score(mid, movie_popularity_counts)

    all_diversities_cosine = []
    all_novelties = []
    all_serendipities = []

    with torch.no_grad():
        all_predicted_ratings_norm = model(X_features_all_users_pt.to(device))

    for user_array_idx in eval_user_array_indices:  # This is the 0-M index for users (column in movies_x_users)
        user_predicted_ratings_for_eval_user = all_predicted_ratings_norm[user_array_idx, :]

        history_internal_indices = np.where(B_train_history_mask_movies_x_users_np[:, user_array_idx] == 1)[0]
        history_movie_ids = set()
        for internal_idx in history_internal_indices:
            if internal_idx in idx_to_movie_id_map:
                original_id = idx_to_movie_id_map[internal_idx]
                # IMPORTANT: Ensure history items are also valid for metric calculation if used in similarity
                if original_id in genre_vectors and original_id in movie_popularity_counts:
                    history_movie_ids.add(original_id)
            # else: internal_idx not in map, shouldn't happen if map is from full movie list

        scores_for_ranking = user_predicted_ratings_for_eval_user.clone()
        # Masking history items (using internal indices for scores_for_ranking)
        for internal_hist_idx in history_internal_indices:  # Use internal indices here
            if internal_hist_idx < scores_for_ranking.shape[0]:
                scores_for_ranking[internal_hist_idx] = -float('inf')

        top_n_internal_indices = torch.topk(scores_for_ranking, top_n).indices.cpu().numpy()

        recommended_movie_ids = []
        for internal_idx in top_n_internal_indices:
            if internal_idx in idx_to_movie_id_map:
                original_movie_id = idx_to_movie_id_map[internal_idx]
                # Filter for only those movies existing in genre and popularity data
                if original_movie_id in genre_vectors and original_movie_id in movie_popularity_counts:
                    recommended_movie_ids.append(original_movie_id)

        if not recommended_movie_ids:
            # print(f"User {user_array_idx}: No valid recommendations after filtering. Skipping.")
            continue

        diversity_cosine = calc_diversity(recommended_movie_ids, cosine_genre_dissim_fn)
        all_diversities_cosine.append(diversity_cosine)

        novelty = calc_novelty(recommended_movie_ids, pop_score_fn)
        all_novelties.append(novelty)

        # For liked_eval_set_movie_ids
        eval_ratings_for_user = X_eval_targets_orig_movies_x_users_np[:, user_array_idx]
        eval_mask_for_user = B_eval_target_mask_movies_x_users_np[:, user_array_idx]

        liked_eval_set_internal_indices = \
        np.where((eval_ratings_for_user * eval_mask_for_user) >= MIN_RATING_FOR_LIKED)[0]
        liked_eval_set_movie_ids = set()
        for internal_idx in liked_eval_set_internal_indices:
            if internal_idx in idx_to_movie_id_map:
                original_id = idx_to_movie_id_map[internal_idx]
                if original_id in genre_vectors:  # Only need genre for similarity in serendipity
                    liked_eval_set_movie_ids.add(original_id)

        serendipity = calc_serendipity(recommended_movie_ids, liked_eval_set_movie_ids, history_movie_ids,
                                       cosine_genre_sim_fn, SERENDIPITY_SIMILARITY_THRESHOLD)
        all_serendipities.append(serendipity)


    print(f" Diversity: {np.size(all_diversities_cosine), all_diversities_cosine[:10], all_diversities_cosine[-10:]}")
    print(f" Novelty: {np.size(all_novelties), all_novelties[:10], all_novelties[-10:]}")
    print(f" Serendipity: {np.size(all_serendipities), all_serendipities[:10], all_serendipities[-10:]}")
    if all_diversities_cosine: print(f"Average Cosine Diversity (Genre): {np.mean(all_diversities_cosine):.4f}")
    if all_novelties: print(f"Average Novelty: {np.mean(all_novelties):.4f}")
    if all_serendipities: print(f"Average Serendipity: {np.mean(all_serendipities):.4f}")