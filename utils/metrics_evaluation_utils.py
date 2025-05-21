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
SERENDIPITY_SIMILARITY_THRESHOLD = 0.5
MIN_RATING_FOR_LIKED = 4.0


def evaluate_beyond_accuracy(
        model,
        X_features_all_users_pt: torch.Tensor,
        eval_user_array_indices: np.ndarray,
        B_train_history_mask_movies_x_users_np: np.ndarray,
        X_eval_targets_orig_movies_x_users_np: np.ndarray,
        B_eval_target_mask_movies_x_users_np: np.ndarray,
        idx_to_movie_id_map: dict,
        top_n: int = TOP_N_RECOMMENDATIONS,
        device: str = 'cpu'
):
    model.eval()
    model.to(device)

    print("\n--- Evaluating Beyond-Accuracy Metrics ---")

    genre_vectors = load_genre_vectors(MOVIES_CSV_PATH)
    # debug
    print(f"genre_vectors: {len(genre_vectors)}")

    movie_popularity_counts, _ = load_ratings_popularity(RATINGS_CSV_PATH)

    cosine_genre_sim_fn = lambda m1, m2: cosine_similarity_genre(m1, m2, genre_vectors)
    cosine_genre_dissim_fn = dissimilarity_from_similarity(cosine_genre_sim_fn)
    pop_score_fn = lambda mid: popularity_score_max_norm(mid, movie_popularity_counts)

    all_diversities_cosine = []
    all_novelties = []
    all_serendipities = []

    with torch.no_grad():
        all_predicted_ratings_norm = model(X_features_all_users_pt.to(device))

    for user_array_idx in eval_user_array_indices:
        user_predicted_ratings_for_eval_user = all_predicted_ratings_norm[user_array_idx, :]

        history_indices_for_user = np.where(B_train_history_mask_movies_x_users_np[:, user_array_idx] == 1)[0]
        history_movie_ids = {idx_to_movie_id_map[idx] for idx in history_indices_for_user if idx in idx_to_movie_id_map}

        scores_for_ranking = user_predicted_ratings_for_eval_user.clone()
        for hist_idx in history_indices_for_user:
            if hist_idx < scores_for_ranking.shape[0]:  # Check bounds
                scores_for_ranking[hist_idx] = -float('inf')

        top_n_indices = torch.topk(scores_for_ranking, top_n).indices.cpu().numpy()

        recommended_movie_ids = []
        for idx in top_n_indices:
            if idx in idx_to_movie_id_map:
                original_movie_id = idx_to_movie_id_map[idx]
                if original_movie_id in genre_vectors and original_movie_id in movie_popularity_counts:
                    recommended_movie_ids.append(original_movie_id)

        if not recommended_movie_ids:
            continue

        if not recommended_movie_ids:
            print(f"User {user_array_idx}: No valid recommendations generated.")  # Add this
            continue

        print(f"User {user_array_idx}: Recommended Movie IDs: {recommended_movie_ids[:5]}")  # Print first 5
        print(f"User {user_array_idx}: History Movie IDs: {list(history_movie_ids)[:5]}")

        diversity_cosine = calc_diversity(recommended_movie_ids, cosine_genre_dissim_fn)
        all_diversities_cosine.append(diversity_cosine)

        novelty = calc_novelty(recommended_movie_ids, pop_score_fn)
        all_novelties.append(novelty)

        eval_ratings_for_user = X_eval_targets_orig_movies_x_users_np[:, user_array_idx]
        eval_mask_for_user = B_eval_target_mask_movies_x_users_np[:, user_array_idx]

        liked_eval_set_indices = np.where((eval_ratings_for_user * eval_mask_for_user) >= MIN_RATING_FOR_LIKED)[0]
        liked_eval_set_movie_ids = {idx_to_movie_id_map[idx] for idx in liked_eval_set_indices if
                                    idx in idx_to_movie_id_map and idx_to_movie_id_map[idx] in genre_vectors}
        print(f"User {user_array_idx}: Liked Eval Set Movie IDs: {list(liked_eval_set_movie_ids)[:5]}")

        serendipity = calc_serendipity(recommended_movie_ids, liked_eval_set_movie_ids, history_movie_ids,
                                       cosine_genre_sim_fn, SERENDIPITY_SIMILARITY_THRESHOLD)
        all_serendipities.append(serendipity)

    if all_diversities_cosine:
        print(f"Average Cosine Diversity (Genre): {np.mean(all_diversities_cosine):.4f}")
    if all_novelties: print(f"Average Novelty: {np.mean(all_novelties):.4f}")
    if all_serendipities: print(f"Average Serendipity: {np.mean(all_serendipities):.4f}")