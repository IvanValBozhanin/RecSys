import pandas as pd

# --- 1) Load the MovieLens 100K into a DataFrame
#    Make sure 'u_csv.csv' is in your working directory
df = pd.read_csv('ml-latest-small/ml-100k/u_csv.csv')

# --- 2) Compute number of movies watched per user
#    (each row is one rating, so count of ratings = count of watched movies)
watched_counts = df.groupby('userId').size()

# --- 3) Compute summary statistics
mean_count   = watched_counts.mean()
median_count = watched_counts.median()
min_count    = watched_counts.min()
max_count    = watched_counts.max()
std_count    = watched_counts.std()

# --- 4) Print the results
print("=== Watched‐Movies Per User Statistics ===")
print(f"Mean   : {mean_count:.2f}")
print(f"Median : {median_count}")
print(f"Min    : {min_count}")
print(f"Max    : {max_count}")
print(f"Std    : {std_count:.2f}")

# Optional: full descriptive breakdown
print("\nFull distribution (counts by user):")
print(watched_counts.describe())
