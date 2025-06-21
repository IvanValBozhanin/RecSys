import pandas as pd
import os

# Adjust these paths as needed
INPUT_PATH  = 'ml-latest-small/ml-100k/u.item'
OUTPUT_PATH = 'ml-latest-small/ml-100k/movies_transformed.csv'

# Genre names in the exact order they appear as columns in u.item
GENRES = [
    'unknown','Action','Adventure','Animation',"Children's","Comedy","Crime",
    "Documentary","Drama","Fantasy","Film-Noir","Horror","Musical",
    "Mystery","Romance","Sci-Fi","Thriller","War","Western"
]

def main():
    # Define column names for u.item
    cols = ['movieId','title','release_date','video_release_date','imdb_url'] + GENRES

    # Read only the relevant columns
    df = pd.read_csv(
        INPUT_PATH,
        sep='|',
        names=cols,
        usecols=['movieId','title'] + GENRES,
        encoding='latin-1'
    )

    # Build the pipe-separated genres string
    def pack_genres(row):
        return '|'.join([genre for genre, flag in zip(GENRES, row) if flag == 1]) or '(no genres listed)'

    df['genres'] = df[GENRES].apply(pack_genres, axis=1)

    # Select final columns and write
    out_df = df[['movieId','title','genres']]
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote transformed movies file with {len(out_df)} entries to:\n  {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
    print("Transformation complete.")
