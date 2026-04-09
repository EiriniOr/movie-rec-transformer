"""
Preprocess MovieLens 25M into user-ordered sequences for training.

Steps:
  1. Load ratings.csv + movies.csv
  2. Sort each user's ratings by timestamp → ordered watch history
  3. Filter to users with >= MIN_RATINGS ratings
  4. Encode movieIds to contiguous integer indices (0-based)
  5. Split: all but last 3 movies → train; last 3 → test
  6. Serialize everything to data/processed/

Run:
  python scripts/preprocess.py
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ── config ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "ml-25m"
OUT_DIR = ROOT / "data" / "processed"

MIN_RATINGS = 20  # discard users with fewer ratings than this
TEST_SIZE = 3  # hold out the last N movies per user for evaluation


def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load ratings and movies CSVs, return as DataFrames."""
    print("Loading ratings.csv …")
    ratings = pd.read_csv(
        RAW_DIR / "ratings.csv",
        dtype={
            "userId": "int32",
            "movieId": "int32",
            "rating": "float32",
            "timestamp": "int64",
        },
    )
    print(f"  {len(ratings):,} ratings, {ratings['userId'].nunique():,} users")

    print("Loading movies.csv …")
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    print(f"  {len(movies):,} movies")

    return ratings, movies


def build_sequences(ratings: pd.DataFrame) -> dict[int, list[int]]:
    """
    For each user, sort their ratings by timestamp and return the ordered
    list of movieIds.  This captures their temporal watch history.
    """
    print("Building per-user sequences …")
    # Sort globally — groupby preserves order within each group
    ratings_sorted = ratings.sort_values("timestamp")

    sequences: dict[int, list[int]] = {}
    for user_id, group in tqdm(ratings_sorted.groupby("userId"), desc="users"):
        sequences[user_id] = group["movieId"].tolist()

    return sequences


def filter_users(
    sequences: dict[int, list[int]], min_ratings: int
) -> dict[int, list[int]]:
    """Drop users with fewer than min_ratings interactions."""
    before = len(sequences)
    sequences = {uid: seq for uid, seq in sequences.items() if len(seq) >= min_ratings}
    after = len(sequences)
    print(
        f"Filtered users: {before:,} → {after:,} (kept users with >= {min_ratings} ratings)"
    )
    return sequences


def build_movie_mappings(
    sequences: dict[int, list[int]],
    movies: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int], dict[int, str]]:
    """
    Create contiguous 0-based integer indices for all movies that appear
    in the filtered sequences.  Movies only in movies.csv but never rated
    by retained users are excluded.

    Returns:
        movie2idx  : movieId → index
        idx2movie  : index → movieId
        idx2title  : index → human-readable title string
    """
    # Gather every movieId that appears in at least one retained sequence
    all_movie_ids: set[int] = set()
    for seq in sequences.values():
        all_movie_ids.update(seq)

    sorted_ids = sorted(all_movie_ids)
    movie2idx = {mid: idx for idx, mid in enumerate(sorted_ids)}
    idx2movie = {idx: mid for mid, idx in movie2idx.items()}

    # Build a lookup from movieId → title for display purposes
    mid_to_title = dict(zip(movies["movieId"], movies["title"]))
    idx2title = {
        idx: mid_to_title.get(mid, f"Unknown ({mid})") for idx, mid in idx2movie.items()
    }

    print(f"Vocabulary size: {len(movie2idx):,} unique movies")
    return movie2idx, idx2movie, idx2title


def encode_sequences(
    sequences: dict[int, list[int]],
    movie2idx: dict[int, int],
) -> dict[int, list[int]]:
    """Replace raw movieIds with integer indices in every sequence."""
    return {uid: [movie2idx[mid] for mid in seq] for uid, seq in sequences.items()}


def train_test_split(
    sequences: dict[int, list[int]],
    test_size: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    For each user:
      - train: all but the last `test_size` movie indices
      - test : the last `test_size` movie indices

    The test set represents movies we want the model to predict
    given the training prefix as context.
    """
    train, test = {}, {}
    for uid, seq in sequences.items():
        train[uid] = seq[:-test_size]
        test[uid] = seq[-test_size:]
    print(f"Split done — {len(train):,} users, {test_size} test items each")
    return train, test


def compute_dataset_stats(
    sequences_train: dict[int, list[int]],
    sequences_test: dict[int, list[int]],
    vocab_size: int,
    ratings_total: int,
) -> dict:
    """Aggregate stats that the Streamlit overview page will display."""
    train_lengths = [len(s) for s in sequences_train.values()]
    return {
        "n_users": len(sequences_train),
        "n_movies": vocab_size,
        "n_ratings_total": ratings_total,
        "avg_sequence_length": round(sum(train_lengths) / len(train_lengths), 1),
        "min_sequence_length": min(train_lengths),
        "max_sequence_length": max(train_lengths),
        "test_items_per_user": len(next(iter(sequences_test.values()))),
    }


def save_artifacts(
    sequences_train: dict,
    sequences_test: dict,
    movie2idx: dict,
    idx2movie: dict,
    idx2title: dict,
    stats: dict,
) -> None:
    """Pickle all processed artifacts and write stats as JSON."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "sequences_train.pkl": sequences_train,
        "sequences_test.pkl": sequences_test,
        "movie2idx.pkl": movie2idx,
        "idx2movie.pkl": idx2movie,
        "idx2title.pkl": idx2title,
    }
    for filename, obj in artifacts.items():
        path = OUT_DIR / filename
        with open(path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        size_mb = path.stat().st_size / 1e6
        print(f"  Saved {filename}: {size_mb:.1f} MB")

    stats_path = OUT_DIR / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print("  Saved dataset_stats.json")


def main() -> None:
    ratings, movies = load_raw_data()
    n_ratings_total = len(ratings)

    sequences = build_sequences(ratings)
    sequences = filter_users(sequences, min_ratings=MIN_RATINGS)

    movie2idx, idx2movie, idx2title = build_movie_mappings(sequences, movies)

    # Encode movieIds → integer indices before splitting
    sequences_encoded = encode_sequences(sequences, movie2idx)

    sequences_train, sequences_test = train_test_split(
        sequences_encoded, test_size=TEST_SIZE
    )

    stats = compute_dataset_stats(
        sequences_train, sequences_test, len(movie2idx), n_ratings_total
    )
    print("\nDataset stats:")
    for k, v in stats.items():
        print(f"  {k}: {v:,}" if isinstance(v, int) else f"  {k}: {v}")

    save_artifacts(
        sequences_train, sequences_test, movie2idx, idx2movie, idx2title, stats
    )
    print("\nPreprocessing complete.")


if __name__ == "__main__":
    main()
