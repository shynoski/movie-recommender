"""
Data loading and evaluation setup, shared by every other file.

1. load_data(): read MovieLens ratings (userId, movieId, rating) and movie titles.
2. split_users(): split USERS (not ratings) 80/10/10 into train / validation / test.
   Models only ever learn from training users, so validation and test users are
   people the model has never seen, like a new visitor to the website.
3. popular_enough(): keep only movies with at least MIN_RATINGS training ratings.
   Movies rated by a handful of people have unreliable similarities; these are the
   only movies any model may use or recommend ("allowed movies").
4. make_eval_cases(): turn each validation/test user into a test case. Of the
   movies they liked (rated >= LIKE_THRESHOLD), 5 random ones become their
   "favorites" (the model's input, like the 5 picks on the website) and the rest
   are "hidden". A good model recommends hidden movies.
5. load_genome(): the optional tag genome (ml-25m only), used for content similarity.

Set the MOVIELENS environment variable (default: ml-25m), e.g.
    MOVIELENS=ml-1m python src/evaluate.py
Download it first with:
    python scripts/download_data.py ml-1m
Files go in data/<dataset name>/ at the repo root.

Run this file directly for a summary of the dataset and the split:
    python src/data.py
"""
import functools
import io
import os
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

# Per-dataset settings.
# min_ratings: movies need this many TRAINING ratings to be used (see popular_enough).
# max_eval_users: cap on val/test users (None = all). The big sets have ~20k
# users per split, which is far more than needed and makes eval slow.
DATASETS = {
    "ml-latest-small": {"min_ratings": 10, "max_eval_users": None},  # 610 users, 100k ratings
    "ml-1m":           {"min_ratings": 10, "max_eval_users": None},  # 6k users, 1M ratings, movies up to 2000
    "ml-25m":          {"min_ratings": 50, "max_eval_users": 5000},  # 162k users, 25M ratings, up to 2019, tag genome
    "ml-32m":          {"min_ratings": 50, "max_eval_users": 5000},  # 200k users, 32M ratings, up to 2023, no genome
}

DATASET = os.environ.get("MOVIELENS", "ml-25m")
if DATASET not in DATASETS:
    raise ValueError(f"MOVIELENS={DATASET!r}; choose one of {list(DATASETS)}")

DATA_DIR = DATA_ROOT / DATASET
LIKE_THRESHOLD = 4.0   # rating >= this counts as "liked" / relevant
N_FAVORITES = 5        # movies given to the model as the user's input
MIN_RATINGS = DATASETS[DATASET]["min_ratings"]
MAX_EVAL_USERS = DATASETS[DATASET]["max_eval_users"]
SEED = 42              # default seed for the split, eval cases and the random baseline

RATING_DTYPES = {"userId": "int32", "movieId": "int32", "rating": "float32"}  # saves memory


def _read_dat(path, names):
    """ml-1m uses '::' separators, which pandas can only parse with its slow Python
    engine. Swapping them for tabs lets the fast C parser do the work."""
    text = path.read_text(encoding="latin-1").replace("::", "\t")
    return pd.read_csv(io.StringIO(text), sep="\t", names=names, header=None, quoting=3)


def load_data(data_dir=None):
    """Return (ratings, movies) DataFrames for the chosen dataset.
    ratings: userId, movieId, rating (0.5-5 stars in half steps; 1-5 whole stars in ml-1m).
    movies: movieId, title, genres."""
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(
            f"{data_dir} not found. Run: python scripts/download_data.py {data_dir.name}"
        )

    if (data_dir / "ratings.dat").exists():  # ml-1m format
        ratings = _read_dat(data_dir / "ratings.dat", ["userId", "movieId", "rating", "timestamp"])
        ratings = ratings[list(RATING_DTYPES)].astype(RATING_DTYPES)
        movies = _read_dat(data_dir / "movies.dat", ["movieId", "title", "genres"])
    else:  # csv format (ml-latest-small, ml-25m, ml-32m)
        ratings = pd.read_csv(
            data_dir / "ratings.csv", usecols=list(RATING_DTYPES), dtype=RATING_DTYPES
        )
        movies = pd.read_csv(data_dir / "movies.csv")
    return ratings, movies


@functools.lru_cache(maxsize=2)
def load_genome(data_dir=None):
    """Tag genome: how strongly each movie has each of ~1,100 tags ("quirky", "dark comedy"...).

    Returns (movie_ids, scores, tag_names), where scores[i, j] is movie_ids[i]'s
    relevance (0-1) for tag_names[j], or None if this dataset has no genome.
    Cached, so loading it from several places costs nothing extra.
    """
    data_dir = Path(data_dir) if data_dir else DATA_DIR
    scores_path = data_dir / "genome-scores.csv"
    if not scores_path.exists():
        return None
    g = pd.read_csv(scores_path, dtype={"movieId": "int32", "tagId": "int32", "relevance": "float32"})
    tags = pd.read_csv(data_dir / "genome-tags.csv").set_index("tagId")["tag"]

    # long format (one row per movie/tag pair) -> movies x tags matrix
    movie_ids, rows = np.unique(g["movieId"].to_numpy(), return_inverse=True)
    tag_ids, cols = np.unique(g["tagId"].to_numpy(), return_inverse=True)
    scores = np.zeros((len(movie_ids), len(tag_ids)), dtype=np.float32)
    scores[rows, cols] = g["relevance"].to_numpy()
    return movie_ids, scores, [tags.get(t, str(t)) for t in tag_ids]


def split_users(ratings, train_frac=0.8, val_frac=0.1, seed=SEED):
    """Split by USER so val/test users are never seen during training.
    Returns (train, val, test) DataFrames holding all ratings of the users in each part.
    The split is fixed by `seed`, so every file sees the same train/val/test users."""
    unique_users = np.sort(ratings["userId"].unique())
    shuffled_users = np.random.default_rng(seed).permutation(unique_users)

    num_train = int(len(shuffled_users) * train_frac)
    num_val = int(len(shuffled_users) * val_frac)

    train_users = shuffled_users[:num_train]
    val_users = shuffled_users[num_train:num_train + num_val]
    test_users = shuffled_users[num_train + num_val:]  # the remaining ~10%

    train_df = ratings[ratings["userId"].isin(train_users)]
    val_df = ratings[ratings["userId"].isin(val_users)]
    test_df = ratings[ratings["userId"].isin(test_users)]
    return train_df, val_df, test_df


def popular_enough(train_ratings, min_ratings=MIN_RATINGS):
    """Movie IDs with enough training ratings to compute meaningful similarity.
    These are the "allowed movies" passed to every model's fit()."""
    movie_counts = train_ratings.groupby("movieId").size()
    return movie_counts[movie_counts >= min_ratings].index


def make_eval_cases(ratings, allowed_movies, n_favorites=N_FAVORITES, seed=SEED,
                    max_users=MAX_EVAL_USERS):
    """For each user: n_favorites liked movies as input, the other liked movies hidden.
    Returns a list of dicts: {"userId", "favorites": [movieId, ...], "hidden": {movieId, ...}}

    Only allowed movies count, since no model can recommend anything else.
    Users with too few liked movies are skipped. If max_users is set, a fixed random
    subset of users is kept (always chosen with SEED, so changing `seed` only changes
    which movies become favorites, not which users are evaluated; robustness.py
    relies on this).
    """
    liked = ratings.loc[
        (ratings["rating"] >= LIKE_THRESHOLD) & ratings["movieId"].isin(allowed_movies),
        ["userId", "movieId"],
    ]

    # users need at least one hidden movie left after picking favorites
    counts = liked.groupby("userId").size()
    eligible = counts.index[counts >= n_favorites + 1].to_numpy()
    if max_users is not None and len(eligible) > max_users:
        eligible = np.sort(np.random.default_rng(SEED).choice(eligible, max_users, replace=False))
    liked = liked[liked["userId"].isin(eligible)]

    # Shuffle each user's liked movies by sorting on a random key; the first
    # n_favorites per user become favorites, the rest are hidden. This is the
    # same idea as .sample() per user, just vectorized so it's fast on big data.
    rng = np.random.default_rng(seed)
    liked = liked.assign(_key=rng.random(len(liked))).sort_values(["userId", "_key"])
    users = liked["userId"].to_numpy()
    movie_ids = liked["movieId"].to_numpy()

    # where each user's block of rows starts and ends
    starts = np.flatnonzero(np.r_[True, users[1:] != users[:-1]])
    ends = np.r_[starts[1:], len(users)]
    return [
        {
            "userId": int(users[s]),
            "favorites": movie_ids[s:s + n_favorites].tolist(),
            "hidden": set(movie_ids[s + n_favorites:e].tolist()),
        }
        for s, e in zip(starts, ends)
    ]


if __name__ == "__main__":
    # Print a summary: dataset size, split sizes, allowed movies, eval cases, genome coverage.
    ratings, movies = load_data()
    train, val, test = split_users(ratings)
    allowed = popular_enough(train)
    val_cases = make_eval_cases(val, allowed)
    test_cases = make_eval_cases(test, allowed)

    print(f"dataset: {DATASET} ({len(ratings):,} ratings, {len(movies):,} movies)")
    print(f"users: train={train.userId.nunique()} val={val.userId.nunique()} test={test.userId.nunique()}")
    print(f"movies with >= {MIN_RATINGS} training ratings: {len(allowed)}")
    print(f"eval cases: val={len(val_cases)} test={len(test_cases)}")
    print("example case:", val_cases[0]["favorites"], f"({len(val_cases[0]['hidden'])} hidden)")
    genome = load_genome()
    if genome is None:
        print("tag genome: not in this dataset (content similarity unavailable)")
    else:
        covered = np.isin(allowed, genome[0]).mean()
        print(f"tag genome: {len(genome[0]):,} movies x {len(genome[2]):,} tags "
              f"({covered:.0%} of candidate movies covered)")