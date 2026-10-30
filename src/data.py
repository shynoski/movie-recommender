"""Load MovieLens, split users, filter sparse movies, build evaluation cases."""

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "ml-latest-small"
LIKE_THRESHOLD = 4.0   # rating >= this counts as "liked" / relevant
N_FAVORITES = 5        # movies given to the model as the user's input
MIN_RATINGS = 10       # movies need this many training ratings to be used
SEED = 42

def load_data(data_dir=DATA_DIR):
    ...

def split_users(ratings, train_frac=0.8, val_frac=0.1, seed=SEED):
    """Split by USER so val/test users are never seen during training"""
    ...

def popular_enough(train_ratings, min_ratings=MIN_RATINGS):
    """Movie IDs with enough training ratings to compute meaningful similarity."""
    ...

def make_eval_cases(ratings, allowed_movies, n_favorites=N_FAVORITES, seed=SEED):
    """For each user: n_favorite liked movies as input, the other liked movies hidden.
    Returns a list of dicts: {"userID", "favourites": [movieID, ...], "hidden": {movieID, ...}}
    Users with too few liked movies are skipped.
    """
    ...

if __name__ == "__main__":
    ratings, movies = load_data()
    train, val, test = split_users(ratings)
    allowed = popular_enough(train)
    val_cases = make_eval_cases(val, allowed)
    test_cases = make_eval_cases(test, allowed)

    print(f"users: train={train.userId.nunique()} val={val.userId.nunique()} test={test.userId.nunique()}")
    print(f"movies with >= {MIN_RATINGS} training ratings: {len(allowed)}")
    print(f"eval cases: val={len(val_cases)} test={len(test_cases)}")
    print("example case:", val_cases[0]["favorites"], f"({len(val_cases[0]['hidden'])} hidden)")