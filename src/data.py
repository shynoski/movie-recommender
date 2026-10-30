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

    #count how many ratins each movie has in training data
    movie_counts = train_ratings.groupby("movieId").size()

    #get movies that have at least min_ratings=10 ratings
    eligible_results = movie_counts[movie_counts >= min_ratings]

    #return IDs of movies that meet thres.
    return eligible_results.index

def make_eval_cases(ratings, allowed_movies, n_favorites=N_FAVORITES, seed=SEED):
    """For each user: n_favorite liked movies as input, the other liked movies hidden.
    Returns a list of dicts: {"userID", "favourites": [movieID, ...], "hidden": {movieID, ...}}
    Users with too few liked movies are skipped.
    """
    #find movies where users rated 4.0 or higher
    liked_movies = ratings[ratings["rating"] >= LIKE_THRESHOLD]

    #Keep only movies with enough training ratings
    liked_movies = liked_movies[
        liked_movies["movieId"].isin(allowed_movies)
    ]

    #store evaluation cases for all eligible users
    eval_cases = []

    #group liked movies by user who rated them
    grouped_users = liked_movies.groupby("userId")

    #go through each user and their liked movies
    for user_id, user_movies in grouped_users:
        #skip users that don't have enough liekd movies
        if len(user_movies) < n_favorites + 1:
            continue

        #randomly select 5 movies from this user's liked movies
        selected_movies = user_movies.sample(n=n_favorites, random_state=seed)

        #get movie IDs and convert them into a list
        favorites = selected_movies["movieId"].tolist()

        #keep movies that were NOT selected as favorites
        remaining_movies = user_movies[
            ~user_movies["movieId"].isin(favorites)
        ]

        #store remaining movie IDs in a set
        hidden = set(remaining_movies["movieId"])

        #store current user's evaluation info.
        case = {
            "userId": user_id,
            "favorites": favorites,
            "hidden": hidden
        }

        #Add this user's evaluation case to list
        eval_cases.append(case)
    return eval_cases

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