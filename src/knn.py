"""
Item based KNN recommender using cosine similarity between movie rating vectors

Each movie is a vector of ratings from training users. Two movies are similar if the same users rated tehm similarity.
For a user's favourites, we can gather each favourite's K nearest neighbors, add up similarity scores, and return the top-scoring movies.

"""

from collections import defaultdict

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

class ItemKNNRecommender:
    def __init__(self, n_neighbors=20, center_ratings=False):
        """
        n_neighbors: K, how many similar movies to pull for each favorite (tune on val)
        center_ratings: subtract each user's mean rating first, so a 3.0 from a harsh rater and a 3.0 from a generous rater aren't treated the same.
        (We can try both!)
        """
        self.n_neighbors = n_neighbors
        self.center_ratings = center_ratings

    def fit(self, train_ratings, allowed_movies):
        r = train_ratings[train_ratings["movieId"]]

        if self.center_ratings:
            r["rating"] = r["rating"] - r.groupby("userId")["rating"].transform("mean")

        self.movie_ids = np.array(sorted(r["movieId"].unique()))
        self.movie_index = {int(m): i for i, m in enumerate(self.movie_ids)}
        user_index = {u: i for i, u in enumerate(r["userId"].unique())}

        rows = r["movieId"].map(self.movie_index).to_numpy()
        cols = r["userId"].map(user_index).to_numpy()
        self.matrix = csr_matrix(
            (r["rating"].to_numpy(), (rows, cols)), shape=(len(self.movie_ids), len(user_index))
        )

        self.k_query = min(self.n_neighbors + 1, len(self.movie_ids))
        self.nn = NearestNeighbors(metric="cosine", algorithm="brute")
        self.nn.fit(self.matrix)

        counts = r.groupby("movieId").size().sort_values(ascending=False)
        self.fallback = [int(m) for m in counts.index]
        return self

    def recommend(self, favorites, k=5):
        ...

    def similar_movies(self, movie_id, n=10):
        ...


if __name__ == "__main__":
    from src.data import load_data, split_users, popular_enough

    ratings, movies = load_data()
    titles = movies.set_index("movieID")["title"]
    train, _, _ = split_users(ratings)
    model = ItemKNNRecommender(n_neighbors=20).fit(train, popular_enough(train))
