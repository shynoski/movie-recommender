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
        r = train_ratings[train_ratings["movieId"].isin(allowed_movies)].copy()

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
        favs = set(int(m) for m in favorites)
        known = [self.movie_index[m] for m in favs if m in self.movie_index]

        scores = defaultdict(float)
        if known:
            distances, indicies = self.nn.kneighbors(
                self.matrix[known], n_neighbors=self.k_query
            )
            for dist_row, idx_row in zip(distances, indicies):
                for dist, j in zip(dist_row, idx_row):
                    candidate = int(self.movie_ids[j])
                    if candidate not in favs:
                        scores[candidate] += 1.0 - dist # this is the cosine similarity btw

        recs = sorted(scores, key=scores.get, reverse=True)[:k]

        # fill slots with popular movies if empty
        for movie_id in self.fallback:
            if len(recs) == k:
                break
            if movie_id not in favs and movie_id not in recs:
                recs.append(movie_id)
        return recs

    def similar_movies(self, movie_id, n=10):
        """handy for sanity checks -- [(movieId, similarity), ...] for one movie"""
        if movie_id not in self.movie_index:
            return[]
        distances, indices = self.nn.kneighbors(
            self.matrix[self.movie_index[movie_id]], n_neighbors=min(n+1, len(self.movie_ids))
        )
        return [
            (int(self.movie_ids[j]), round(1.0 - d, 3))
            for d, j in zip(distances[0], indices[0])
            if int(self.movie_ids[j]) != movie_id
        ][:n]

if __name__ == "__main__":
    from data import load_data, split_users, popular_enough

    ratings, movies = load_data()
    titles = movies.set_index("movieId")["title"]
    train, _, _ = split_users(ratings)
    model = ItemKNNRecommender(n_neighbors=20).fit(train, popular_enough(train))

    for movie_id in [79702, 66934, 152970]: # these are my favoirte movies! Scott Pilgrim vs the world, Dr horrible's sing along blog, and Hunt for the Wilderpeople
        print(f"\nMost similar to {titles.get(movie_id, movie_id)}:")
        for m, similar in model.similar_movies(movie_id, n=5):
            print(f"    {similar:.3f}   {titles.get(m, m)}")

        favs = [79702, 66934, 152970, 4993, 66097] # + LOTR and Coraline these are more of my favorite movies (＾▽＾)
        print("\nFavorites:", [titles.get(m, m) for m in favs])
        print("Recommendations:", [titles.get(m, m) for m in model.recommend(favs)])