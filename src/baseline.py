"""Baseline recommenders: popularity (target to beat) and random (lower bound).

Every recommender shares the same interface so evaluate.py can treat them alike:
    model.fit (train_ratings, allow_movies)
    model.recommend(favorites, k=5) --> list of movieIds (never include favorites)

"""

import numpy as np

from data import LIKE_THRESHOLD, SEED

class PopularityRecommender:
    """Recommends the moveis liked by the most training users, same list for everyone. """
    

    def fit(self, train_ratings, allowed_movies):
        liked = train_ratings[
            (train_ratings["rating"] >= LIKE_THRESHOLD) & train_ratings["movieId"].isin(allowed_movies)
        ]
        counts = liked.groupby("movieId").size().sort_values(ascending=False)
        self.ranked = [int(m) for m in counts.index]
        return self

    def recommend(self, favorites, k=5):
        favs = set(favorites)
        recs = []
        for movie_id in self.ranked:
            if movie_id not in favs:
                recs.append(movie_id)
                if len(recs) == k:
                    break

        return recs

class RandomReccommender:
    """Recommends k random allowed movies. Any real model should beat this easily."""

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def fit(self, train_ratings, allowed_movies):
        self.movies = sorted(int(m) for m in allowed_movies)
        return self

    def recommend(self, favorites, k=5):
        favs = set(favorites)
        candidates = [m for m in self.movies if m not in favs]
        return [int(m) for m in self.rng.choice(candidates, size=k, replace=False)]