"""Baseline recommenders: popularity (target to beat) and random (lower bound).

Every recommender shares the same interface so evaluate.py can treat them alike:
    model.fit (train_ratings, allow_movies)
    model.recommend(favorites, k=5) --> list of movieIds (never include favorites)

"""

import numpy as np

from src.data import LIKE_THRESHOLD, SEED

class PopularityRecommender:
    """Recommends the moveis liked by the most training users, same list for everyone. """
    ...

    def fit(self, train_ratings, allowed_movies):
        ...

    def recommedn(self, favorites, k=5):
        ...

class RandomReccommender:
    """Recommends k random allowed movies. Any real model should beat this easily."""

    def __init__(self, seed=SEED):
        ...

    def fit(self, train_ratings, allowed_movies):
        ...

    def recommend(self, favorites, k=5):
        ...