"""
Baseline recommenders: popularity (the target to beat) and random (the lower bound).

The KNN model only counts as useful if it beats PopularityRecommender, which gives
every user the same list of crowd favorites. RandomReccommender shows what pure
chance scores; any real model should beat it easily.

Every recommender in the project (these two and ItemKNNRecommender in knn.py) has
the same interface, so evaluate.py and robustness.py can treat them alike:
    model.fit(train_ratings, allowed_movies)    -> returns the model
    model.recommend(favorites, k=5)             -> list of k movieIds, best first,
                                                   never one of the favorites
favorites is a list of movieIds. (ItemKNNRecommender also accepts a
{movieId: "love" | "like" | "seen"} dict; the baselines ignore the tiers and just
avoid recommending those movies.)
"""

import numpy as np

from data import LIKE_THRESHOLD, SEED


class PopularityRecommender:
    """Recommends the movies liked (rated >= LIKE_THRESHOLD) by the most training users.
    Same list for everyone, minus any of the user's own favorites."""

    def fit(self, train_ratings, allowed_movies):
        """Rank allowed movies by how many training users liked them."""
        liked = train_ratings[
            (train_ratings["rating"] >= LIKE_THRESHOLD) & train_ratings["movieId"].isin(allowed_movies)
        ]
        counts = liked.groupby("movieId").size().sort_values(ascending=False)
        self.ranked = [int(m) for m in counts.index]
        return self

    def recommend(self, favorites, k=5):
        """The k highest-ranked movies that aren't favorites."""
        favs = set(favorites)
        recs = []
        for movie_id in self.ranked:
            if movie_id not in favs:
                recs.append(movie_id)
                if len(recs) == k:
                    break

        return recs


class RandomReccommender:
    """Recommends k random allowed movies (never a favorite).

    One random generator is shared by all calls, so results are reproducible for the
    same seed and the same order of calls."""

    def __init__(self, seed=SEED):
        self.rng = np.random.default_rng(seed)

    def fit(self, train_ratings, allowed_movies):
        """Nothing to learn: just remember which movies may be recommended."""
        self.movies = sorted(int(m) for m in allowed_movies)
        return self

    def recommend(self, favorites, k=5):
        """k different random movies that aren't favorites."""
        favs = set(favorites)
        candidates = [m for m in self.movies if m not in favs]
        return [int(m) for m in self.rng.choice(candidates, size=k, replace=False)]