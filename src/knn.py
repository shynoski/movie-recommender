"""
Item-based KNN recommender using cosine similarity between movie rating vectors.

Each movie is a vector of ratings from training users. Two movies are similar if the
same users rated them similarly. For a user's favorites, we gather each favorite's K
nearest neighbors, add up the similarity scores, and return the top-scoring movies.

Scaling to the bigger MovieLens sets: instead of running a nearest-neighbor search on
every recommend() call, fit() precomputes every movie's top neighbors once, working
through the movies in batches so memory stays bounded. recommend() is then just a
table lookup. The table is cached, so models that differ only in K (like the sweep in
evaluate.py) share a single computation.
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

TABLE_NEIGHBORS = 100      # neighbors stored per movie; any K up to this reuses one table
BATCH_CELLS = 20_000_000   # similarity scores held in memory at once (~80 MB of float32)

_TABLE_CACHE = {}


def neighbor_table(matrix, table_k, verbose=True):
    """Top-`table_k` cosine neighbors for every row of a (movies x users) matrix.

    Rows are scaled to unit length, so a dot product between two rows is their cosine
    similarity. We multiply one batch of movies against all movies at a time, keep each
    movie's best neighbors, and throw the rest of the batch away.
    """
    n = matrix.shape[0]
    table_k = min(table_k, n - 1)
    unit = normalize(matrix, norm="l2", axis=1).astype(np.float32)
    unit_t = unit.T.tocsr()
    batch = max(1, BATCH_CELLS // n)

    nbr_idx = np.empty((n, table_k), dtype=np.int32)
    nbr_sim = np.empty((n, table_k), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        rows = np.arange(stop - start)
        sims = (unit[start:stop] @ unit_t).toarray()
        sims[rows, rows + start] = -np.inf  # a movie isn't its own neighbor

        top = np.argpartition(-sims, table_k - 1, axis=1)[:, :table_k]  # best k, unordered
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1, kind="stable")           # now order them
        nbr_idx[start:stop] = np.take_along_axis(top, order, axis=1)
        nbr_sim[start:stop] = np.take_along_axis(top_sims, order, axis=1)

        if verbose and n > batch:
            print(f"\r  similarity table: {stop:,}/{n:,} movies ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    if verbose and n > batch:
        print()
    return nbr_idx, nbr_sim


class ItemKNNRecommender:
    def __init__(self, n_neighbors=20, center_ratings=False):
        """
        n_neighbors: K, how many similar movies to pull for each favorite (tune on val)
        center_ratings: subtract each user's mean rating first, so a 3.0 from a harsh
            rater and a 3.0 from a generous rater aren't treated the same.
        """
        self.n_neighbors = n_neighbors
        self.center_ratings = center_ratings

    def fit(self, train_ratings, allowed_movies):
        allowed = np.unique(np.asarray(allowed_movies, dtype=np.int64))
        table_k = max(self.n_neighbors, TABLE_NEIGHBORS)
        key = (id(train_ratings), len(train_ratings), self.center_ratings,
               hash(allowed.tobytes()), table_k)
        if key not in _TABLE_CACHE:
            _TABLE_CACHE[key] = self._build(train_ratings, allowed, table_k)
        state = _TABLE_CACHE[key]

        self.movie_ids = state["movie_ids"]
        self.movie_index = state["movie_index"]
        self.nbr_idx = state["nbr_idx"]
        self.nbr_sim = state["nbr_sim"]
        self.fallback = state["fallback"]
        return self

    def _build(self, train_ratings, allowed, table_k):
        r = train_ratings[train_ratings["movieId"].isin(allowed)]
        values = r["rating"].to_numpy(np.float32)
        if self.center_ratings:
            values = values - r.groupby("userId")["rating"].transform("mean").to_numpy(np.float32)

        movie_ids = np.unique(r["movieId"].to_numpy())
        rows = np.searchsorted(movie_ids, r["movieId"].to_numpy())
        cols, users = pd.factorize(r["userId"])
        matrix = csr_matrix((values, (rows, cols)), shape=(len(movie_ids), len(users)))

        nbr_idx, nbr_sim = neighbor_table(matrix, table_k)
        counts = np.bincount(rows, minlength=len(movie_ids))
        return {
            "train_ref": train_ratings,  # keeps id(train_ratings) valid for the cache key
            "movie_ids": movie_ids,
            "movie_index": {int(m): i for i, m in enumerate(movie_ids)},
            "nbr_idx": nbr_idx,
            "nbr_sim": nbr_sim,
            "fallback": [int(m) for m in movie_ids[np.argsort(-counts, kind="stable")]],
        }

    def recommend(self, favorites, k=5):
        favs = set(int(m) for m in favorites)
        known = np.array([self.movie_index[m] for m in favs if m in self.movie_index], dtype=np.int64)

        recs = []
        if known.size:
            k_nb = min(self.n_neighbors, self.nbr_idx.shape[1])
            cand = self.nbr_idx[known, :k_nb].ravel()
            sims = self.nbr_sim[known, :k_nb].ravel()
            keep = sims > 0  # zero/negative similarity is not evidence of a match
            cand, inverse = np.unique(cand[keep], return_inverse=True)
            scores = np.bincount(inverse, weights=sims[keep])  # sum of similarities
            not_fav = ~np.isin(cand, known)
            cand, scores = cand[not_fav], scores[not_fav]
            best = np.argsort(-scores, kind="stable")[:k]
            recs = [int(self.movie_ids[j]) for j in cand[best]]

        # fill any empty slots with popular movies
        for movie_id in self.fallback:
            if len(recs) == k:
                break
            if movie_id not in favs and movie_id not in recs:
                recs.append(movie_id)
        return recs

    def similar_movies(self, movie_id, n=10):
        """Handy for sanity checks -- [(movieId, similarity), ...] for one movie (n <= 100)."""
        if movie_id not in self.movie_index:
            return []
        i = self.movie_index[movie_id]
        return [
            (int(self.movie_ids[j]), round(float(s), 3))
            for j, s in zip(self.nbr_idx[i, :n], self.nbr_sim[i, :n])
            if s > 0
        ]


if __name__ == "__main__":
    from data import DATASET, MIN_RATINGS, load_data, popular_enough, split_users

    ratings, movies = load_data()
    titles = movies.set_index("movieId")["title"]
    title_to_id = {t: int(m) for m, t in titles.items()}
    train, _, _ = split_users(ratings)
    model = ItemKNNRecommender(n_neighbors=20).fit(train, popular_enough(train))

    # Look movies up by title: movieIds aren't the same across every MovieLens release.
    my_favorites = [
        "Scott Pilgrim vs. the World (2010)",
        "Dr. Horrible's Sing-Along Blog (2008)",
        "Hunt for the Wilderpeople (2016)",
        "Lord of the Rings: The Fellowship of the Ring, The (2001)",
        "Coraline (2009)",
    ]
    # ml-1m stops at 2000, so fall back to classics it does have
    classics = [
        "Toy Story (1995)",
        "Princess Bride, The (1987)",
        "Matrix, The (1999)",
        "Back to the Future (1985)",
        "Groundhog Day (1993)",
    ]

    favs = []
    for title in my_favorites:
        movie_id = title_to_id.get(title)
        if movie_id is None:
            print(f"(not in {DATASET}: {title})")
        elif movie_id not in model.movie_index:
            print(f"(fewer than {MIN_RATINGS} training ratings: {title})")
        else:
            favs.append(movie_id)
    if len(favs) < 3:
        print("Too few of those favorites are usable here, using classics instead.")
        favs = [title_to_id[t] for t in classics if t in title_to_id]

    for movie_id in favs[:3]:
        print(f"\nMost similar to {titles[movie_id]}:")
        for m, similar in model.similar_movies(movie_id, n=5):
            print(f"    {similar:.3f}   {titles.get(m, m)}")

    print("\nFavorites:", [titles[m] for m in favs])
    print("Recommendations:", [titles.get(m, m) for m in model.recommend(favs)])