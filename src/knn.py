"""
Item-based KNN recommender: "people who liked your movies also liked these".

1. fit(): build a neighbor table (once per group of settings, then cached).
   - Make a movies x users matrix from the TRAINING users' ratings. Each movie is
     a row vector: its ratings from every training user (0 where unrated).
   - Two movies are similar when the same users rated them similarly. We measure
     this with cosine similarity between their row vectors (1 = identical rating
     pattern, 0 = nothing in common).
   - For every movie we store its TABLE_NEIGHBORS (100) most similar movies and
     their similarity scores. That table is all recommend() needs.

2. recommend(): score candidate movies from the user's picks.
   - Each pick has a tier: LOVE, LIKE or SEEN (a plain list of movieIds means all LIKE).
   - For every LOVE/LIKE pick, look up its K nearest neighbors (K = n_neighbors).
     Each neighbor earns  similarity ** sim_power * tier weight  points
     (TIER_WEIGHTS: LOVE = 2, LIKE = 1). SEEN picks earn their neighbors nothing.
   - Add up the points per movie. A movie close to several picks scores highest.
   - Remove every pick (all three tiers), return the k highest scores.
   - If fewer than k movies scored above zero, fill the rest with the most-rated
     movies the user hasn't picked (score 0).

3. Optional extras (all off by default), applied in this order when on:
   a. content_weight: blend in a second neighbor table built from the tag genome
      (what a movie is *about*, e.g. "quirky", "dark comedy"), instead of ratings.
   b. pop_penalty: divide scores by a movie's rating count, to surface less-known movies.
   c. pop_weight: mix scores with plain popularity, to favor well-known movies.

SETTINGS (ItemKNNRecommender arguments)

  n_neighbors     K, neighbors used per pick. Up to 100 reuses the cached table.
  center_ratings  subtract each user's average rating before computing similarity,
                  so a harsh rater's 3.0 and a generous rater's 3.0 aren't equal.
  binarize        None: use the 0.5-5 star ratings as they are.
                  "liked": 1 if the user rated it >= LIKE_THRESHOLD, else nothing.
                  "rated": 1 for every rating, whatever the stars.
                  (center_ratings is ignored when binarize is set.)
  shrinkage       shrink a pair's similarity by  common / (common + shrinkage),
                  where common = users who rated both movies. Pairs that only a
                  few people rated together are trusted less.
  sim_power       raise similarities to this power before adding them up. Above 1,
                  very close neighbors count much more than loose ones.
  content_weight  0-1. Final score = (1 - w) * ratings score + w * content score,
                  each scaled to a max of 1. 1.0 = content only. Needs ml-25m.
  pop_penalty     >= 0. Divide each score by (number of training ratings) ** p.
  pop_weight      0-1. Scale scores to [0, 1], then mix in w * popularity
                  (log number of likes, also scaled to [0, 1]).

The table depends only on the data and on center_ratings / binarize / shrinkage.
Models that differ only in K, sim_power, the tiers or the extras share one table,
so evaluating many variants costs one table build, not one per variant.

Run this file directly for a sanity check of similar movies and recommendations:
    python src/knn.py
"""
import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from sklearn.preprocessing import normalize

from data import LIKE_THRESHOLD, load_genome

TABLE_NEIGHBORS = 100      # neighbors stored per movie; any K up to this reuses one table
BATCH_CELLS = 20_000_000   # similarity scores held in memory at once (~80 MB of float32)

# How much each tier's picks count when scoring candidates (see "HOW IT WORKS", step 2).
# SEEN is not here on purpose: it adds nothing and only keeps the movie out of the results.
# These change only how recommend() combines picks, never the fitted table.
TIER_WEIGHTS = {"love": 2.0, "like": 1.0}
TIERS = ("love", "like", "seen")

# Built tables, shared by every model instance with the same table settings (see fit()).
_TABLE_CACHE = {}    # ratings tables
_CONTENT_CACHE = {}  # tag-genome tables


def neighbor_table(matrix, table_k, shrinkage=0.0, verbose=True):
    """Top-`table_k` cosine neighbors for every row of a (movies x features) matrix.
    Works for the sparse ratings matrix and the dense genome matrix.

    Returns (nbr_idx, nbr_sim), both shaped (movies, table_k): row i holds movie i's
    neighbors as row positions in `matrix`, and their similarities, best first.

    Rows are scaled to unit length, so a dot product between two rows is their cosine
    similarity. Comparing all movies at once would need a movies x movies matrix (too
    big for ml-25m), so we multiply one batch of movies against all movies, keep each
    movie's best neighbors, and throw the rest of the batch away. With shrinkage, a
    second product of 0/1 "who rated it" matrices counts each pair's shared users.
    """
    n = matrix.shape[0]
    table_k = min(table_k, n - 1)
    unit = normalize(matrix, norm="l2", axis=1).astype(np.float32)
    unit_t = unit.T.tocsr() if issparse(unit) else unit.T
    if shrinkage > 0:
        rated = matrix.copy().astype(np.float32)
        rated.data[:] = 1.0  # 1 wherever a rating exists
        rated_t = rated.T.tocsr()
    batch = max(1, BATCH_CELLS // n // (2 if shrinkage > 0 else 1))  # rows per batch

    nbr_idx = np.empty((n, table_k), dtype=np.int32)
    nbr_sim = np.empty((n, table_k), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, batch):
        stop = min(start + batch, n)
        rows = np.arange(stop - start)
        sims = unit[start:stop] @ unit_t  # (batch, all movies) cosine similarities
        sims = sims.toarray() if issparse(sims) else np.asarray(sims)
        if shrinkage > 0:
            common = (rated[start:stop] @ rated_t).toarray()  # users who rated both
            sims *= common / (common + shrinkage)
            del common
        sims[rows, rows + start] = -np.inf  # a movie isn't its own neighbor

        top = np.argpartition(-sims, table_k - 1, axis=1)[:, :table_k]  # best k, unordered
        top_sims = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_sims, axis=1, kind="stable")           # now order them
        nbr_idx[start:stop] = np.take_along_axis(top, order, axis=1)
        nbr_sim[start:stop] = np.take_along_axis(top_sims, order, axis=1)

        if verbose and n > 10_000:
            print(f"\r  similarity table: {stop:,}/{n:,} movies ({time.time() - t0:.0f}s)",
                  end="", flush=True)
    if verbose and n > 10_000:
        print()
    return nbr_idx, nbr_sim


class ItemKNNRecommender:
    """The KNN model. See the module docstring for how it works and what each setting does.

    Same interface as the baselines (baseline.py):
        model.fit(train_ratings, allowed_movies)
        model.recommend(picks, k=5) -> list of movieIds, best first, never a pick
    plus recommend_with_scores() and similar_movies() for demos and sanity checks.

    Internally, movies are referred to by their row position in self.movie_ids
    ("row" or "index" below), not by movieId; self.movie_index maps movieId -> row.
    """

    def __init__(self, n_neighbors=20, center_ratings=False, binarize=None,
                 shrinkage=0.0, sim_power=1.0, pop_weight=0.0, pop_penalty=0.0,
                 content_weight=0.0):
        """Store and check the settings. Defaults = plain KNN with no extras.
        Each setting is explained under SETTINGS in the module docstring."""
        if binarize not in (None, "liked", "rated"):
            raise ValueError("binarize must be None, 'liked' or 'rated'")
        if not 0.0 <= pop_weight <= 1.0:
            raise ValueError("pop_weight must be between 0 and 1")
        if not 0.0 <= content_weight <= 1.0:
            raise ValueError("content_weight must be between 0 and 1")
        if pop_penalty < 0:
            raise ValueError("pop_penalty must be >= 0")
        self.n_neighbors = n_neighbors
        self.center_ratings = center_ratings and binarize is None
        self.binarize = binarize
        self.shrinkage = float(shrinkage)
        self.sim_power = float(sim_power)
        self.pop_weight = float(pop_weight)
        self.pop_penalty = float(pop_penalty)
        self.content_weight = float(content_weight)

    def fit(self, train_ratings, allowed_movies):
        """Get the neighbor table(s) for these training ratings and settings.

        allowed_movies: the movieIds the model may use and recommend (popular_enough()
        in data.py). Tables are looked up in the cache first and only built if missing.
        Returns self, so `model = ItemKNNRecommender(...).fit(train, allowed)` works.
        """
        allowed = np.unique(np.asarray(allowed_movies, dtype=np.int64))
        table_k = max(self.n_neighbors, TABLE_NEIGHBORS)
        # only settings that change the neighbor table go in the key
        key = (id(train_ratings), len(train_ratings), hash(allowed.tobytes()), table_k,
               self.center_ratings, self.binarize, self.shrinkage)
        if key not in _TABLE_CACHE:
            _TABLE_CACHE[key] = self._build(train_ratings, allowed, table_k)
        state = _TABLE_CACHE[key]

        self.movie_ids = state["movie_ids"]          # row -> movieId (sorted)
        self.movie_index = state["movie_index"]      # movieId -> row
        self.nbr_idx = state["nbr_idx"]              # row -> neighbor rows, best first
        self.nbr_sim = state["nbr_sim"]              # row -> neighbor similarities
        self.fallback = state["fallback"]            # movieIds, most-rated first
        self.popularity = state["popularity"]        # row -> popularity in [0, 1]
        self.rating_counts = state["rating_counts"]  # row -> number of training ratings

        self.content_idx = self.content_sim = None
        if self.content_weight > 0:
            ckey = (hash(allowed.tobytes()), table_k)
            if ckey not in _CONTENT_CACHE:
                _CONTENT_CACHE[ckey] = self._build_content(table_k)
            self.content_idx, self.content_sim = _CONTENT_CACHE[ckey]
        return self

    def _build_content(self, table_k):
        """Neighbor table from tag-genome vectors (what each movie is about), with the
        same rows as the ratings table so the two can be blended.
        Movies without genome scores get no content neighbors (similarity -inf)."""
        genome = load_genome()
        if genome is None:
            raise ValueError("content_weight needs a dataset with a tag genome (ml-25m)")
        g_ids, g_scores, _ = genome
        has = np.isin(self.movie_ids, g_ids)
        rows = np.flatnonzero(has)  # positions in self.movie_ids
        vectors = g_scores[np.searchsorted(g_ids, self.movie_ids[has])]
        # Every movie has *some* relevance for every tag, so raw cosine says everything is
        # similar. Centering each tag on its average makes the vector mean "more/less
        # quirky than a typical movie", which separates movies much better.
        vectors = vectors - vectors.mean(axis=0)

        local_idx, local_sim = neighbor_table(vectors, table_k)
        width = local_idx.shape[1]
        idx = np.zeros((len(self.movie_ids), width), dtype=np.int32)
        sim = np.full((len(self.movie_ids), width), -np.inf, dtype=np.float32)
        idx[rows] = rows[local_idx]  # map back to positions in self.movie_ids
        sim[rows] = local_sim
        return idx, sim

    def _build(self, train_ratings, allowed, table_k):
        """Build the ratings matrix and its neighbor table, plus the popularity
        numbers the fallback and the pop_* extras use. Returns a dict (the cache entry)."""
        r = train_ratings[train_ratings["movieId"].isin(allowed)]
        # every allowed movie gets a row, even if binarizing leaves it empty
        movie_ids = np.unique(r["movieId"].to_numpy())
        all_rows = np.searchsorted(movie_ids, r["movieId"].to_numpy())
        rating_counts = np.bincount(all_rows, minlength=len(movie_ids))
        liked = (r["rating"] >= LIKE_THRESHOLD).to_numpy()
        like_counts = np.bincount(all_rows[liked], minlength=len(movie_ids))

        # the values that go in the matrix (see binarize / center_ratings in SETTINGS)
        if self.binarize == "liked":
            r = r[liked]
        values = r["rating"].to_numpy(np.float32)
        if self.binarize:
            values = np.ones_like(values)
        elif self.center_ratings:
            values = values - r.groupby("userId")["rating"].transform("mean").to_numpy(np.float32)

        # sparse movies x users matrix: rows are movies, columns are training users
        rows = np.searchsorted(movie_ids, r["movieId"].to_numpy())
        cols, users = pd.factorize(r["userId"])
        matrix = csr_matrix((values, (rows, cols)), shape=(len(movie_ids), len(users)))

        nbr_idx, nbr_sim = neighbor_table(matrix, table_k, shrinkage=self.shrinkage)
        log_likes = np.log1p(like_counts)
        return {
            "train_ref": train_ratings,  # keeps id(train_ratings) valid for the cache key
            "movie_ids": movie_ids,
            "movie_index": {int(m): i for i, m in enumerate(movie_ids)},
            "nbr_idx": nbr_idx,
            "nbr_sim": nbr_sim,
            "fallback": [int(m) for m in movie_ids[np.argsort(-rating_counts, kind="stable")]],
            "rating_counts": rating_counts,
            # same signal as the popularity baseline (number of likes), log-scaled to [0, 1]
            "popularity": log_likes / max(log_likes.max(), 1e-12),
        }

    def _knn_scores(self, rows, weights, exclude, nbr_idx=None, nbr_sim=None):
        """Step 2 of "HOW IT WORKS": score the neighbors of the user's LOVE/LIKE picks.

        rows: the picks' rows; weights: their tier weights (same order);
        exclude: rows that may not be returned (every pick, including SEEN).
        Each neighbor earns similarity ** sim_power * weight, summed per candidate.
        Returns (candidate rows, scores). Uses the ratings table unless another
        neighbor table (the content table) is passed in.
        """
        nbr_idx = self.nbr_idx if nbr_idx is None else nbr_idx
        nbr_sim = self.nbr_sim if nbr_sim is None else nbr_sim
        k_nb = min(self.n_neighbors, nbr_idx.shape[1])
        cand = nbr_idx[rows, :k_nb].ravel()
        sims = nbr_sim[rows, :k_nb].ravel()
        w = np.repeat(weights, k_nb)  # each pick's weight, once per neighbor
        keep = sims > 0  # zero/negative similarity is not evidence of a match
        cand, sims, w = cand[keep], sims[keep].astype(np.float64), w[keep]
        if self.sim_power != 1.0:
            sims = sims ** self.sim_power
        sims = sims * w  # after sim_power, so LOVE's weight isn't squared too
        # sum the points of each candidate that appears more than once
        cand, inverse = np.unique(cand, return_inverse=True)
        scores = np.bincount(inverse, weights=sims, minlength=len(cand))
        keep = ~np.isin(cand, exclude)
        return cand[keep], scores[keep]

    @staticmethod
    def _parse_picks(picks):
        """Turn the user's picks into {movieId: tier}, checking the tier names.
        A plain list of movieIds (what evaluate.py passes) means all "like"."""
        if isinstance(picks, dict):
            sel = {int(m): str(t).lower() for m, t in picks.items()}
        else:
            sel = {int(m): "like" for m in picks}
        bad = set(sel.values()) - set(TIERS)
        if bad:
            raise ValueError(f"unknown tier(s) {sorted(bad)}; use {TIERS}")
        return sel

    def recommend(self, picks, k=5):
        """Recommend k movies.

        picks: {movieId: "love" | "like" | "seen"}, or a plain list of movieIds (all "like").
        Returns k movieIds, best first. No pick of any tier is ever recommended.
        Picks the model doesn't know (not in allowed_movies) are ignored.
        """
        return [m for m, _ in self.recommend_with_scores(picks, k)]

    def recommend_with_scores(self, picks, k=5):
        """Like recommend(), but returns [(movieId, score), ...]. Higher score = more confident.

        Without extras, the score is the tier-weighted sum of similarities to the user's
        LOVE/LIKE picks. With extras on, it is the blended score from _blended_scores().
        Popular-movie fill-ins score 0.
        """
        sel = self._parse_picks(picks)
        favs = set(sel)
        exclude = np.array([self.movie_index[m] for m in sel if m in self.movie_index], dtype=np.int64)
        # only LOVE/LIKE picks that the model knows contribute points
        scored = [(self.movie_index[m], TIER_WEIGHTS[t]) for m, t in sel.items()
                  if t in TIER_WEIGHTS and m in self.movie_index]
        rows = np.array([i for i, _ in scored], dtype=np.int64)
        weights = np.array([w for _, w in scored], dtype=np.float64)
        if rows.size:
            cand, scores = self._knn_scores(rows, weights, exclude)
        else:
            cand, scores = np.array([], int), np.array([])

        if self.pop_weight > 0 or self.pop_penalty > 0 or self.content_weight > 0:
            # extras on: every movie gets a score, keep the best positive ones
            total = self._blended_scores(rows, weights, exclude, cand, scores)
            best = np.argsort(-total, kind="stable")[:k]
            recs = [(int(self.movie_ids[j]), float(total[j])) for j in best if total[j] > 0]
        else:
            # plain KNN: only the picks' neighbors have scores
            best = np.argsort(-scores, kind="stable")[:k]
            recs = [(int(self.movie_ids[cand[j]]), float(scores[j])) for j in best]

        # fill any empty slots with popular movies the user hasn't picked
        chosen = {m for m, _ in recs}
        for movie_id in self.fallback:
            if len(recs) == k:
                break
            if movie_id not in favs and movie_id not in chosen:
                recs.append((movie_id, 0.0))
                chosen.add(movie_id)
        return recs

    def _blended_scores(self, rows, weights, exclude, cand, scores):
        """Step 3 of "HOW IT WORKS": a score for every movie (one per row), with the extras
        applied in order: ratings/content blend -> popularity penalty -> popularity blend.
        cand/scores are the plain ratings-KNN result from _knn_scores(). Picks get -inf."""
        total = np.zeros(len(self.movie_ids))
        # a. ratings + content, each scaled so its best candidate scores 1
        if scores.size:
            total[cand] = (1 - self.content_weight) * scores / scores.max()
        if self.content_weight > 0 and rows.size:
            c_cand, c_scores = self._knn_scores(rows, weights, exclude,
                                                self.content_idx, self.content_sim)
            if c_scores.size:
                total[c_cand] += self.content_weight * c_scores / c_scores.max()
        # b. push down movies with many ratings
        if self.pop_penalty > 0:
            total /= np.maximum(self.rating_counts, 1) ** self.pop_penalty
        # c. mix in plain popularity
        if self.pop_weight > 0:
            if total.max() > 0:
                total /= total.max()
            total = (1 - self.pop_weight) * total + self.pop_weight * self.popularity
        total[exclude] = -np.inf
        return total

    def similar_movies(self, movie_id, n=10, by="ratings"):
        """Handy for sanity checks -- [(movieId, similarity), ...] for one movie (n <= 100).
        by="ratings" uses the ratings table; by="content" uses the tag-genome table
        (only built when the model was fit with content_weight > 0)."""
        if movie_id not in self.movie_index:
            return []
        idx, sim = (self.content_idx, self.content_sim) if by == "content" else (self.nbr_idx, self.nbr_sim)
        if idx is None:
            raise ValueError("no content table: fit with content_weight > 0")
        i = self.movie_index[movie_id]
        return [
            (int(self.movie_ids[j]), round(float(s), 3))
            for j, s in zip(idx[i, :n], sim[i, :n])
            if s > 0
        ]


if __name__ == "__main__":
    # Sanity check: fit a few variants on the training users, then show each pick's most
    # similar movies and each variant's recommendations for one example set of picks.
    # This is for eyeballing whether results make sense; the numbers are in evaluate.py.
    from data import DATASET, MIN_RATINGS, load_data, popular_enough, split_users

    ratings, movies = load_data()
    titles = movies.set_index("movieId")["title"]
    title_to_id = {t: int(m) for m, t in titles.items()}
    train, _, _ = split_users(ratings)
    allowed = popular_enough(train)
    has_genome = load_genome() is not None

    variants = {"ratings KNN": ItemKNNRecommender(n_neighbors=100, sim_power=2)}
    if has_genome:
        variants["+ content 0.5"] = ItemKNNRecommender(n_neighbors=100, sim_power=2, content_weight=0.5)
        variants["content only"] = ItemKNNRecommender(n_neighbors=100, content_weight=1.0)
    variants["+ pop_penalty 0.5"] = ItemKNNRecommender(n_neighbors=100, sim_power=2, pop_penalty=0.5)
    for model in variants.values():
        model.fit(train, allowed)
    model = next(iter(variants.values()))
    counts = train.groupby("movieId").size()

    # Look movies up by title: movieIds aren't the same across every MovieLens release.
    saschas_picks = {
        "Coraline (2009)": "love",
        "Scott Pilgrim vs. the World (2010)": "love",
        "Lord of the Rings: The Fellowship of the Ring, The (2001)": "like",
        "Dr. Horrible's Sing-Along Blog (2008)": "like",
        "Kick-Ass (2010)": "seen",
    }
    # ml-1m stops at 2000, so fall back to classics it does have
    classics = {
        "Toy Story (1995)": "love",
        "Princess Bride, The (1987)": "love",
        "Matrix, The (1999)": "love",
        "Back to the Future (1985)": "like",
        "Groundhog Day (1993)": "seen",
    }

    # keep only picks the model can use, and say why any were dropped
    picks = {}
    for title, tier in saschas_picks.items():
        movie_id = title_to_id.get(title)
        if movie_id is None:
            print(f"(not in {DATASET}: {title})")
        elif movie_id not in model.movie_index:
            print(f"(fewer than {MIN_RATINGS} training ratings: {title})")
        else:
            picks[movie_id] = tier
            if has_genome and movie_id not in load_genome()[0]:
                print(f"(no tag-genome scores, ratings only: {title})")
    if sum(t != "seen" for t in picks.values()) < 3:
        print("Too few of those picks are usable here, using classics instead.")
        picks = {title_to_id[t]: tier for t, tier in classics.items() if t in title_to_id}
    favs = [m for m, t in picks.items() if t != "seen"]  # LOVE/LIKE picks only

    def label(m):
        return f"{titles.get(m, m)}  [{counts.get(m, 0):,} ratings]"

    print("\nPicks:")
    for m, tier in picks.items():
        print(f"    {tier.upper():<5} {label(m)}")

    for movie_id in favs[:3]:
        print(f"\nMost similar to {titles[movie_id]}:")
        print("  by ratings:")
        for m, similar in model.similar_movies(movie_id, n=5):
            print(f"    {similar:.3f}   {label(m)}")
        if has_genome:
            print("  by content (tag genome):")
            for m, similar in variants["content only"].similar_movies(movie_id, n=5, by="content"):
                print(f"    {similar:.3f}   {label(m)}")

    for name, v in variants.items():
        print(f"\nRecommendations, {name}:")
        for m, score in v.recommend_with_scores(picks):
            print(f"    {score:6.3f}   {label(m)}")