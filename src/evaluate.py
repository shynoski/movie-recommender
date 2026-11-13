"""Evaluate recommenders with Precision@5 and Recall@5 on held-out users.

Each validation (or test) user is one case from make_eval_cases() in data.py:
5 movies they liked are given to the model as "favorites", and the rest of the
movies they liked are hidden. The model recommends 5 movies, and we check how
many of them are hidden liked movies:

  Precision@5  share of the 5 recommendations that are hidden liked movies
  Recall@5     share of the user's hidden liked movies that made the top 5

Both are averaged over all users. Every model is compared to the popularity
baseline (target to beat) and the random baseline (lower bound).

Three extra columns describe what kind of movies a model recommends:

  AvgRatings  average number of training ratings of the recommended movies
              (high = mostly blockbusters)
  Tail%       share of recommendations outside the popular "head", the small
              set of most-rated movies that together hold half of all ratings
  Cover%      share of all allowed movies recommended to at least one user

Workflow:

1. Compare the models in knn_variants() on the validation users:
       python src/evaluate.py            (add --plot to save results/tradeoff_validation.png)
2. Pick the winner and put it in final_model().
3. Run the test users ONCE, at the very end, with the final settings:
       python src/evaluate.py --test
   Test results are only fair if nothing was tuned on them.

robustness.py checks whether the validation differences are bigger than noise.
"""

import argparse
from pathlib import Path

import numpy as np

from baseline import PopularityRecommender, RandomReccommender
from data import load_data, load_genome, make_eval_cases, popular_enough, split_users
from knn import ItemKNNRecommender

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def precision_at_k(recs, hidden, k=5):
    """Fraction of the k recommendations that are hidden liked movies."""
    return len(set(recs[:k]) & hidden) / k


def recall_at_k(recs, hidden, k=5):
    """Fraction of the hidden liked movies that appear in the k recommendations."""
    if not hidden:
        return 0.0
    return len(set(recs[:k]) & hidden) / len(hidden)


def evaluate(model, cases, k=5):
    """Average Precision@k and Recall@k over all evaluation cases.
    A quick two-number version of score_model(), handy from a notebook or the shell;
    main() uses score_model() instead."""
    precisions, recalls = [], []
    for case in cases:
        recs = model.recommend(case["favorites"], k=k)
        precisions.append(precision_at_k(recs, case["hidden"], k))
        recalls.append(recall_at_k(recs, case["hidden"], k))
    return float(np.mean(precisions)), float(np.mean(recalls))


class PopularityStats:
    """Training rating counts, and which movies count as the popular "head".
    The head is the smallest set of most-rated movies that together hold `head_share`
    (default half) of all training ratings; every other allowed movie is the "tail"."""

    def __init__(self, train_ratings, allowed_movies, head_share=0.5):
        counts = (train_ratings[train_ratings["movieId"].isin(allowed_movies)]
                  .groupby("movieId").size().sort_values(ascending=False))
        self.counts = counts.to_dict()
        n_head = int(np.searchsorted(counts.cumsum().to_numpy(), head_share * counts.sum())) + 1
        self.head = set(counts.index[:n_head])
        self.n_allowed = len(counts)


def score_model(model, cases, pop, k=5):
    """All metrics for one model (see the module docstring), from a single pass of
    recommendations. Returns a dict: precision, recall, avg_ratings, tail, coverage."""
    precisions, recalls, rec_counts, tail = [], [], [], []
    recommended = set()
    for case in cases:
        recs = model.recommend(case["favorites"], k=k)
        precisions.append(precision_at_k(recs, case["hidden"], k))
        recalls.append(recall_at_k(recs, case["hidden"], k))
        rec_counts += [pop.counts.get(m, 0) for m in recs]
        tail += [m not in pop.head for m in recs]
        recommended.update(recs)
    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "avg_ratings": float(np.mean(rec_counts)),
        "tail": float(np.mean(tail)),
        "coverage": len(recommended) / pop.n_allowed,
    }


REFERENCE = "ItemKNN K=100"  # the plain KNN every tweak is compared against


def knn_variants():
    """KNN models to compare on validation: the plain reference plus one tweak at a time,
    so any difference from REFERENCE comes from that one tweak. The settings are
    explained in knn.py's module docstring. Edit this list to explore further
    (e.g. combine the tweaks that win). Content variants need the tag genome (ml-25m).

    Notes from earlier experiments (ml-1m): sim_power 2 helped most;
    binarize/shrinkage barely mattered."""
    models = {
        REFERENCE: ItemKNNRecommender(n_neighbors=100),
        "+ sim_power 2": ItemKNNRecommender(n_neighbors=100, sim_power=2),
        "+ pop_penalty 0.25": ItemKNNRecommender(n_neighbors=100, pop_penalty=0.25),
        "+ pop_penalty 0.5": ItemKNNRecommender(n_neighbors=100, pop_penalty=0.5),
        "+ pop_penalty 1.0": ItemKNNRecommender(n_neighbors=100, pop_penalty=1.0),
    }
    if load_genome() is not None:
        models.update({
            "+ content 0.25": ItemKNNRecommender(n_neighbors=100, content_weight=0.25),
            "+ content 0.5": ItemKNNRecommender(n_neighbors=100, content_weight=0.5),
            "+ content 0.75": ItemKNNRecommender(n_neighbors=100, content_weight=0.75),
            "content only": ItemKNNRecommender(n_neighbors=100, content_weight=1.0),
            "+ sim_power 2 + content 0.5": ItemKNNRecommender(
                n_neighbors=100, sim_power=2, content_weight=0.5),
        })
    return models


def final_model():
    """The single model to report on the test set (and the one the demo uses).
    Set this once validation picks a winner; it is currently the plain reference."""
    return ItemKNNRecommender(n_neighbors=100)


def plot_tradeoff(results, path):
    """Scatter plot of Precision@5 vs long-tail share, one dot per model.
    Up and to the right is better on both: accurate AND not just blockbusters."""
    import matplotlib
    matplotlib.use("Agg")  # save to file, no window
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    offsets = [(5, 4), (5, -11), (5, 13), (5, -20)]  # staggered so clustered labels don't collide
    shown = [(n, m) for n, m in results.items() if n != "Random"]  # Random is far off the chart
    shown.sort(key=lambda nm: (nm[1]["tail"], nm[1]["precision"]))
    for i, (name, m) in enumerate(shown):
        ax.scatter(100 * m["tail"], m["precision"], s=40)
        ax.annotate(name, (100 * m["tail"], m["precision"]), fontsize=8,
                    xytext=offsets[i % len(offsets)], textcoords="offset points")
    ax.set_xlabel("Recommendations from the long tail (%)")
    ax.set_ylabel("Precision@5")
    ax.set_title("Accuracy vs. niche-ness (validation)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(exist_ok=True)
    fig.savefig(path, dpi=150)
    print(f"\nSaved {path}")


def main():
    """Fit every model on the training users, score it on validation (or test) users,
    and print one table row per model."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="evaluate on the test split")
    parser.add_argument("--plot", action="store_true", help="save results/tradeoff_<split>.png")
    args = parser.parse_args()

    ratings, _ = load_data()
    train, val, test = split_users(ratings)
    allowed = popular_enough(train)
    split_name = "TEST" if args.test else "VALIDATION"
    cases = make_eval_cases(test if args.test else val, allowed)
    pop = PopularityStats(train, allowed)

    models = {
        "Random": RandomReccommender(),
        "Popularity": PopularityRecommender(),
    }
    if args.test:  # test split: only the final pick, never the whole sweep
        models["Final"] = final_model()
    else:
        models.update(knn_variants())

    print(f"{split_name}: {len(cases)} users, {len(allowed)} candidate movies "
          f"({len(pop.head)} popular 'head' movies collect half of all ratings)\n")
    header = f"{'Model':<30}{'Prec@5':>8}{'Rec@5':>8}{'AvgRatings':>12}{'Tail%':>7}{'Cover%':>8}"
    print(header)
    print("-" * len(header))
    results = {}
    for name, model in models.items():
        model.fit(train, allowed)
        m = results[name] = score_model(model, cases, pop)
        print(f"{name:<30}{m['precision']:>8.4f}{m['recall']:>8.4f}{m['avg_ratings']:>12,.0f}"
              f"{100 * m['tail']:>7.1f}{100 * m['coverage']:>8.1f}")

    if args.plot:
        plot_tradeoff(results, RESULTS_DIR / f"tradeoff_{split_name.lower()}.png")


if __name__ == "__main__":
    main()