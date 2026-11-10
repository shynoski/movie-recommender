"""
Evaluate recommenders with Precision@k and Recall@k on held-out users.

Run from the repo root:
    python -m src.evaluate
    python -m src.evaluate --test # RUN ONCE AT THE END WITH FINAL SETTINGS

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
    """Average Precision@k and Recall@k over all evaluation cases."""
    precisions, recalls = [], []
    for case in cases:
        recs = model.recommend(case["favorites"], k=k)
        precisions.append(precision_at_k(recs, case["hidden"], k))
        recalls.append(recall_at_k(recs, case["hidden"], k))
    return float(np.mean(precisions)), float(np.mean(recalls))
 
 
class PopularityStats:
    """Training rating counts, and which movies count as the popular "head"."""
 
    def __init__(self, train_ratings, allowed_movies, head_share=0.5):
        counts = (train_ratings[train_ratings["movieId"].isin(allowed_movies)]
                  .groupby("movieId").size().sort_values(ascending=False))
        self.counts = counts.to_dict()
        n_head = int(np.searchsorted(counts.cumsum().to_numpy(), head_share * counts.sum())) + 1
        self.head = set(counts.index[:n_head])
        self.n_allowed = len(counts)
 
 
def score_model(model, cases, pop, k=5):
    """All metrics for one model, from a single pass of recommendations."""
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
    """KNN models to compare: the plain reference plus one tweak at a time.
    Edit this list to explore further (e.g. combine the tweaks that win).
    Earlier experiments (ml-1m): sim_power 2 helped most; binarize/shrinkage barely mattered."""
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
    """The single model to report on the test set. Set this once validation picks a winner."""
    return ItemKNNRecommender(n_neighbors=100)
 
 
def plot_tradeoff(results, path):
    """Precision vs long-tail share: up and to the right is better on both."""
    import matplotlib
    matplotlib.use("Agg")
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="evaluate on the test split")
    parser.add_argument("--plot", action="store_true", help="save results/tradeoff.png")
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
 