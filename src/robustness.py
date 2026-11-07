"""Does KNN's lead over popularity survive different random choices of favorites?

Keeps the official train/val/test split (test stays untouched), re-draws each
validation user's 5 favorites with several seeds, and reports:
  1. mean +/- std of Precision@5 and Recall@5 across seeds
  2. a bootstrap 95% CI (resampling users) for each KNN model's relative
     Recall@5 improvement over popularity

Run: python src/robustness.py
"""
import numpy as np

from baseline import PopularityRecommender
from data import load_data, make_eval_cases, popular_enough, split_users
from evaluate import precision_at_k, recall_at_k
from knn import ItemKNNRecommender

SEEDS = range(10)
K_VALUES = [10, 20, 50, 100]
N_BOOTSTRAP = 2000


def per_user_scores(model, cases, k=5):
    """Arrays of per-user precision and recall (one entry per eval case)."""
    p, r = [], []
    for case in cases:
        recs = model.recommend(case["favorites"], k=k)
        p.append(precision_at_k(recs, case["hidden"], k))
        r.append(recall_at_k(recs, case["hidden"], k))
    return np.array(p), np.array(r)


def main():
    ratings, _ = load_data()
    train, val, _ = split_users(ratings)
    allowed = popular_enough(train)

    models = {"Popularity": PopularityRecommender()}
    for n in K_VALUES:
        models[f"ItemKNN K={n}"] = ItemKNNRecommender(n_neighbors=n)
    for model in models.values():
        model.fit(train, allowed)

    # results[name] = (precision array, recall array), each shaped (n_seeds, n_users)
    results = {}
    for name, model in models.items():
        ps, rs = [], []
        for seed in SEEDS:
            cases = make_eval_cases(val, allowed, seed=seed)
            p, r = per_user_scores(model, cases)
            ps.append(p)
            rs.append(r)
        results[name] = (np.array(ps), np.array(rs))

    n_users = results["Popularity"][1].shape[1]
    print(f"VALIDATION: {n_users} users, {len(SEEDS)} seeds\n")
    print(f"{'Model':<18}{'Precision@5':>18}{'Recall@5':>20}")
    print("-" * 56)
    for name, (p, r) in results.items():
        pm, rm = p.mean(axis=1), r.mean(axis=1)  # one mean per seed
        print(f"{name:<18}{pm.mean():>10.4f} ± {pm.std():.4f}{rm.mean():>12.4f} ± {rm.std():.4f}")

    # Bootstrap: resample users (each user's recall averaged over seeds)
    rng = np.random.default_rng(0)
    pop_r = results["Popularity"][1].mean(axis=0)
    idx = rng.integers(0, n_users, size=(N_BOOTSTRAP, n_users))
    print(f"\nRelative Recall@5 improvement over popularity (95% bootstrap CI):")
    for name, (_, r) in results.items():
        if name == "Popularity":
            continue
        knn_r = r.mean(axis=0)
        point = knn_r.mean() / pop_r.mean() - 1
        rel = knn_r[idx].mean(axis=1) / pop_r[idx].mean(axis=1) - 1
        lo, hi = np.percentile(rel, [2.5, 97.5])
        print(f"  {name:<16}{point:>+8.1%}   [{lo:+.1%}, {hi:+.1%}]")


if __name__ == "__main__":
    main()