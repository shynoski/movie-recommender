"""Does KNN's lead over popularity survive different random choices of favorites?

Keeps the official train/val/test split (test stays untouched), re-draws each
validation user's 5 favorites with several seeds, and reports:
  1. mean +/- std of Precision@5 and Recall@5 across seeds
  2. a bootstrap 95% CI (resampling users) for each KNN model's relative
     Recall@5 improvement over popularity
"""
import numpy as np

from baseline import PopularityRecommender
from data import load_data, make_eval_cases, popular_enough, split_users
from evaluate import REFERENCE, knn_variants, precision_at_k, recall_at_k

SEEDS = range(10)
N_BOOTSTRAP = 2000


def per_user_scores(model, cases, k=5):
    """Arrays of per-user precision and recall (one entry per eval case)."""
    p, r = [], []
    for case in cases:
        recs = model.recommend(case["favorites"], k=k)
        p.append(precision_at_k(recs, case["hidden"], k))
        r.append(recall_at_k(recs, case["hidden"], k))
    return np.array(p), np.array(r)


def relative_ci(model_scores, base_scores, idx):
    """Point estimate and 95% CI of mean(model) / mean(base) - 1 over bootstrap samples idx."""
    point = model_scores.mean() / base_scores.mean() - 1
    rel = model_scores[idx].mean(axis=1) / base_scores[idx].mean(axis=1) - 1
    lo, hi = np.percentile(rel, [2.5, 97.5])
    return point, lo, hi


def fmt(point, lo, hi):
    mark = "*" if lo > 0 or hi < 0 else " "  # * = CI excludes zero
    return f"{point:>+7.1%} [{lo:+.1%}, {hi:+.1%}]{mark}"


def main():
    ratings, _ = load_data()
    train, val, _ = split_users(ratings)
    allowed = popular_enough(train)

    models = {"Popularity": PopularityRecommender(), **knn_variants()}
    for model in models.values():
        model.fit(train, allowed)

    seed_cases = [make_eval_cases(val, allowed, seed=seed) for seed in SEEDS]

    # results[name] = (precision array, recall array), each shaped (n_seeds, n_users)
    results = {}
    for name, model in models.items():
        ps, rs = zip(*(per_user_scores(model, cases) for cases in seed_cases))
        results[name] = (np.array(ps), np.array(rs))

    n_users = results["Popularity"][1].shape[1]
    print(f"VALIDATION: {n_users} users, {len(SEEDS)} seeds\n")
    print(f"{'Model':<30}{'Precision@5':>18}{'Recall@5':>20}")
    print("-" * 68)
    for name, (p, r) in results.items():
        pm, rm = p.mean(axis=1), r.mean(axis=1)  # one mean per seed
        print(f"{name:<30}{pm.mean():>10.4f} ± {pm.std():.4f}{rm.mean():>12.4f} ± {rm.std():.4f}")

    # Bootstrap: resample users (each user's score averaged over seeds).
    # The same resampled users are used for every comparison, which is what makes it paired.
    rng = np.random.default_rng(0)
    idx = rng.integers(0, n_users, size=(N_BOOTSTRAP, n_users))
    user_p = {name: p.mean(axis=0) for name, (p, _) in results.items()}
    user_r = {name: r.mean(axis=0) for name, (_, r) in results.items()}

    print("\nRelative Recall@5 improvement over popularity (95% bootstrap CI):")
    for name in results:
        if name != "Popularity":
            print(f"  {name:<30}{fmt(*relative_ci(user_r[name], user_r['Popularity'], idx))}")

    print(f"\nPaired comparison vs {REFERENCE} (95% bootstrap CI, * = CI excludes 0):")
    print(f"  {'Model':<30}{'Precision@5':>26}{'Recall@5':>28}")
    for name in results:
        if name in ("Popularity", REFERENCE):
            continue
        p_ci = relative_ci(user_p[name], user_p[REFERENCE], idx)
        r_ci = relative_ci(user_r[name], user_r[REFERENCE], idx)
        print(f"  {name:<30}{fmt(*p_ci):>28}{fmt(*r_ci):>28}")


if __name__ == "__main__":
    main()