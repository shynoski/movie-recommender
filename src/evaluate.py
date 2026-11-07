"""
Evaluate recommenders with Precision@k and Recall@k on held-out users.

Run from the repo root:
    python -m src.evaluate
    python -m src.evaluate --test # RUN ONCE AT THE END WITH FINAL SETTINGS

"""

import argparse
import numpy as np

from baseline import PopularityRecommender, RandomReccommender
from data import load_data, make_eval_cases, popular_enough, split_users
from knn import ItemKNNRecommender

def precision_at_k(recs, hidden, k=5):
    """Fraction of the k recommendations that are hidden liked movies."""
    return len(set(recs[:k]) & hidden ) / k

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

def main(): 
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="evaluate on the test split")
    args = parser.parse_args()

    ratings, _ = load_data()
    train, val, test = split_users(ratings)
    allowed = popular_enough(train)
    split_name = "TEST" if args.test else "VALIDATION"
    cases = make_eval_cases(test if args.test else val, allowed)

    models = {
        "Random": RandomReccommender(),
        "Popularity": PopularityRecommender(),
    }
    for n in [10, 20, 50, 100]:
        models[f"ItemKNN K={n}"] = ItemKNNRecommender(n_neighbors=n)
        models[f"ItemKNN K={n} centered"] = ItemKNNRecommender(n_neighbors=n, center_ratings=True)

    print(f"{split_name}: {len(cases)} users, {len(allowed)} candidate movies\n")
    print(f"{'Model':<26}{'Precision@5':>12}{'Recall@5':>10}")
    print("-" * 48)
    for name, model in models.items():
        model.fit(train, allowed)
        p, r = evaluate(model, cases)
        print(f"{name:<26}{p:>12.4f}{r:>10.4f}")        

if __name__ == "__main__":
    main()