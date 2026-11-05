"""
Evaluate recommenders with Precision@k and Recall@k on held-out users.

Run from the repo root:
    python -m src.evaluate
    python -m src.evaluate --test # RUN ONCE AT THE END WITH FINAL SETTINGS

"""

import argparse
import numpy as np

from src.baseline import PopularityRecommender, RandomReccommender
from src.data import load_data, make_eval_cases, popular_enough, split_users
from src.knn import ItemKNNRecommender

def precision_at_k(recs, hidden, k=5):
    """Fraction of the k recommendations that are hidden liked movies."""
    ...

def recall_at_k(recs, hidden, k=5):
    """Fraction of the hidden liked movies that appear in the k recommendations."""
    ...

def evaluate(model, cases, k=5):
    """Average Precision@k and Recall@k over all evaluation cases."""
    ...

def main(): 
    ...

if __name__ == "__main__":
    main()