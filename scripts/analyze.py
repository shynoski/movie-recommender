import pandas as pd

ratings = pd.read_csv("data/ml-latest-small/ratings.csv")
movies = pd.read_csv("data/ml-latest-small/movies.csv")

print(ratings.shape, movies.shape)
print(ratings["rating"].value_counts().sort_index())
print(ratings.groupby("userId").size().describe())    # ratings per user
print(ratings.groupby("movieId").size().describe())   # ratings per movie