import urllib.request, zipfile, pathlib

URL = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)

zip_path = DATA_DIR / "ml-latest-small.zip"
urllib.request.urlretrieve(URL, zip_path)
with zipfile.ZipFile(zip_path) as z:
    z.extractall(DATA_DIR)
print("Done:", list((DATA_DIR / "ml-latest-small").iterdir()))