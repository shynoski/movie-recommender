"""Download a MovieLens dataset into data/<name>/.

    python scripts/download_data.py            # ml-25m (default, ~250 MB zip)
    python scripts/download_data.py ml-32m     # ~240 MB zip, ~900 MB unpacked
"""
import argparse
import pathlib
import urllib.request
import zipfile
 
BASE_URL = "https://files.grouplens.org/datasets/movielens/{}.zip"
DATASETS = ["ml-latest-small", "ml-1m", "ml-25m", "ml-32m"]
DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
 
 
def progress(blocks, block_size, total):
    if total > 0:
        done = min(blocks * block_size, total)
        print(f"\r  {done / 1e6:6.1f} / {total / 1e6:.1f} MB", end="", flush=True)
 
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("name", nargs="?", default="ml-25m", choices=DATASETS)
    parser.add_argument("--keep-zip", action="store_true", help="don't delete the zip afterwards")
    args = parser.parse_args()
 
    target = DATA_DIR / args.name
    if target.exists() and any(target.iterdir()):
        print(f"Already downloaded: {target}")
        return
 
    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / f"{args.name}.zip"
    print(f"Downloading {args.name} ...")
    urllib.request.urlretrieve(BASE_URL.format(args.name), zip_path, reporthook=progress)
    print("\nExtracting ...")
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(DATA_DIR)
    if not args.keep_zip:
        zip_path.unlink()
    print("Done:", sorted(p.name for p in target.iterdir()))
 
 
if __name__ == "__main__":
    main()
 