"""Combine each review/label Excel pair into one CSV, check it, and print statistics.

Usage (from the repository root):
    python scripts/build_labeled_csv.py --data-dir data            # CSV files, checks, checksums
    python scripts/build_labeled_csv.py --data-dir data --words    # also word statistics (needs PyThaiNLP)

Outputs in --data-dir:
    perfume_labeled.csv, indigo_labeled.csv   UTF-8 with BOM; columns id, review, label
    SHA256SUMS                                checksums; verify with `sha256sum -c SHA256SUMS`
"""
import argparse
import hashlib
import os
import sys

import pandas as pd

DATASETS = {
    "perfume": {"reviews": "perfume_review_cleaned_dataset.xlsx",
                "labels": "perfume_class_cleaned_dataset.xlsx", "n": 4656},
    "indigo": {"reviews": "indigo_review_cleaned_dataset.xlsx",
               "labels": "indigo_class_cleaned_dataset.xlsx", "n": 3279},
}


def load_pair(data_dir, cfg):
    """Read one single-column review file and its label file (no header rows)."""
    reviews = pd.read_excel(os.path.join(data_dir, cfg["reviews"]), header=None)
    labels = pd.read_excel(os.path.join(data_dir, cfg["labels"]), header=None)
    if reviews.shape[1] != 1 or labels.shape[1] != 1:
        sys.exit(f"Expected one column in each file, found {reviews.shape[1]} and {labels.shape[1]}")
    reviews = reviews[0].fillna("").astype(str).str.strip()
    labels = labels[0]
    if len(reviews) != len(labels):
        sys.exit(f"{cfg['reviews']}: {len(reviews)} reviews but {len(labels)} labels")
    if labels.isna().any() or not set(labels.astype(int).unique()) <= {0, 1}:
        sys.exit(f"{cfg['labels']}: labels must be 0 or 1")
    return reviews, labels.astype(int)


def word_stats(reviews):
    from pythainlp import word_tokenize
    tokens = [[t for t in word_tokenize(r, engine="longest") if t.strip()] for r in reviews]
    n = pd.Series([len(t) for t in tokens])
    vocab = {w for t in tokens for w in t}
    return n.mean(), n.max(), len(vocab)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--words", action="store_true", help="word statistics with PyThaiNLP (slower)")
    args = ap.parse_args()

    files = []
    for name, cfg in DATASETS.items():
        reviews, labels = load_pair(args.data_dir, cfg)
        if len(reviews) != cfg["n"]:
            print(f"NOTE {name}: {len(reviews)} rows (the article reports {cfg['n']})")
        n_empty = int((reviews == "").sum())
        if n_empty:
            print(f"NOTE {name}: {n_empty} empty reviews")

        out = os.path.join(args.data_dir, f"{name}_labeled.csv")
        pd.DataFrame({"id": range(len(reviews)), "review": reviews, "label": labels}).to_csv(
            out, index=False, encoding="utf-8-sig", lineterminator="\n")

        pos, neg = int((labels == 1).sum()), int((labels == 0).sum())
        chars = reviews.str.len()
        print(f"\n{name}: {len(reviews):,} reviews")
        print(f"  positive (1): {pos:,} ({pos / len(labels):.1%})   negative (0): {neg:,} ({neg / len(labels):.1%})")
        print(f"  characters  : mean {chars.mean():.1f}, max {chars.max():,}")
        if args.words:
            mean_w, max_w, vocab = word_stats(reviews)
            print(f"  words       : mean {mean_w:.1f}, max {max_w:,}, vocabulary {vocab:,}")
        files += [cfg["reviews"], cfg["labels"], f"{name}_labeled.csv"]

    with open(os.path.join(args.data_dir, "SHA256SUMS"), "w", newline="\n") as f:
        for fn in files:
            f.write(f"{sha256(os.path.join(args.data_dir, fn))}  {fn}\n")
    print(f"\nWrote CSV files and SHA256SUMS to {args.data_dir}/")


if __name__ == "__main__":
    main()
