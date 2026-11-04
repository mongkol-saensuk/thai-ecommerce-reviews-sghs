"""Cleaning rules used to produce the released review files (Step 3.1 of the article).

Usage as a module:
    from clean_review import clean_review
    clean_review("ส่งเร็ว 2 วัน ดีมาก!! 👍 Good")   # -> "ส่งเร็ว วัน ดีมาก"

Usage from the command line (one review per row, single column, no header):
    python scripts/clean_review.py raw_reviews.xlsx cleaned_reviews.xlsx
"""
import re
import sys

from pythainlp.util import normalize


def clean_review(text: str) -> str:
    text = normalize(str(text))                       # normalise Thai character order
    text = re.sub(r"[0-9\u0E50-\u0E59]", " ", text)   # Arabic and Thai digits
    text = re.sub(r"[A-Za-z]", " ", text)             # Latin letters
    text = re.sub(r"[^\u0E00-\u0E7F\s]", " ", text)   # symbols, emoji, punctuation
    return re.sub(r"\s+", " ", text).strip()


if __name__ == "__main__":
    import pandas as pd

    if len(sys.argv) != 3:
        sys.exit(__doc__)
    raw = pd.read_excel(sys.argv[1], header=None)[0].fillna("")
    pd.DataFrame(raw.map(clean_review)).to_excel(sys.argv[2], header=False, index=False)
    print(f"Cleaned {len(raw):,} reviews -> {sys.argv[2]}")
