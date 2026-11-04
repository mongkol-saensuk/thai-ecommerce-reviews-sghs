r"""Cleaning rules used for the released review files (Step 3.1 of the article).

Each rule calls re.sub(pattern, "", text), deleting every matched character:
    [0-9]                 Arabic digits
    [A-Za-z]              Latin (English) letters
    [^\u0E00-\u0E7F\s]    any character outside the Thai Unicode block other than whitespace
                          (special symbols, emoji, punctuation)
Thai characters are kept unchanged (no Thai normalisation). Leading and trailing spaces are
removed; runs of spaces left around deleted characters are kept, because whitespace
tokens are discarded after word segmentation.

Usage as a module:
    from clean_review import clean_review
    clean_review("ส่งเร็ว 2 วัน ดีมาก!! 👍 Good")   # -> "ส่งเร็ว  วัน ดีมาก"

Usage from the command line (one review per row, single column, no header):
    python scripts/clean_review.py raw_reviews.xlsx cleaned_reviews.xlsx
"""
import re
import sys


def clean_review(text) -> str:
    text = re.sub(r"[0-9]", "", str(text))           # Arabic digits
    text = re.sub(r"[A-Za-z]", "", text)             # Latin letters
    text = re.sub(r"[^\u0E00-\u0E7F\s]", "", text)   # symbols, emoji, punctuation
    return text.strip()


if __name__ == "__main__":
    import pandas as pd

    if len(sys.argv) != 3:
        sys.exit(__doc__)
    raw = pd.read_excel(sys.argv[1], header=None)[0].fillna("")
    pd.DataFrame(raw.map(clean_review)).to_excel(sys.argv[2], header=False, index=False)
    print(f"Cleaned {len(raw):,} reviews -> {sys.argv[2]}")
