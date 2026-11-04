# Thai e-commerce customer reviews with opinion labels, and SG-HS workflow notebook

[![Data: CC BY 4.0](https://img.shields.io/badge/data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Code: MIT](https://img.shields.io/badge/code-MIT-blue.svg)](LICENSE)
[![Open experiments in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongkol-saensuk/thai-ecommerce-reviews-sghs/blob/main/notebooks/SG_HS_MethodsX_experiments.ipynb)

This repository contains two labelled datasets of Thai customer reviews from the Shopee and Lazada marketplaces (**perfume**, 4,656 reviews; **indigo-dyed products**, 3,279 reviews), each review labelled **positive (1)** or **negative (0)**. It also contains the Google Colab notebook that classifies the reviews with Word2Vec embeddings, in particular **Skip-gram with hierarchical softmax (SG-HS)**, followed by CNN, BiLSTM and RNN classifiers.

The repository holds the data and code for the method article *"A step-by-step workflow using Skip-gram word embedding with hierarchical softmax for classifying Thai customer opinions on e-commerce products"* by C. Chantawat, S. Witchakool, A. Choompol and M. Saensuk. The article citation will be added here once it is published.

[สรุปภาษาไทยอยู่ท้ายเอกสาร](#สรุปภาษาไทย)

## Contents

- [Repository structure](#repository-structure)
- [Datasets](#datasets)
  - [Data quality notes](#data-quality-notes)
- [File format and loading](#file-format-and-loading)
- [Notebooks](#notebooks)
- [Reproducibility](#reproducibility)
- [How to cite](#how-to-cite)
- [License and terms of use](#license-and-terms-of-use)
- [Authors and contact](#authors-and-contact)
- [สรุปภาษาไทย](#สรุปภาษาไทย)

## Repository structure

```
thai-ecommerce-reviews-sghs/
├── data/
│   ├── perfume_review_cleaned_dataset.xlsx   perfume reviews  (4,656 rows)
│   ├── perfume_class_cleaned_dataset.xlsx    perfume labels   (4,656 rows)
│   ├── indigo_review_cleaned_dataset.xlsx    indigo reviews   (3,279 rows)
│   ├── indigo_class_cleaned_dataset.xlsx     indigo labels    (3,279 rows)
│   ├── perfume_labeled.csv                   perfume reviews and labels in one file
│   ├── indigo_labeled.csv                    indigo reviews and labels in one file
│   ├── SHA256SUMS                            checksums of the data files
│   └── LICENSE                               CC BY 4.0
├── notebooks/
│   └── SG_HS_MethodsX_experiments.ipynb      experiments of the article (Steps 3.2–6, 108 runs)
├── scripts/
│   ├── clean_review.py                       cleaning rules of the released reviews
│   └── build_labeled_csv.py                  rebuilds the CSV files, statistics and checksums
├── CITATION.cff
├── LICENSE                                   MIT (code)
├── requirements.txt
└── README.md
```

The notebook reads the Excel files by name, so keep the file names unchanged.

## Datasets

| Item | Perfume dataset | Indigo dataset |
|:--|:--|:--|
| Products | Perfume | Indigo-dyed shawls, shirts, skirts and bags |
| Sources | Shopee, Lazada | Shopee, Lazada |
| Extraction tool | WebHarvy | Comment Exporter |
| Posting period | 18 Jan 2022 – 20 Jun 2022 | Jan 2020 – Apr 2024 |
| Date collected | 15 Sep 2023 | Apr 2024 |
| Number of reviews | 4,656 | 3,279 |
| Positive (1) | 2,281 (49.0%) | 2,491 (76.0%) |
| Negative (0) | 2,375 (51.0%) | 788 (24.0%) |
| Review length, mean / max (characters) | 58.8 / 1,124 | 84.2 / 584 |
| Review length, mean / max (words)\* | 14.7 / 236 | 20.4 / 150 |
| Vocabulary size (distinct words)\* | 4,122 | 3,320 |

\* After word segmentation with PyThaiNLP `word_tokenize(text, engine="longest")`, with whitespace tokens removed.

### Collection

Each extracted record contained a comment identifier, the posting date and the review text. Only the review text was kept. User names, comment identifiers, dates and other personal identifiers are **not** included.

### Labelling

Three experts labelled every review independently. The final label is the majority vote, that is, the label chosen by at least two of the three annotators.

| Label | Meaning |
|:--:|:--|
| 1 | Positive opinion |
| 0 | Negative opinion |

### Cleaning

The released reviews are already cleaned. Python's `re.sub(pattern, "", text)` was applied three times to each review, deleting every matched character:

| Pattern | Removes |
|:--|:--|
| `[0-9]` | Arabic digits |
| `[A-Za-z]` | Latin (English) letters |
| `[^\u0E00-\u0E7F\s]` | any character outside the Thai Unicode block (U+0E00–U+0E7F) other than whitespace: special symbols, emoji and punctuation |

Thai characters were kept unchanged. No Thai normalisation was applied, so misspellings, repeated characters (for example ค่าา and เร็วๆๆๆ) and duplicated vowel or tone marks remain as written. Thai digits (๐–๙) are not matched by `[0-9]`.

The same rules are in [`scripts/clean_review.py`](scripts/clean_review.py), for cleaning new reviews in the same way:

```python
import re

def clean_review(text) -> str:
    text = re.sub(r"[0-9]", "", str(text))           # Arabic digits
    text = re.sub(r"[A-Za-z]", "", text)             # Latin letters
    text = re.sub(r"[^\u0E00-\u0E7F\s]", "", text)   # symbols, emoji, punctuation
    return text.strip()
```

### Data quality notes

The files are released exactly as they were used in the article, so that row numbers, labels and the train/test split stay the same. Please note:

| Item | Perfume | Indigo |
|:--|--:|--:|
| Empty reviews (nothing left after cleaning) | 2 (rows 1258, 2265) | 8 (rows 758, 835, 2593, 2598, 2617, 2618, 2623, 2936) |
| Exact duplicate reviews (copies after the first) | 33 | 199 |
| Reviews with two or more consecutive spaces | 628 | 0 |
| Reviews with leading or trailing spaces | 473 | 0 |
| Reviews that still contain Thai digits | 0 | 4 (rows 1070, 1477, 1734, 1772) |

Spaces do not affect the models, because whitespace tokens are discarded after word segmentation. Empty reviews are empty cells in the Excel files (row 1258 of the perfume file contains only spaces) and empty strings in the CSV files. If you build a new benchmark from these data, consider removing empty and duplicate reviews before splitting, so that identical reviews cannot appear in both the training and the test set.

### Examples (perfume dataset)

The reviews below are copied verbatim from `perfume_review_cleaned_dataset.xlsx`; `id` is the row number starting at 0.

| id | Review (Thai, as in the file) | English translation | Label |
|:--:|:--|:--|:--:|
| 2 | สินค้ามีคุณภาพดีมากครับ ทดลองใช้แล้วใช้งานได้ดีแนะนำร้านนี้ ส่งของเร็วๆๆๆๆๆๆๆๆๆ ไว้มาอุดหนุนใหม่ๆๆๆๆๆๆๆๆๆ | The product is of very good quality; I tried it and it works well. I recommend this shop; very fast shipping. I'll come back to buy again. | 1 |
| 3 | แพคสินค้ามเรีนบร้อยดี จัดส่งรวดเร็ว ตรงปก ส่งครบ หอมดีค่ะ ขนาดสะดวกพกพา ฉีดได้เรื่อยๆ ราคาถูกมากๆ แนะนำค่ะ | The packaging is neat, delivered quickly, as described and complete. It smells nice and is convenient to carry. You can spray it anytime; very cheap. Recommended! | 1 |
| 4 | สินค้าดีเหมาะสมกับราคารอสินค้าไม่นานประมาณวัน บริการขนส่งดีโทรมาบอกก่อนเข้ามาส่งสินค้า ไว้โอกาสหน้าสั่งเพิ่มจ้า | The product is good for the price. I didn't wait long, about a day. Good delivery service; they called before delivering. I will order more next time! | 1 |
| 3388 | กลิ่นติดไม่ค่อยนาน | The scent doesn't last long. | 0 |
| 3390 | คุณภาพน้ำหอมไม่เหมือนตอนแรกที่สั่งมา | The quality of the perfume is not the same as when I first ordered. | 0 |

In row 3, *มเรีนบร้อย* is a misspelling of *เรียบร้อย* ("neat").

### Train/test split used in the article

The article splits each dataset 70:30 in the row order of the files, without stratification:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reviews, labels, test_size=0.3, random_state=42)
```

| Test set | Reviews | Positive | Negative | Majority-class rate |
|:--|--:|--:|--:|--:|
| Perfume | 1,397 | 701 | 696 | 50.18% |
| Indigo | 984 | 758 | 226 | 77.03% |

The indigo dataset is imbalanced, so report macro-F1, balanced accuracy or MCC alongside accuracy.

## File format and loading

**Excel files.** Each file has one sheet with one column, no header row and one review (or one label) per row. Row *i* of a review file corresponds to row *i* of its label file.

| Dataset | Reviews | Labels |
|:--|:--|:--|
| Perfume | `data/perfume_review_cleaned_dataset.xlsx` | `data/perfume_class_cleaned_dataset.xlsx` |
| Indigo | `data/indigo_review_cleaned_dataset.xlsx` | `data/indigo_class_cleaned_dataset.xlsx` |

```python
import pandas as pd
reviews = pd.read_excel("data/perfume_review_cleaned_dataset.xlsx", header=None)[0]
labels  = pd.read_excel("data/perfume_class_cleaned_dataset.xlsx", header=None)[0]
```

**CSV files.** `perfume_labeled.csv` and `indigo_labeled.csv` combine each pair of Excel files. They are UTF-8 with a byte-order mark, so Thai text displays correctly when opened in Excel. Columns:

| Column | Description |
|:--|:--|
| `id` | Row number starting at 0, identical to the row index of the Excel files |
| `review` | Cleaned review text |
| `label` | 1 = positive, 0 = negative |

```python
import pandas as pd
url = "https://raw.githubusercontent.com/mongkol-saensuk/thai-ecommerce-reviews-sghs/main/data/perfume_labeled.csv"
df = pd.read_csv(url)
```

`data/SHA256SUMS` lets you confirm that your copy is unchanged: run `sha256sum -c SHA256SUMS` inside `data/`.

## Notebooks

| Notebook | What it does | Run |
|:--|:--|:--|
| `SG_HS_MethodsX_experiments.ipynb` | The experiments of the article: word segmentation, three Word2Vec variants (CBOW, Skip-gram with negative sampling, SG-HS), six vector sizes (50–300) and three classifiers (CNN, BiLSTM, RNN) on both datasets, 108 runs in total. Reports accuracy, balanced accuracy, per-class and macro/weighted F1, MCC, Cohen's κ, ROC-AUC, PR-AUC, confusion matrices and training times, with Wilcoxon signed-rank tests, figures and an analysis of misclassified reviews. | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mongkol-saensuk/thai-ecommerce-reviews-sghs/blob/main/notebooks/SG_HS_MethodsX_experiments.ipynb) |

The explanations inside the notebook are in Thai; the code and its comments are in English.

### Running in Google Colab

1. Click the **Open in Colab** badge. The notebook opens with a T4 GPU runtime; check this under *Runtime → Change runtime type*.
2. Run the cells in order. In the settings cell, `DATA_SOURCE = "github"` clones this repository and reads the files in `data/`, so nothing needs to be uploaded. To use your own copy in Google Drive, set `DATA_SOURCE = "drive"` and put the four Excel files in `My Drive/dataset_perfume`.
3. Allow Google Drive access when asked. Results are saved to `My Drive/experiments/<EXPERIMENT_NAME>/`. If the session disconnects, run the notebook again with the same name; finished runs are skipped.
4. Run once with `QUICK_TEST = True` (a few minutes) to check that everything works, then set it to `False` for the full experiment.

### Running outside Colab

The notebook uses `google.colab.drive` for storage. To run it in local Jupyter, install the packages in `requirements.txt`, remove the two `drive` lines in the settings cell, and set `DATA_DIR` and `RESULTS_DIR` to local folders.

## Reproducibility

The notebook was tested on Google Colab with Python 3.13, TensorFlow 2.21.0, Keras 3.13.2, Gensim 4.4.0, PyThaiNLP 5.4.0, scikit-learn 1.6.1, NumPy 2.1.3 and pandas 2.2.3.

All random seeds are set (`SPLIT_SEED = 42`; classifier seeds in `SEEDS`). Results can still differ slightly between runs and from the tables of the article, because Word2Vec trains with several threads and some GPU operations are not deterministic. For near bit-exact repetition, set `W2V_WORKERS = 1` and `DETERMINISTIC_OPS = True` in the settings cell.

`scripts/build_labeled_csv.py` rebuilds the CSV files and checksums from the Excel files and prints the statistics in the table above (word counts require PyThaiNLP):

```bash
pip install pandas openpyxl pythainlp
python scripts/build_labeled_csv.py --data-dir data --words
```

## How to cite

GitHub shows a **Cite this repository** button generated from [`CITATION.cff`](CITATION.cff). In BibTeX:

```bibtex
@misc{chantawat2026thaireviews,
  author       = {Chantawat, Charuwat and Witchakool, Suwiwat and Choompol, Atchara and Saensuk, Mongkol},
  title        = {Thai e-commerce customer reviews of perfume and indigo-dyed products with opinion labels, and {SG-HS} workflow notebook},
  year         = {2026},
  howpublished = {GitHub repository},
  url          = {https://github.com/mongkol-saensuk/thai-ecommerce-reviews-sghs}
}
```

Please also cite the accompanying article once it is published.

## License and terms of use

- **Data** (`data/`): [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt the data for any purpose if you give appropriate credit by citing this repository.
- **Code** (`notebooks/`, `scripts/`): [MIT License](LICENSE).

The review texts were written by customers of Shopee and Lazada and contain no user names or other identifiers. The labels, cleaning and curation are the work of the authors. Users of the data are responsible for complying with the terms of the source platforms and with applicable law. Please do not attempt to re-identify the authors of the reviews.

## Authors and contact

- **Charuwat Chantawat**, Department of Educational Research and Development, Faculty of Education, Mahasarakham University, Mahasarakham 44000, Thailand
- **Suwiwat Witchakool**, Department of Mathematics and Statistics, Faculty of Science and Information Technology, Sakon Nakhon Rajabhat University, Sakon Nakhon 47000, Thailand
- **Atchara Choompol**, Department of Computer Engineering, Faculty of Engineering and Industrial Technology, Kalasin University, Kalasin 46000, Thailand
- **Mongkol Saensuk** (corresponding author), Department of Mathematics and Statistics, Faculty of Science and Information Technology, Sakon Nakhon Rajabhat University, Sakon Nakhon 47000, Thailand. Email: mongkol@snru.ac.th

Questions and problem reports are welcome through [GitHub Issues](https://github.com/mongkol-saensuk/thai-ecommerce-reviews-sghs/issues).

---

## สรุปภาษาไทย

Repository นี้มี**ชุดข้อมูลรีวิวสินค้าภาษาไทย 2 ชุด** จาก Shopee และ Lazada พร้อมคลาส และ **โค้ด Google Colab** ที่ใช้ในบทความ MethodsX เรื่องการจำแนกความคิดเห็นลูกค้าด้วย Skip-gram ร่วมกับ hierarchical softmax (SG-HS)

**ข้อมูล**

| ชุดข้อมูล | จำนวนรีวิว | เชิงบวก (1) | เชิงลบ (0) | ไฟล์รีวิว | ไฟล์คลาส |
|:--|--:|--:|--:|:--|:--|
| น้ำหอม | 4,656 | 2,281 | 2,375 | `perfume_review_cleaned_dataset.xlsx` | `perfume_class_cleaned_dataset.xlsx` |
| ผลิตภัณฑ์ผ้าคราม | 3,279 | 2,491 | 788 | `indigo_review_cleaned_dataset.xlsx` | `indigo_class_cleaned_dataset.xlsx` |

- ไฟล์ Excel แต่ละไฟล์มี 1 คอลัมน์ ไม่มีหัวตาราง แถวที่ *i* ของไฟล์รีวิวตรงกับแถวที่ *i* ของไฟล์คลาส
- ไฟล์ `*_labeled.csv` รวมรีวิวและคลาสไว้ในไฟล์เดียว (คอลัมน์ `id`, `review`, `label`) เปิดใน Excel แล้วอ่านภาษาไทยได้ทันที
- คลาสกำหนดโดยผู้เชี่ยวชาญ 3 คน ตัดสินด้วยเสียงข้างมาก
- รีวิวผ่านการทำความสะอาดแล้ว (ลบตัวเลข อักษรละติน และสัญลักษณ์) คำที่สะกดผิดยังคงไว้ตามที่ลูกค้าเขียน และไม่มีชื่อผู้ใช้หรือข้อมูลระบุตัวตน ถ้าจะทำความสะอาดรีวิวใหม่แบบเดียวกัน ใช้ `scripts/clean_review.py`
- ไฟล์เผยแพร่ตามที่ใช้ในบทความทุกแถว (เพื่อให้ผลและการแบ่ง train/test ตรงกัน) จึงมีรีวิวว่าง (น้ำหอม 2 แถว, ผ้าคราม 8 แถว) และรีวิวซ้ำ (น้ำหอม 33 แถว, ผ้าคราม 199 แถว) ดูรายละเอียดใน [Data quality notes](#data-quality-notes)

**วิธีรันโค้ด**

1. กดปุ่ม **Open in Colab** ในตาราง [Notebooks](#notebooks)
2. รันทีละเซลล์ตามลำดับ โค้ดจะดึงข้อมูลจาก GitHub นี้ให้อัตโนมัติ (`DATA_SOURCE = "github"`) ถ้าต้องการใช้ไฟล์ใน Google Drive ของตัวเอง ให้เปลี่ยนเป็น `"drive"`
3. อนุญาตให้เข้าถึง Google Drive เพื่อบันทึกผลไว้ที่ `My Drive/experiments/`
4. แนะนำให้รัน `QUICK_TEST = True` ก่อนหนึ่งรอบ

**สัญญาอนุญาต** ข้อมูลใช้ CC BY 4.0 (นำไปใช้ได้โดยอ้างอิงที่มา) และโค้ดใช้ MIT

**ติดต่อ** มงคล แสนสุข (mongkol@snru.ac.th) ภาควิชาคณิตศาสตร์และสถิติ คณะวิทยาศาสตร์และเทคโนโลยีสารสนเทศ มหาวิทยาลัยราชภัฏสกลนคร
