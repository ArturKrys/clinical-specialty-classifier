# NLP Project 2026 - Help the Doctor (Group 51)

Classify clinical texts into one of 12 medical specialties.

## Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt

```

## Layout

```
data/
  original_train.csv    train.csv from the professor (has broken lines, do not read with pd.read_csv)
  original_test.csv     test_no_labels.csv from the professor (398 records, no labels)
  clean_train.csv       repaired training set (2,662 records), written by 01_data_analysis.ipynb
paper/                  51.tex, biblio.bib
Testing/Manual/         notebooks (paths are relative to this folder: ../../data/...)
```


## Before submitting

- [ ] `results.txt`: one label per line, same order as `data/original_test.csv` (= `test_no_labels.csv`), no header
- [ ] `51.pdf` ≤ 3 pages, evaluation-tips section removed from the .tex
- [ ] Zip named `51.zip` (not rar): `51.pdf`, code, `results.txt`, any extra datasets
