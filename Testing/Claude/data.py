"""Loading and cleaning utilities for the 'Help the doctor' dataset.

Why not just ``pd.read_csv(path, sep=';')``?
-------------------------------------------
Both CSV files contain a handful of *corrupted blocks*. In each block a field
was cut at ~32,758 characters (Excel's 32,767-char cell limit) and an
unclosed double quote (") was left behind. A quote-aware parser such as pandas
then treats everything up to the next stray quote as ONE field, so several
complete records are silently swallowed into the ``description`` of another
record, and leftover text lands in the ``medical_specialty`` column.

In ``train.csv`` every real record sits on exactly one physical line (CRLF), so
we parse line by line and repair the three kinds of broken lines:

* fragment line      ``Label;"<truncated description>``         -> dropped
* cut record         ``Label;desc;name;<transcription cut>";;;`` -> kept
* continuation line  ``<rest of that transcription>;;;;``      -> merged
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import pandas as pd

TEXT_COLS = ["description", "sample_name", "transcription", "keywords"]
TRAIN_COLS = ["medical_specialty"] + TEXT_COLS

LABELS = [
    "Surgery", "Cardiovascular-Pulmonary", "Orthopedic", "Radiology",
    "General Medicine", "Gastroenterology", "Neurology", "Obstetrics-Gynecology",
    "Neurosurgery", "Ophthalmology", "Psychiatry-Psychology", "Dermatology",
]

# How each label is written as the first keyword (see the analysis notebook).
LABEL_TO_KEYWORD = {lab: lab.lower().replace("-", " / ") for lab in LABELS}


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_physical_lines(path: str | Path) -> list[str]:
    with open(path, encoding="utf-8", newline="") as f:
        return re.split(r"\r?\n", f.read())  # CRLF in the original, LF after some git checkouts


def _split_fields(line: str, n: int) -> list[str]:
    """Split one physical line into exactly ``n`` fields."""
    if line.count('"') % 2 == 0:  # balanced quotes -> normal CSV rules
        fields = next(csv.reader([line], delimiter=";", quotechar='"'))
    else:  # broken quoting -> plain split, strip stray quotes
        fields = [x.strip('"') for x in line.split(";")]
    fields = [x.replace('""', '"').strip() for x in fields]
    return (fields + [""] * n)[:n]


def load_train(path: str | Path = "train.csv", verbose: bool = True):
    """Robust loader for train.csv.

    Returns
    -------
    df  : DataFrame with columns TRAIN_COLS (+ ``line`` = 1-based line in file)
    log : DataFrame describing every line that had to be repaired
    """
    lines = _read_physical_lines(path)
    header, body = lines[0], lines[1:]
    assert header.split(";") == TRAIN_COLS, f"unexpected header: {header}"

    records, log = [], []
    for lineno, line in enumerate(body, start=2):
        first = line.split(";", 1)[0]
        if line.count(";") == 1 and '"' in line:
            log.append((lineno, "fragment (dropped)", line[:80]))
            continue
        if first not in LABELS:  # text continuing the previous record
            prev = records[-1]
            prev[3] = (prev[3].rstrip('";') + _split_fields(line, 5)[0].strip('"')).strip()
            log.append((lineno, "continuation (merged into previous)", line[:80]))
            continue
        records.append(_split_fields(line, 5) + [lineno])

    df = pd.DataFrame(records, columns=TRAIN_COLS + ["line"])
    log = pd.DataFrame(log, columns=["line", "action", "start_of_line"])
    if verbose:
        print(f"train: {len(body)} physical lines -> {len(df)} records "
              f"({len(log)} lines repaired)")
    return df, log


def load_test(path: str | Path = "test_no_labels.csv", verbose: bool = True):
    """Loader for test_no_labels.csv (no header, no label column).

    We deliberately use the *quote-aware* pandas parser here: it reproduces how
    the organisers built this file (the records swallowed inside the four
    corrupted blocks still carry their labels, i.e. the organisers' parser saw
    each block as a single record). One prediction per returned row -> 409
    lines in results.txt. Confirm with the course staff if in doubt.
    """
    df = pd.read_csv(path, sep=";", header=None, names=TEXT_COLS,
                     dtype=str, keep_default_na=False)
    df = df.apply(lambda s: s.str.strip())
    if verbose:
        n_lines = len(_read_physical_lines(path))
        print(f"test: {n_lines} physical lines -> {len(df)} records")
    return df


# --------------------------------------------------------------------------- #
# Cleaning (regex)
# --------------------------------------------------------------------------- #
# Disclaimer appended by the source website, glued to the last keyword
# (e.g. "...scopeNOTE,: Thesetranscribed medical transcription sample reports...")
BOILERPLATE_RE = re.compile(r"NOTE\s*,?\s*:\s*These\s*transcribed medical.*$",
                            flags=re.IGNORECASE | re.DOTALL)
# Line breaks of the original reports were flattened to ",": "...pain.,She is..."
NEWLINE_RE = re.compile(r"(?<=[.:])\s*,(?=\s*\S)")
# Section headers such as "PREOPERATIVE DIAGNOSIS:" or "HEENT:"
SECTION_RE = re.compile(r"(?:(?<=^)|(?<=[.,:]))\s*([A-Z][A-Z0-9 /&()\-]{2,50}):")
# Numbered duplicates in sample_name, e.g. "Vitrectomy - 2"
SAMPLE_NUM_RE = re.compile(r"\s*-\s*\d+\s*$")
MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    text = BOILERPLATE_RE.sub("", text)
    text = NEWLINE_RE.sub("\n", text)
    text = MULTISPACE_RE.sub(lambda m: "\n" if "\n" in m.group() else " ", text)
    return text.strip(" ,;\n")


def clean_keywords(kw: str) -> str:
    kw = BOILERPLATE_RE.sub("", kw)
    return ", ".join(k.strip() for k in kw.split(",") if k.strip())


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["description", "transcription"]:
        out[col] = out[col].map(clean_text)
    out["sample_name"] = out["sample_name"].str.replace(SAMPLE_NUM_RE, "", regex=True).str.strip()
    out["keywords"] = out["keywords"].map(clean_keywords)
    return out


def first_keyword(kw: str) -> str:
    return kw.split(",")[0].strip().lower() if kw else ""


def split_sections(transcription: str) -> list[tuple[str, str]]:
    """Split a transcription into (HEADER, body) pairs using SECTION_RE."""
    matches = list(SECTION_RE.finditer(transcription))
    if not matches:
        return [("", transcription)]
    sections = []
    if matches[0].start() > 0:
        sections.append(("", transcription[: matches[0].start()]))
    for m, nxt in zip(matches, matches[1:] + [None]):
        body = transcription[m.end(): nxt.start() if nxt else len(transcription)]
        sections.append((m.group(1).strip(), body.strip(" ,\n")))
    return sections


def group_ids(df: pd.DataFrame) -> pd.Series:
    """Group id for leakage-free splits: rows with the same transcription (or,
    if empty, the same description) belong to the same group."""
    key = df["transcription"].where(df["transcription"] != "", df["description"])
    key = key.str.lower().str.replace(r"\W+", "", regex=True)
    return key.factorize()[0]
