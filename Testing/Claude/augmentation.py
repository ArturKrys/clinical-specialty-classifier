"""Data-augmentation strategies for the 'Help the doctor' dataset.

All functions are pure (text in -> text out) and take a ``random.Random`` so
results are reproducible. IMPORTANT: augment ONLY the training part of a split,
after splitting, otherwise copies of a validation document leak into training.

Strategies
----------
1. EDA (Wei & Zou, 2019): random swap / deletion / insertion and (optionally)
   WordNet synonym replacement.
2. Medical abbreviation swap: replace "MRI" <-> "magnetic resonance imaging",
   etc. A domain-aware alternative to WordNet, which knows little medicine.
3. Section crop: keep a random subset of the report's sections (sections are
   delimited by headers such as "PREOPERATIVE DIAGNOSIS:"). Produces a shorter,
   still coherent report.
4. Field dropout: blank ``keywords``/``description``/``sample_name`` at random.
   ~22% of the test rows have no keywords, so the model must not rely on them.
5. Back-translation (optional, needs ``transformers`` + internet once).
"""

from __future__ import annotations

import random
import re

import pandas as pd

from data import TEXT_COLS, split_sections

# --------------------------------------------------------------------------- #
# 1. EDA
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"\S+")

try:  # WordNet is optional: run nltk.download("wordnet") once to enable it
    from nltk.corpus import wordnet as _wn
    _wn.synsets("pain")
except Exception:  # noqa: BLE001
    _wn = None

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "with", "was",
    "is", "were", "be", "been", "he", "she", "it", "his", "her", "this", "that",
    "at", "by", "as", "from", "are", "had", "has", "have", "patient",
}
# Negations / laterality carry meaning in clinical text: never touch them.
PROTECTED = {"no", "not", "without", "denies", "negative", "left", "right",
             "bilateral", "none"}


def _synonyms(word: str) -> list[str]:
    if _wn is None:
        return []
    syns = {l.name().replace("_", " ") for s in _wn.synsets(word) for l in s.lemmas()}
    syns.discard(word)
    return sorted(syns)


def eda(text: str, rng: random.Random, alpha: float = 0.1,
        ops: tuple[str, ...] = ("swap", "delete", "insert", "synonym")) -> str:
    """Apply ONE randomly chosen EDA operation to ~alpha of the tokens."""
    words = _TOKEN_RE.findall(text)
    if len(words) < 4:
        return text
    n = max(1, int(alpha * len(words)))
    candidates = [i for i, w in enumerate(words)
                  if w.lower().strip(".,:;") not in STOPWORDS | PROTECTED]
    op = rng.choice([o for o in ops if o != "synonym" or _wn is not None])

    if op == "swap":
        for _ in range(n):
            i, j = rng.randrange(len(words)), rng.randrange(len(words))
            words[i], words[j] = words[j], words[i]
    elif op == "delete":
        drop = set(rng.sample(candidates, min(n, len(candidates))))
        words = [w for i, w in enumerate(words) if i not in drop]
    elif op == "insert":  # insert copies of existing content words
        for _ in range(n):
            words.insert(rng.randrange(len(words)), words[rng.choice(candidates)])
    elif op == "synonym":
        rng.shuffle(candidates)
        replaced = 0
        for i in candidates:
            syns = _synonyms(words[i].lower().strip(".,:;"))
            if syns:
                words[i] = rng.choice(syns)
                replaced += 1
            if replaced >= n:
                break
    return " ".join(words)


# --------------------------------------------------------------------------- #
# 2. Medical abbreviation swap
# --------------------------------------------------------------------------- #
ABBREVIATIONS = {
    "MRI": "magnetic resonance imaging", "CT": "computed tomography",
    "EKG": "electrocardiogram", "ECG": "electrocardiogram",
    "EEG": "electroencephalogram", "EMG": "electromyography",
    "CABG": "coronary artery bypass graft", "CAD": "coronary artery disease",
    "CHF": "congestive heart failure", "COPD": "chronic obstructive pulmonary disease",
    "MI": "myocardial infarction", "HTN": "hypertension", "DM": "diabetes mellitus",
    "GERD": "gastroesophageal reflux disease", "EGD": "esophagogastroduodenoscopy",
    "GI": "gastrointestinal", "UTI": "urinary tract infection",
    "TIA": "transient ischemic attack", "CVA": "cerebrovascular accident",
    "ACL": "anterior cruciate ligament", "ORIF": "open reduction internal fixation",
    "D&C": "dilation and curettage", "BP": "blood pressure", "HR": "heart rate",
    "SOB": "shortness of breath", "ICU": "intensive care unit",
    "PACU": "post-anesthesia care unit", "EBL": "estimated blood loss",
    "H&P": "history and physical", "ADHD": "attention deficit hyperactivity disorder",
    "IOP": "intraocular pressure", "BCC": "basal cell carcinoma",
}
_EXPANSIONS = {v: k for k, v in ABBREVIATIONS.items()}
_ABBR_RE = re.compile(r"\b(" + "|".join(map(re.escape, ABBREVIATIONS)) + r")\b")
_EXP_RE = re.compile(r"\b(" + "|".join(map(re.escape, _EXPANSIONS)) + r")\b", re.I)


def abbreviation_swap(text: str, rng: random.Random, p: float = 0.5) -> str:
    """Expand abbreviations / contract expansions, each with probability p."""
    text = _ABBR_RE.sub(lambda m: ABBREVIATIONS[m.group()] if rng.random() < p else m.group(), text)
    return _EXP_RE.sub(lambda m: _EXPANSIONS.get(m.group().lower(), m.group())
                       if rng.random() < p else m.group(), text)


# --------------------------------------------------------------------------- #
# 3. Section crop
# --------------------------------------------------------------------------- #
def section_crop(text: str, rng: random.Random, keep: float = 0.6) -> str:
    """Keep a random ~``keep`` fraction of the sections (order preserved)."""
    sections = split_sections(text)
    if len(sections) < 3:
        return text
    k = max(2, round(keep * len(sections)))
    idx = sorted(rng.sample(range(len(sections)), k))
    return "\n".join(f"{h}: {b}" if h else b for h, b in (sections[i] for i in idx))


# --------------------------------------------------------------------------- #
# 4. Field dropout (row-level)
# --------------------------------------------------------------------------- #
def field_dropout(row: pd.Series, rng: random.Random,
                  probs: dict[str, float] | None = None) -> pd.Series:
    probs = probs or {"keywords": 0.5, "description": 0.2, "sample_name": 0.2}
    row = row.copy()
    for col, p in probs.items():
        if rng.random() < p:
            row[col] = ""
    return row


# --------------------------------------------------------------------------- #
# 5. Back-translation (optional, slow)
# --------------------------------------------------------------------------- #
def back_translate(texts: list[str], pivot: str = "fr", max_chars: int = 2000) -> list[str]:
    """en -> pivot -> en with MarianMT (Helsinki-NLP). Downloads ~600 MB once."""
    from transformers import MarianMTModel, MarianTokenizer

    def _translate(batch, src, tgt):
        name = f"Helsinki-NLP/opus-mt-{src}-{tgt}"
        tok, model = MarianTokenizer.from_pretrained(name), MarianMTModel.from_pretrained(name)
        enc = tok([t[:max_chars] for t in batch], return_tensors="pt",
                  padding=True, truncation=True, max_length=512)
        return tok.batch_decode(model.generate(**enc), skip_special_tokens=True)

    return _translate(_translate(texts, "en", pivot), pivot, "en")


# --------------------------------------------------------------------------- #
# Putting it together
# --------------------------------------------------------------------------- #
def augment_row(row: pd.Series, rng: random.Random,
                methods: tuple[str, ...] = ("section_crop", "abbrev", "eda", "field_dropout")) -> pd.Series:
    row = row.copy()
    if "section_crop" in methods:
        row["transcription"] = section_crop(row["transcription"], rng)
    if "abbrev" in methods:
        for col in ("description", "transcription"):
            row[col] = abbreviation_swap(row[col], rng)
    if "eda" in methods:
        row["transcription"] = eda(row["transcription"], rng)
    if "field_dropout" in methods:
        row = field_dropout(row, rng)
    return row


def oversample_minority(df: pd.DataFrame, label_col: str = "medical_specialty",
                        target: int | None = None, seed: int = 42,
                        methods: tuple[str, ...] = ("section_crop", "abbrev", "eda", "field_dropout"),
                        ) -> pd.DataFrame:
    """Add augmented copies of rows from classes smaller than ``target``
    (default: the median class size) until every class reaches ``target``.
    Augmented rows get ``augmented=True``."""
    rng = random.Random(seed)
    counts = df[label_col].value_counts()
    target = target or int(counts.median())
    new_rows = []
    for label, n in counts.items():
        pool = df[df[label_col] == label]
        for _ in range(max(0, target - n)):
            new_rows.append(augment_row(pool.iloc[rng.randrange(len(pool))], rng, methods))
    out = pd.concat([df.assign(augmented=False),
                     pd.DataFrame(new_rows).assign(augmented=True)], ignore_index=True)
    return out
