import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import re
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics.pairwise import paired_cosine_distances
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer



# Cleaning Data
def classify_line(line, LABELS):
    first_field = line.split(";", 1)[0].strip()

    if line.count(";") == 1 and line.count('"') % 2 == 1:
        return "fragment"

    if first_field in LABELS:
        return "new_record"

    if (first_field not in LABELS) and line.endswith(";;;;"):
        return "continuation"

    return "unknown"

def parse_broken_lines(line, type="current"):
    # "prev": first half of a cut record -> remove its trailing ';' and the quote before them
    # "current": a closing line (or a glued record) -> remove the extra ';' and the closing quote
    if type == "prev":
        while line.endswith(";"):
            line = line[:-1]
        if line.endswith('"'):
            line = line[:-1]
    elif type == "current":
        while line.endswith(";") and line.count(";") > 4:
            line = line[:-1]
        if line.endswith('"'):
            line = line[:-1]
    return line

def clean_field(field):
    """Undo CSV quoting by hand: ' "a ""b"" c" ' -> 'a "b" c'."""
    field = field.strip()
    if len(field) >= 2 and field[0] == field[-1] == '"':
        field = field[1:-1]
    field = re.sub(r'"{2,}', '"', field)  # "" (or """" inside the broken blocks) -> "
    if field.count('"') % 2:              # lone quote left over from a broken block
        field = field.strip('"')
    return field.strip()


#------------------------------------------------------------------------------------------
# Building Input For Models
def build_input_variants(df, columns):

    if not columns:
        raise ValueError(
            "At least one input column must be provided."
        )
    
    for col in columns: 
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' is not in the DataFrame."
            )
    
    result = pd.Series("", index=df.index, dtype="string")
    
    for col in columns:
        values = df[col].fillna("").astype(str).str.strip()
        result = result + " " + col + ": " + values

    return result.str.replace(
        r"\s+",
        " ",
        regex=True,
    ).str.strip()



#------------------------------------------------------------------------------------------
# Preprocessing Functions
stemmer = PorterStemmer()

def preprocessing(text):
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]+\b', text)
    return words
    #return [stemmer.stem(word) for word in words]


def extract_headers(text):
    HEADER_PATTERN = re.compile(
        r"\b[A-Z][A-Z0-9 /&()\-]{2,50}:"
    )

    if not isinstance(text, str):
        return []

    return [
        header.rstrip(":").strip()
        for header in HEADER_PATTERN.findall(text)
    ]


def headers_for_model(headers):
    return " ".join(
        "HEADER_" + header.replace(" ", "_")
        for header in headers
    )

def headers_words(headers):
    return " ".join(headers)
 
# Processing Keywords
def first_keyword(value):
    value = str(value).strip()
    if not value:
        return ""
    return value.split(",")[0].strip()

def word_normalized(value):
    value = str(value).strip().lower()
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return " ".join(value.split())



#------------------------------------------------------------------------------------------
# Data Augmentation: clinical EDA (our adaptation of EDA, Wei and Zou 2019)

# A new sentence starts after ".", "!" or "?", then "," or spaces, then a capital letter or a digit
SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])(?=[,\s]+[A-Z0-9])")

def sentence_dropout(text, rng, p=0.2):
    # EDA deletes single words; we delete whole sentences, so the kept ones keep their meaning ("no fracture")
    sentences = SENTENCE_BOUNDARY.split(text)
    if len(sentences) < 3:
        return text
    kept = [s for s in sentences if rng.random() > p]
    return "".join(kept or sentences).lstrip(", ")


# EDA uses WordNet synonyms; the "synonyms" of clinical notes are their abbreviations
ABBREVIATIONS = {
    "MRI": "magnetic resonance imaging", "CT": "computed tomography", "EKG": "electrocardiogram",
    "ECG": "electrocardiogram", "EEG": "electroencephalogram", "EMG": "electromyography",
    "IV": "intravenous", "CBC": "complete blood count", "PACU": "post-anesthesia care unit",
    "ICU": "intensive care unit", "GI": "gastrointestinal", "MI": "myocardial infarction",
    "BP": "blood pressure", "HR": "heart rate", "CVA": "cerebrovascular accident",
    "DVT": "deep vein thrombosis", "COPD": "chronic obstructive pulmonary disease",
    "HTN": "hypertension", "EGD": "esophagogastroduodenoscopy", "CAD": "coronary artery disease",
    "TIA": "transient ischemic attack", "LMA": "laryngeal mask airway",
    "ACL": "anterior cruciate ligament", "EBL": "estimated blood loss",
    "GERD": "gastroesophageal reflux disease", "UTI": "urinary tract infection",
    "CABG": "coronary artery bypass graft", "CHF": "congestive heart failure",
    "DM": "diabetes mellitus", "SOB": "shortness of breath",
}
EXPANSIONS = {}
for short_form, long_form in ABBREVIATIONS.items():
    EXPANSIONS.setdefault(long_form, short_form)                                  # electrocardiogram -> EKG
ABBREVIATION = re.compile(r"\b(?:" + "|".join(ABBREVIATIONS) + r")\b"              # MRI (exact case)
                          r"|(?i:\b(?:" + "|".join(EXPANSIONS) + r")\b)")          # magnetic resonance imaging

def replace_abbreviations(text, rng, p=0.5):
    def replace(match):
        word = match.group()
        if rng.random() > p:
            return word
        return ABBREVIATIONS.get(word) or EXPANSIONS[word.lower()]
    return ABBREVIATION.sub(replace, text)


def clinical_eda_copy(row, rng, p_sentence=0.2, p_abbreviation=0.5):
    # New version of a record: same label, keywords and sample name; changed description and transcription
    row = row.copy()
    row["transcription"] = sentence_dropout(row["transcription"], rng, p_sentence)
    row["description"] = replace_abbreviations(row["description"], rng, p_abbreviation)
    row["transcription"] = replace_abbreviations(row["transcription"], rng, p_abbreviation)
    headers = extract_headers(row["transcription"])                              # headers follow the new text
    row["transcription_headers_model"] = headers_for_model(headers)
    row["transcription_headers_words"] = headers_words(headers)
    return row

def exact_copy(row, rng):
    return row


def oversample(df, train_indices, make_copy, label="medical_specialty", seed=42):
    # Adds new rows (make_copy of a random record of the same class) until every class of the training part
    # is as large as the largest one. Returns df with the new rows at the end and the new training indices.
    rng = random.Random(seed)
    train = df.loc[train_indices]
    target = train[label].value_counts().max()
    new_rows = []
    for _, group in train.groupby(label):
        for _ in range(target - len(group)):
            new_rows.append(make_copy(group.iloc[rng.randrange(len(group))], rng))
    new_rows = pd.DataFrame(new_rows)
    new_rows.index = pd.RangeIndex(df.index.max() + 1, df.index.max() + 1 + len(new_rows))
    return pd.concat([df, new_rows]), pd.Index(train_indices).append(new_rows.index)


def eda_similarity(df, columns, seed=42):
    # Mean cosine similarity of the TF-IDF vectors of each record and of one clinical EDA version of it (1 = identical)
    versions = df.apply(clinical_eda_copy, axis=1, rng=random.Random(seed))
    original_text = build_input_variants(df, columns)
    versions_text = build_input_variants(versions, columns)
    tfidf = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=50_000)
    tfidf.fit(original_text)
    return (1 - paired_cosine_distances(tfidf.transform(original_text), tfidf.transform(versions_text))).mean()



#------------------------------------------------------------------------------------------
# Create Models For TF-IDF
def evaluate_input(X, name, classifier, classifier_name, train_indices, valid_indices, y_train, y_valid):
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=50_000,
            ),
        ),
        (
            "classifier",
            clone(classifier),
        ),
    ])

    model.fit(
        X.loc[train_indices],
        y_train,
    )

    predictions = model.predict(X.loc[valid_indices])

    return [predictions, 
            {
                "model": classifier_name,
                "input": name,
                "accuracy": accuracy_score(y_valid, predictions),
                "macro_f1": f1_score(
                    y_valid,
                    predictions,
                    average="macro",),
            },
            model]


def evaluate_input_extra(
    X,
    name,
    classifier,
    classifier_name,
    train_indices,
    valid_indices,
    y_train,
    y_valid,
):
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                #analyzer='char_wb',
                #ngram_range=(3,6),
                ngram_range=(1, 2),
                min_df=2,
                sublinear_tf=True,
                max_features=50_000,
            ),
        ),
        (
            "classifier",
            clone(classifier),
        ),
    ])

    model.fit(
        X.loc[train_indices],
        y_train,
    )

    predictions = model.predict(
        X.loc[valid_indices]
    )

    labels = sorted(y_train.unique())

    f1_per_class = f1_score(
        y_valid,
        predictions,
        labels=labels,
        average=None,
        zero_division=0,
    )

    lowest_f1_index = f1_per_class.argmin()

    return [predictions, {
        "model": classifier_name,
        "input": name,
        "accuracy": accuracy_score(
            y_valid,
            predictions,
        ),
        "macro_f1": f1_score(
            y_valid,
            predictions,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "lowest_f1": f1_per_class[lowest_f1_index],
        "lowest_f1_label": labels[lowest_f1_index],
    }]

def fit_pipeline(X, y, classifier):
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, max_features=50_000)),
        ("classifier", clone(classifier)),
    ]).fit(X, y)

def show_classification_report(y_true, predictions, labels): # y_true is y_valid
    print(
        classification_report(
            y_true,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )


#------------------------------------------------------------------------------------------
# Plots For The Paper
SHORT_LABELS = {"Cardiovascular-Pulmonary": "Cardio-Pulm", "General Medicine": "Gen. Medicine",
                "Gastroenterology": "Gastro", "Obstetrics-Gynecology": "Obs-Gyn",
                "Psychiatry-Psychology": "Psych"}

def plot_confusion_matrix(y_true, predictions, labels):
    # Rows: true class; columns: predicted class; number: records; colour: share of the true class
    counts = confusion_matrix(y_true, predictions, labels=labels)
    shares = counts / counts.sum(axis=1, keepdims=True)
    names = [SHORT_LABELS.get(label, label) for label in labels]

    with plt.rc_context({"font.size": 7, "axes.spines.top": False, "axes.spines.right": False}):
        fig, ax = plt.subplots(figsize=(3.3, 3.1))
        ax.imshow(shares, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)), names, rotation=45, ha="right")
        ax.set_yticks(range(len(labels)), names)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.tick_params(length=0)
        for (row, col), count in np.ndenumerate(counts):
            if count:
                ax.text(col, row, count, ha="center", va="center", fontsize=6,
                        color="white" if shares[row, col] > 0.6 else "#0b0b0b")
    return fig
