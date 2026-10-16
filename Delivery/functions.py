import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import classification_report
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