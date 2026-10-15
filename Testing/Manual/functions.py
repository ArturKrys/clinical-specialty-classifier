import pandas as pd
import re
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import classification_report
from nltk.stem import PorterStemmer

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


# Preprocessing Functions
stemmer = PorterStemmer()

def preprocessing(text):
    #if not isinstance(text, str):
    #    return []
    text = text.lower()
    words = re.findall(r'\b[a-z0-9]+\b', text)
    return [stemmer.stem(word) for word in words]


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




# Create Models For TF-IDF
def evaluate_input(X, name, classifier, classifier_name, train_indices, valid_indices, y_train, y_valid):
    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                #tokenizer=preprocessing,
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

    return {
        "model": classifier_name,
        "input": name,
        "accuracy": accuracy_score(y_valid, predictions),
        "macro_f1": f1_score(
            y_valid,
            predictions,
            average="macro",
        ),
    }


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
                tokenizer=preprocessing,
                #lowercase=True,
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

def show_classification_report(y_true, predictions, labels):
    print(
        classification_report(
            y_true,
            predictions,
            labels=labels,
            zero_division=0,
        )
    )