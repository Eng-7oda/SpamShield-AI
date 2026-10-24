"""Shared preprocessing utilities for training and inference."""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_resources() -> None:
    """Download the small NLTK resources needed by this project if missing."""
    resources = [
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def clean_and_lemmatize(text: str) -> str:
    """Match the preprocessing used in the user's notebook.

    1) Remove non-English letters, 2) lowercase, 3) split into tokens,
    4) remove English stopwords, 5) lemmatize verbs.
    """
    ensure_nltk_resources()
    text = re.sub(r"[^a-zA-Z]", " ", str(text)).lower()
    tokens = text.split()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word, pos="v") for word in tokens]
    return " ".join(tokens)
