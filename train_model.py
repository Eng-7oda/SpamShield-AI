"""Train the SMS Spam Detection model and save reusable artifacts."""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from model_utils import clean_and_lemmatize

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "spam.csv"
ARTIFACT_PATH = BASE_DIR / "spam_model_bundle.joblib"
METRICS_PATH = BASE_DIR / "model_metrics.json"


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="latin-1")
    # Original SMS Spam Collection files often contain three empty extra columns.
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    if {"v1", "v2"}.issubset(df.columns):
        df = df.rename(columns={"v1": "Target", "v2": "Text"})
    elif not {"Target", "Text"}.issubset(df.columns):
        raise ValueError("Dataset must have columns v1/v2 or Target/Text.")
    df = df[["Target", "Text"]].dropna()
    df["Target"] = df["Target"].astype(str).str.strip().str.lower()
    df["Text"] = df["Text"].astype(str)
    df = df[df["Target"].isin(["ham", "spam"])]
    if df.empty:
        raise ValueError("No ham/spam rows found in spam.csv.")
    return df


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find {DATA_PATH.name}. Put your SMS dataset named spam.csv in {BASE_DIR}."
        )

    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df):,} SMS messages")
    print("Preprocessing text (first run may download NLTK resources)...")
    df["ProcessedText"] = df["Text"].apply(clean_and_lemmatize)

    # Explicit mapping keeps the positive class predictable: ham=0, spam=1.
    label_map = {"ham": 0, "spam": 1}
    y = df["Target"].map(label_map).astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        df["ProcessedText"], y, test_size=0.30, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer()
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = SVC()  # Same classifier family as the user's current notebook.
    model.fit(x_train_tfidf, y_train)
    y_pred = model.predict(x_test_tfidf)

    report = classification_report(
        y_test, y_pred, labels=[0, 1], target_names=["Ham", "Spam"], output_dict=True, zero_division=0
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "spam_precision": float(precision_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "spam_recall": float(recall_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "spam_f1": float(f1_score(y_test, y_pred, pos_label=1, zero_division=0)),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "dataset_rows": int(len(df)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "label_mapping": label_map,
        "model_name": "Support Vector Classifier (SVC)",
        "vectorizer_name": "TF-IDF",
    }

    bundle = {
        "model": model,
        "vectorizer": vectorizer,
        "label_map": label_map,
        "metrics": metrics,
        "preprocessing": "non-letters removed; lowercase; English stopwords removed; verbs lemmatized",
    }
    joblib.dump(bundle, ARTIFACT_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\nTraining complete")
    print(f"Accuracy:        {metrics['accuracy']:.4f}")
    print(f"Spam precision:  {metrics['spam_precision']:.4f}")
    print(f"Spam recall:     {metrics['spam_recall']:.4f}")
    print(f"Spam F1-score:   {metrics['spam_f1']:.4f}")
    print(f"Saved model bundle: {ARTIFACT_PATH}")
    print(f"Saved metrics:      {METRICS_PATH}")


if __name__ == "__main__":
    main()
