# SpamShield AI — SMS Spam Detection

A Streamlit dashboard for an SMS classifier using TF-IDF + Support Vector Classifier (SVC). It includes a dashboard, live message analyzer, hold-out model metrics, confusion matrix, and session prediction history with CSV export.

## Project files

- `app.py` — Streamlit UI.
- `train_model.py` — loads `spam.csv`, trains the model, evaluates it, and saves reusable artifacts.
- `model_utils.py` — shared preprocessing used by training and inference.
- `requirements.txt` — Python dependencies.
- `spam.csv` — **you provide this file** (the SMS Spam Collection dataset).

## 1. Set up the environment (Windows / Anaconda Prompt)

```bash
cd path\to\SpamShield_AI
conda activate ml
python -m pip install -r requirements.txt
```

If you do not have the `ml` conda environment, create one first:

```bash
conda create -n ml python=3.12 -y
conda activate ml
python -m pip install -r requirements.txt
```

## 2. Add the dataset

Copy your existing `spam.csv` into this folder, next to `train_model.py`. The code supports the original dataset columns `v1` and `v2` and the renamed columns `Target` and `Text`.

## 3. Train and save the model

```bash
python train_model.py
```

The script creates `spam_model_bundle.joblib` and `model_metrics.json`. It uses a stratified 70/30 train/test split, fits TF-IDF **only on the training data**, and evaluates the SVM on the held-out test data. The first run may download NLTK stopwords and WordNet data.

## 4. Launch the UI

```bash
streamlit run app.py
```

Streamlit will print a local URL, usually `http://localhost:8501`. Open it in your browser.

## Important implementation notes

- Train the model before opening the app. The UI does not silently invent model metrics or predictions.
- The same text preprocessing function is called during training and inference.
- The fitted TF-IDF vectorizer is saved and reused with `.transform()` for incoming messages; it is **not refitted** on individual messages.
- The SVC is not configured for probability estimates, so the app displays the predicted class rather than an unsupported confidence percentage.
- History is session-only and is not a persistent database. Use the CSV export if needed, but handle message content carefully.
- The preprocessing intentionally follows the notebook's English-only cleaning, so digits, punctuation, URLs, and non-English characters are removed. This can remove useful spam signals and is worth testing as a future improvement.
- Only load `spam_model_bundle.joblib` if you trust its source; joblib/pickle files can execute code when loaded.
