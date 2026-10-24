# 🛡️ SpamShield AI — SMS Spam Detection

An interactive Machine Learning application that classifies SMS messages as **Spam** or **Ham (Legitimate)** using TF-IDF feature extraction and a Support Vector Classifier (SVC).

SpamShield AI provides an intuitive Streamlit dashboard for analyzing messages, exploring model evaluation metrics, and reviewing prediction history.

## ✨ Features

- **📊 Interactive Dashboard** — Explore project information and model evaluation results.
- **🔍 Message Analyzer** — Enter an SMS message and classify it as Spam or Ham.
- **🤖 Machine Learning Pipeline** — Transform text into numerical features using TF-IDF and classify messages using SVC.
- **📈 Model Evaluation** — View evaluation metrics and a confusion matrix based on held-out test data.
- **📝 Prediction History** — Review predictions made during the current session.
- **📥 CSV Export** — Export prediction history for further analysis.
- **♻️ Reusable Model Artifacts** — Save and reuse the trained model and TF-IDF vectorizer.
- **🧹 Text Preprocessing** — Apply shared text-cleaning logic during training and inference.

## 🧰 Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Pandas | Dataset loading and manipulation |
| Scikit-learn | TF-IDF, SVC, data splitting, and evaluation |
| NLTK | Text preprocessing resources |
| Streamlit | Interactive web application |
| Matplotlib / Seaborn | Evaluation visualizations, where used |

## 🏗️ Project Structure

```text
SpamShield-AI/
├── app.py                  # Streamlit application
├── train_model.py          # Model training and evaluation
├── model_utils.py          # Shared text preprocessing utilities
├── spam.csv                # SMS spam dataset
├── model_metrics.json      # Saved evaluation metrics
├── requirements.txt        # Python dependencies
├── .gitignore              # Ignored files and artifacts
└── README.md               # Project documentation
```

The trained model bundle (`spam_model_bundle.joblib`) is generated locally during training and is not required to be committed to the repository.

## 📂 Dataset

The project uses the **SMS Spam Collection Dataset**, which contains SMS messages labeled as spam or legitimate (ham).

The training script supports datasets with the following column formats:

- `v1` and `v2`
- `Target` and `Text`

If you are using a different version of the dataset, make sure its file format and column names are compatible with the training script.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Eng-7oda/SpamShield-AI.git
cd SpamShield-AI
```

### 2. Create and activate a Python environment

Using Anaconda:

```bash
conda create -n ml python=3.12 -y
conda activate ml
```

Alternatively, use an existing Python environment compatible with the project dependencies.

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Verify the dataset

Make sure `spam.csv` is located in the project root directory, alongside `train_model.py`.

If the dataset is not included in your downloaded copy, obtain the SMS Spam Collection Dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) and prepare it in the format expected by the training script.

## 🧠 Train the Model

Run the training script:

```bash
python train_model.py
```

The script loads and preprocesses the dataset, splits it into training and testing sets using a stratified 70/30 split, trains the TF-IDF and SVC pipeline, and evaluates its performance on the held-out test set.

The generated artifacts include:

- `spam_model_bundle.joblib` — The trained model and associated vectorizer.
- `model_metrics.json` — Saved evaluation metrics.

**Note:** The first run may download the NLTK resources required by the preprocessing pipeline.

## 🚀 Run the Application

After training completes successfully, launch the Streamlit application:

```bash
streamlit run app.py
```

Open the local URL printed in your terminal, usually:

```text
http://localhost:8501
```

You can then analyze SMS messages, explore evaluation metrics, and review prediction history through the dashboard.

## 📏 Model Evaluation

The application provides model evaluation information, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The reported metrics are based on the held-out test set. Refer to `model_metrics.json` for the actual results produced by your training run.

For spam detection, precision and recall are particularly important: false positives may incorrectly flag legitimate messages, while false negatives may allow spam messages to pass undetected.

## 🔄 Machine Learning Workflow

```text
SMS Dataset
    ↓
Data Loading and Validation
    ↓
Text Preprocessing
    ↓
Stratified Train/Test Split
    ↓
TF-IDF Feature Extraction
    ↓
Support Vector Classifier (SVC)
    ↓
Model Evaluation
    ↓
Save Model and Metrics
    ↓
Streamlit Application
    ↓
SMS Prediction and History
```

The TF-IDF vectorizer is fitted on the training data and reused during inference, helping maintain consistency between training and prediction.

## ⚠️ Limitations

- The current preprocessing pipeline is primarily designed for English-language SMS messages.
- Cleaning removes digits, punctuation, URLs, and non-English characters, which may remove useful information for spam detection.
- Prediction history is stored only for the current application session; it is not a persistent database.
- The application displays predicted classes rather than unsupported probability-based confidence scores.
- Model performance depends on the dataset, preprocessing choices, and training configuration.

## 🔮 Future Improvements

- Add support for multilingual SMS messages.
- Experiment with alternative text representations and classification algorithms.
- Improve preprocessing to preserve useful spam indicators, such as URLs and suspicious numbers.
- Add persistent prediction history and analytics.
- Deploy the application to a public hosting platform.
- Evaluate robustness on unseen and more recent spam messages.

## 🔐 Security Note

Only load serialized model files from trusted sources. Files created with `joblib` or Python pickle may execute malicious code when loaded.

## 👨‍💻 Author

**Mahmoud Mosad (Eng-7oda)**

- GitHub: [@Eng-7oda](https://github.com/Eng-7oda)
- Project Repository: [SpamShield-AI](https://github.com/Eng-7oda/SpamShield-AI)

---

*SpamShield AI — Machine Learning for SMS Spam Detection.*
