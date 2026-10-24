from pathlib import Path
from datetime import datetime
import io
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from model_utils import clean_and_lemmatize

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "spam_model_bundle.joblib"

st.set_page_config(page_title="SpamShield AI", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root { --navy:#081225; --panel:#101d35; --panel2:#142541; --blue:#3b82f6; --cyan:#38bdf8; --muted:#9aabc5; }
html, body, [class*="css"] { font-family: Inter, sans-serif; }
.stApp { background: radial-gradient(ellipse at top right, #122b52 0%, #081225 45%, #070d1b 100%); color:#edf4ff; }
section[data-testid="stSidebar"] { background: linear-gradient(180deg,#0b1730,#080f20); border-right:1px solid #1c3152; }
section[data-testid="stSidebar"] * { color:#e7efff; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }
.hero { padding: 1.5rem 1.7rem; border:1px solid #223a60; border-radius:22px; background:linear-gradient(120deg,rgba(26,53,94,.92),rgba(13,27,50,.92)); margin-bottom:1.3rem; box-shadow:0 18px 55px rgba(0,0,0,.15); }
.eyebrow { color:#7dd3fc; font-size:.76rem; font-weight:800; letter-spacing:.16em; text-transform:uppercase; }
.hero h1 { font-size:2.1rem; letter-spacing:-.04em; margin:.35rem 0 .4rem; color:#f5f8ff; }
.hero p { color:#aebfda; margin:0; font-size:.98rem; }
.metric-card { background:linear-gradient(145deg,#12223e,#0d192e); border:1px solid #233b5d; border-radius:17px; padding:1.1rem 1.2rem; min-height:132px; }
.metric-label { color:#9fb2d0; font-size:.83rem; font-weight:600; margin-bottom:.55rem; }
.metric-value { color:#f4f8ff; font-size:1.8rem; font-weight:800; letter-spacing:-.04em; }
.metric-note { color:#6f89b0; font-size:.76rem; margin-top:.35rem; }
.section-title { font-size:1.15rem; font-weight:750; color:#f3f7ff; margin:.35rem 0 .8rem; }
.result-spam { border:1px solid #8f3446; background:linear-gradient(120deg,rgba(127,29,50,.28),rgba(42,20,42,.5)); border-radius:18px; padding:1.25rem; }
.result-ham { border:1px solid #1d806b; background:linear-gradient(120deg,rgba(6,95,70,.27),rgba(12,42,52,.5)); border-radius:18px; padding:1.25rem; }
.small-muted { color:#9aabc5; font-size:.86rem; }
div[data-testid="stTextArea"] textarea { background:#0b1629; color:#f0f6ff; border:1px solid #2a4266; border-radius:13px; }
.stButton > button { border-radius:11px; border:1px solid #4388ff; background:linear-gradient(100deg,#2563eb,#0ea5e9); color:white; font-weight:700; min-height:2.8rem; }
.stButton > button:hover { border-color:#7dd3fc; color:white; filter:brightness(1.08); }
[data-testid="stDataFrame"] { border:1px solid #223a60; border-radius:12px; overflow:hidden; }
hr { border-color:#203451; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)


def load_bundle():
    if not BUNDLE_PATH.exists():
        return None
    try:
        return joblib.load(BUNDLE_PATH)
    except Exception as exc:
        st.error(f"Could not load the saved model: {exc}")
        return None


bundle = load_bundle()
if "history" not in st.session_state:
    st.session_state.history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

with st.sidebar:
    st.markdown("<div style='padding:8px 0 18px'><div style='font-size:2rem'>🛡️</div><div style='font-size:1.35rem;font-weight:800;letter-spacing:-.04em'>SpamShield <span style='color:#60a5fa'>AI</span></div><div style='font-size:.78rem;color:#8fa6c9'>SMS THREAT DETECTION</div></div>", unsafe_allow_html=True)
    page = st.radio("NAVIGATION", ["Dashboard", "Message Analyzer", "Model Metrics", "Prediction History"], label_visibility="visible")
    st.markdown("---")
    if bundle:
        st.success("Model artifacts loaded")
        st.caption("Classifier: SVM · Features: TF-IDF")
    else:
        st.warning("Model not trained yet")
        st.caption("Run `python train_model.py` first.")
    st.markdown("<div style='position:fixed;bottom:22px;color:#62799c;font-size:.72rem'>SPAMSHIELD AI · ML PROJECT</div>", unsafe_allow_html=True)


def hero(title, subtitle, eyebrow="SMS SECURITY INTELLIGENCE"):
    st.markdown(f"<div class='hero'><div class='eyebrow'>{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>", unsafe_allow_html=True)


def metric_card(label, value, note):
    st.markdown(f"<div class='metric-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div><div class='metric-note'>{note}</div></div>", unsafe_allow_html=True)


def show_metrics():
    if not bundle:
        st.info("Train the model first to see evaluation metrics. Run `python train_model.py` in the project folder.")
        return
    metrics = bundle["metrics"]
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("TEST ACCURACY", f"{metrics['accuracy'] * 100:.2f}%", "Overall correct predictions")
    with c2: metric_card("SPAM PRECISION", f"{metrics['spam_precision'] * 100:.2f}%", "Predicted spam that was spam")
    with c3: metric_card("SPAM RECALL", f"{metrics['spam_recall'] * 100:.2f}%", "Spam messages successfully caught")
    with c4: metric_card("SPAM F1-SCORE", f"{metrics['spam_f1'] * 100:.2f}%", "Precision/recall balance")

    st.markdown("<div class='section-title'>Confusion Matrix</div>", unsafe_allow_html=True)
    cm = metrics["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(6.2, 4.2), facecolor="#101d35")
    ax.set_facecolor("#101d35")
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Ham", "Spam"])
    ax.set_yticks([0, 1], labels=["Ham", "Spam"])
    ax.set_xlabel("Predicted label", color="#dce8ff", labelpad=10)
    ax.set_ylabel("Actual label", color="#dce8ff", labelpad=10)
    ax.tick_params(colors="#dce8ff")
    for spine in ax.spines.values(): spine.set_color("#304766")
    threshold = max(max(row) for row in cm) / 2 if cm else 0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i][j]), ha="center", va="center", fontsize=16, fontweight="bold", color="white" if cm[i][j] > threshold else "#dce8ff")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    report = metrics.get("classification_report", {})
    if report:
        st.markdown("<div class='section-title'>Classification Report</div>", unsafe_allow_html=True)
        rows = []
        for label in ["Ham", "Spam"]:
            item = report.get(label, {})
            rows.append({"Class": label, "Precision": item.get("precision", 0), "Recall": item.get("recall", 0), "F1-score": item.get("f1-score", 0), "Support": int(item.get("support", 0))})
        st.dataframe(pd.DataFrame(rows).style.format({"Precision":"{:.3f}","Recall":"{:.3f}","F1-score":"{:.3f}"}), hide_index=True, use_container_width=True)
    st.caption(f"Evaluation split: {metrics['test_rows']:,} test messages out of {metrics['dataset_rows']:,} dataset rows. Metrics are from the saved hold-out test set.")


if page == "Dashboard":
    hero("Smarter SMS. Safer inbox.", "Detect suspicious text messages with a machine-learning classifier built using TF-IDF and Support Vector Machines.")
    if not bundle:
        st.warning("Your model has not been trained or its artifacts are missing.")
        st.code("# Place spam.csv beside the Python files, then run:\npython train_model.py\nstreamlit run app.py", language="bash")
    else:
        metrics = bundle["metrics"]
        cols = st.columns(4)
        with cols[0]: metric_card("MODEL STATUS", "Ready", "Saved artifacts loaded")
        with cols[1]: metric_card("TEST ACCURACY", f"{metrics['accuracy']*100:.2f}%", "Measured on held-out data")
        with cols[2]: metric_card("TRAINING SAMPLES", f"{metrics['train_rows']:,}", "Messages used to fit the model")
        with cols[3]: metric_card("TEST SAMPLES", f"{metrics['test_rows']:,}", "Messages used for evaluation")
    st.markdown("<div class='section-title'>How it works</div>", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    with a:
        st.markdown("### 01 · Preprocess")
        st.markdown("<p class='small-muted'>Normalize message text, remove English stopwords, and lemmatize words using the same preprocessing for training and inference.</p>", unsafe_allow_html=True)
    with b:
        st.markdown("### 02 · Vectorize")
        st.markdown("<p class='small-muted'>Convert text into numerical TF-IDF features using the vectorizer fitted on the training data.</p>", unsafe_allow_html=True)
    with c:
        st.markdown("### 03 · Classify")
        st.markdown("<p class='small-muted'>The trained SVM classifies the message as Ham (legitimate) or Spam (potentially unwanted).</p>", unsafe_allow_html=True)
    st.info("This is a machine-learning prediction, not a guarantee that a message is safe. Avoid clicking links in unexpected messages.")

elif page == "Message Analyzer":
    hero("Message Analyzer", "Paste an SMS and let the trained SVM classify it.", "REAL-TIME MESSAGE CHECK")
    if not bundle:
        st.error("Model bundle not found. Train it first using `python train_model.py`.")
    else:
        sample_choice = st.selectbox("Try an example", ["Write my own message", "Example · Legitimate", "Example · Suspicious"])
        defaults = {"Write my own message": "", "Example · Legitimate": "Hi, are we still meeting for lunch at 1 pm?", "Example · Suspicious": "URGENT! You have won a free prize. Call now to claim your reward!"}
        message = st.text_area("SMS MESSAGE", value=defaults[sample_choice], height=175, placeholder="Paste the full SMS message here...", help="The message is processed locally by this Streamlit app; don't enter sensitive personal information.")
        left, right = st.columns([1, 2])
        with left:
            analyze = st.button("🔍  Analyze message", use_container_width=True, type="primary")
        with right:
            st.caption(f"{len(message)} characters · English-language preprocessing")
        if analyze:
            if not message.strip():
                st.warning("Enter a message before analyzing it.")
            else:
                with st.spinner("Preprocessing text and running the SVM classifier..."):
                    processed = clean_and_lemmatize(message)
                    vector = bundle["vectorizer"].transform([processed])
                    prediction = int(bundle["model"].predict(vector)[0])
                    label = "Spam" if prediction == 1 else "Ham"
                    explanation = "Potentially unwanted or deceptive message. Treat links and requests for personal information with caution." if prediction == 1 else "The model classified this message as legitimate (Ham). Still verify unexpected links or requests."
                    result = {"Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Prediction": label, "Message": message.strip(), "Processed text": processed}
                    st.session_state.last_result = result
                    st.session_state.history.insert(0, result)
                if label == "Spam":
                    st.markdown(f"<div class='result-spam'><div style='font-size:.78rem;letter-spacing:.12em;color:#fda4af;font-weight:800'>ANALYSIS RESULT</div><h2 style='color:#fecdd3;margin:.35rem 0'>🚨 Spam detected</h2><div style='color:#e9b8c3'>{explanation}</div></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='result-ham'><div style='font-size:.78rem;letter-spacing:.12em;color:#6ee7b7;font-weight:800'>ANALYSIS RESULT</div><h2 style='color:#a7f3d0;margin:.35rem 0'>🛡️ Classified as Ham</h2><div style='color:#b2d9d3'>{explanation}</div></div>", unsafe_allow_html=True)
                with st.expander("View preprocessing output"):
                    st.code(processed or "(No tokens remain after preprocessing)")
                st.caption("No confidence percentage is shown because this SVC model was not trained with probability estimates.")

elif page == "Model Metrics":
    hero("Model Performance", "Evaluation metrics from the held-out test split—not made-up demo values.", "MODEL QUALITY & VALIDATION")
    show_metrics()

elif page == "Prediction History":
    hero("Prediction History", "Review the messages classified during this app session.", "RECENT ANALYSES")
    history = st.session_state.history
    if not history:
        st.info("No predictions yet. Open Message Analyzer and analyze your first SMS.")
    else:
        df_history = pd.DataFrame(history)
        c1, c2, c3 = st.columns(3)
        with c1: metric_card("TOTAL ANALYZED", str(len(df_history)), "This session")
        with c2: metric_card("SPAM FLAGS", str(int((df_history["Prediction"] == "Spam").sum())), "Classified as spam")
        with c3: metric_card("HAM FLAGS", str(int((df_history["Prediction"] == "Ham").sum())), "Classified as legitimate")
        st.dataframe(df_history[["Timestamp", "Prediction", "Message"]], use_container_width=True, hide_index=True)
        csv_bytes = df_history.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ Download history as CSV", data=csv_bytes, file_name="spamshield_prediction_history.csv", mime="text/csv")
        if st.button("Clear session history"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()
        st.caption("History is stored in Streamlit session state and is cleared when the session resets. Messages may contain personal information; export and share with care.")
