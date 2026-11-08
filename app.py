"""Single-page glass-style dashboard for the Online Shoppers classifiers."""
from pathlib import Path
import json, joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

ROOT=Path(__file__).resolve().parent
MD=ROOT/"model"; TARGET="Revenue"
MODELS={"Logistic Regression":"logistic_regression.joblib","Decision Tree":"decision_tree.joblib",
"K-Nearest Neighbors":"knn.joblib","Gaussian Naive Bayes":"gaussian_nb.joblib","Support Vector Machine":"svm.joblib"}
METRICS=["Accuracy","AUC","Precision","Recall","F1-score","MCC"]

st.set_page_config(page_title="Shopper Insight Lab",page_icon="🧩",layout="wide",initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;500;600;700;800;900&display=swap');

/* Global typography and readable ink */
:root {
    --ink: #263247;
    --ink-strong: #19263d;
    --muted: #526176;
    --coral: #df4545;
    --glass: rgba(255,255,255,.88);
}
html, body, [class*="css"], button, input, textarea, select {
    font-family: 'Nunito', sans-serif !important;
}
.stApp {
    color: var(--ink) !important;
    background:
      radial-gradient(circle at 8% 8%, rgba(255,210,110,.40), transparent 28%),
      radial-gradient(circle at 92% 12%, rgba(255,135,135,.30), transparent 30%),
      linear-gradient(135deg,#fff8e9,#fff0ec 52%,#edf4ff);
}
.block-container {max-width:1440px;padding-top:1.5rem;padding-bottom:3rem;}
.stApp p, .stApp label, .stApp li, .stApp [data-testid="stMarkdownContainer"] {
    color: var(--ink) !important;
}
h1,h2,h3,h4,h5,h6 {color:var(--ink-strong)!important;font-weight:900!important;}

/* Hero banner */
.hero {
    padding:30px 34px;border-radius:28px;
    background:linear-gradient(115deg,rgba(222,65,65,.98),rgba(255,133,91,.96));
    border:1px solid rgba(255,255,255,.8);
    box-shadow:0 18px 45px rgba(164,70,57,.18);margin-bottom:22px;
}
.hero h1 {color:#fff!important;font-size:clamp(2rem,4vw,3rem);margin:0;}
.hero p {color:#fff!important;margin:7px 0 0;}
.eyebrow {color:#fff8e5!important;font-size:.78rem;font-weight:900;letter-spacing:2px;}

/* Frosted glass surfaces and metrics */
.glass {
    background:rgba(255,255,255,.88)!important;
    border:1px solid rgba(255,255,255,.98)!important;
    border-radius:22px;padding:18px 21px;margin:8px 0 15px;
    box-shadow:0 10px 30px rgba(48,63,89,.09);
    color:var(--ink)!important;
}
div[data-testid="stMetric"] {
    background:rgba(255,255,255,.94)!important;border:1px solid #fff!important;
    border-radius:18px!important;padding:14px!important;
    box-shadow:0 8px 24px rgba(48,63,89,.08);
}
[data-testid="stMetricLabel"],[data-testid="stMetricLabel"] p {color:#526176!important;font-weight:800!important;}
[data-testid="stMetricValue"],[data-testid="stMetricValue"] div {color:#19263d!important;font-weight:900!important;}

/* Buttons */
.stButton>button,.stDownloadButton>button {
    background:linear-gradient(110deg,#df4545,#f47758)!important;
    color:#fff!important;border:0!important;border-radius:13px!important;font-weight:900!important;
}
.stButton>button *, .stDownloadButton>button * {color:#fff!important;}

/* FILE UPLOADER: explicitly override Streamlit's dark dropzone */
[data-testid="stFileUploader"] {
    background:rgba(255,255,255,.94)!important;
    border:1px dashed #df8d75!important;border-radius:18px!important;
    padding:12px!important;color:#263247!important;
}
[data-testid="stFileUploader"] section,
[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] > div {
    background:#fff!important;
    color:#263247!important;
    border-radius:13px!important;
}
[data-testid="stFileUploader"] *,
[data-testid="stFileUploaderDropzone"] * {
    color:#263247!important;
    opacity:1!important;
}
[data-testid="stFileUploader"] button {
    background:#fff!important;color:#263247!important;
    border:1px solid #aab5c5!important;border-radius:10px!important;
}
[data-testid="stFileUploader"] button * {color:#263247!important;}
[data-testid="stFileUploader"] small {color:#526176!important;}

/* SELECTBOXES / DROPDOWNS: selected value, arrow, menu and options */
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {color:#263247!important;font-weight:800!important;}
div[data-testid="stSelectbox"] [data-baseweb="select"] > div,
div[data-testid="stMultiSelect"] [data-baseweb="select"] > div {
    background:#fff!important;color:#263247!important;
    border:1px solid #c5cedb!important;border-radius:12px!important;
}
div[data-testid="stSelectbox"] [data-baseweb="select"] *,
div[data-testid="stMultiSelect"] [data-baseweb="select"] * {
    color:#263247!important;fill:#263247!important;opacity:1!important;
}
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"],
div[role="listbox"] {
    background:#fff!important;color:#263247!important;
    border:1px solid #d7deea!important;
}
li[role="option"], div[role="option"] {
    background:#fff!important;color:#263247!important;opacity:1!important;
}
li[role="option"] *, div[role="option"] * {color:#263247!important;opacity:1!important;}
li[role="option"]:hover, div[role="option"]:hover,
li[role="option"][aria-selected="true"], div[role="option"][aria-selected="true"] {
    background:#fff0e8!important;color:#9f2f2f!important;
}

/* Text inputs and other form controls */
[data-baseweb="input"] > div, [data-baseweb="textarea"] > div,
.stTextInput input, .stNumberInput input, textarea {
    background:#fff!important;color:#263247!important;
    border:1px solid #c5cedb!important;
}
input::placeholder, textarea::placeholder {color:#69778b!important;opacity:1!important;}
[data-testid="stRadio"] label, [data-testid="stCheckbox"] label {color:#263247!important;}
[data-testid="stRadio"] *, [data-testid="stCheckbox"] * {color:#263247!important;}

/* Tables, expanders and alerts */
[data-testid="stDataFrame"],[data-testid="stTable"] {
    background:rgba(255,255,255,.96)!important;border-radius:14px!important;
}
div[data-testid="stExpander"] {
    background:rgba(255,255,255,.86)!important;
    border:1px solid #fff!important;border-radius:16px!important;
}
div[data-testid="stExpander"] summary,
div[data-testid="stExpander"] summary p {color:#263247!important;font-weight:800!important;}
div[data-testid="stAlert"] {color:#263247!important;}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><div class="eyebrow">ONLINE SHOPPING · MACHINE LEARNING STUDIO</div>
<h1>Shopper Insight Lab ✨</h1><p>Explore visitor behaviour, compare five classifiers, and see where predictions meet reality.</p></div>""",unsafe_allow_html=True)


#-----------------------------------------------------------------------
meta_file = MD / "metadata.json"
missing_models = [filename for filename in MODELS.values() if not (MD / filename).exists()]
if not meta_file.exists() or missing_models:
    st.error("Trained model files are missing. Keep the model/ folder beside app.py and run train.py if needed.")
    if missing_models:
        st.write("Missing model files:", ", ".join(missing_models))
    st.stop()

meta = json.loads(meta_file.read_text(encoding="utf-8"))

@st.cache_resource
def load_model(path):
    return joblib.load(path)

@st.cache_data
def read_csv_bytes(file_bytes):
    from io import BytesIO
    return pd.read_csv(BytesIO(file_bytes))

def read_csv_path(path):
    return pd.read_csv(path)

def target_values(series):
    mapping = {
        "false": 0, "true": 1, "0": 0, "1": 1,
        "no": 0, "yes": 1, "0.0": 0, "1.0": 1
    }
    return series.astype(str).str.strip().str.lower().map(mapping)

def calculate_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob) if pd.Series(y_true).nunique() == 2 else np.nan,
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-score": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred)
    }

def make_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.imshow(cm, cmap="YlOrRd")
    ax.set_xticks([0, 1], ["No purchase", "Purchase"])
    ax.set_yticks([0, 1], ["No purchase", "Purchase"])
    ax.set_xlabel("Predicted class", color="#263247", fontsize=10, fontweight="bold")
    ax.set_ylabel("Actual class", color="#263247", fontsize=10, fontweight="bold")
    ax.set_title("Confusion matrix", color="#19263d", fontsize=13, fontweight="bold")
    ax.tick_params(colors="#263247")
    threshold = cm.max() / 2 if cm.size else 0
    for (row, col), value in np.ndenumerate(cm):
        ax.text(col, row, f"{value:,}", ha="center", va="center",
                color="white" if value > threshold else "#263247",
                fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig

def feature_dictionary(frame, target):
    feature_cols = [col for col in frame.columns if col != target]
    categorical, numerical = [], []
    for col in feature_cols:
        dtype = frame[col].dtype
        if (pd.api.types.is_object_dtype(dtype)
            or pd.api.types.is_string_dtype(dtype)
            or pd.api.types.is_bool_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)):
            categorical.append(col)
        else:
            numerical.append(col)
    rows = []
    for col in feature_cols:
        rows.append({
            "Feature": col,
            "Role": "Input feature",
            "Feature type": "Categorical" if col in categorical else "Numerical",
            "Data type": str(frame[col].dtype),
            "Unique values": int(frame[col].nunique(dropna=True)),
            "Missing values": int(frame[col].isna().sum())
        })
    return feature_cols, numerical, categorical, pd.DataFrame(rows)

def validate_dataset(frame, source_name):
    if frame.empty:
        st.error(f"{source_name} contains no records.")
        st.stop()
    frame = frame.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    expected_features = meta.get("feature_columns", [])
    required = expected_features + [TARGET]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        st.error("This CSV does not match the model's expected dataset format.")
        st.write("Missing columns:", ", ".join(missing))
        st.info("Use the labelled Online Shoppers CSV format with all 17 input features and the Revenue target.")
        st.stop()
    labels = target_values(frame[TARGET])
    if labels.isna().any():
        st.error("Revenue must contain True/False, Yes/No, or 0/1 values.")
        st.stop()
    frame[TARGET] = labels.astype(int)
    return frame

# ---------------------------------------------------------------------------

st.markdown("### 01 · Dataset selection")
st.markdown(
    "<p style='color:#526176'>By default, this dashboard loads the original UCI Online Shoppers "
    "Purchasing Intention dataset from <code>data/online_shoppers_intention.csv</code>. "
    "To test another labelled dataset, download the sample CSV, upload it, and click "
    "<b>Evaluate Dataset</b>. Every section below will then refresh using that uploaded file.</p>",
    unsafe_allow_html=True
)

upload_col, download_col = st.columns([1.45, 1])
with upload_col:
    uploaded_file = st.file_uploader(
        "Upload a labelled test CSV",
        type=["csv"],
        help="The CSV must contain the same 17 input features and the Revenue target."
    )

with download_col:
    st.markdown(
        "<div class='glass'><b>Try the application yourself</b><br>"
        "<span style='color:#526176'>Download this labelled sample, upload it, then submit it for evaluation.</span></div>",
        unsafe_allow_html=True
    )
    sample_file = ROOT / "sample_upload_test_data.csv"
    preset_file = ROOT / "test_data.csv"
    downloadable = sample_file if sample_file.exists() else preset_file
    if downloadable.exists():
        st.download_button(
            "⬇ Download labelled test CSV",
            data=downloadable.read_bytes(),
            file_name="sample_upload_test_data.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.warning("Place sample_upload_test_data.csv or test_data.csv beside app.py to enable this download.")

current_upload_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
button_col, help_col = st.columns([1, 2])
with button_col:
    submit_clicked = st.button(
        "🚀 Evaluate Dataset",
        type="primary",
        use_container_width=True,
        disabled=current_upload_bytes is None
    )
with help_col:
    st.markdown(
        "<div class='glass' style='padding:12px 16px;margin:0'>"
        "<b>How it works</b><br><span style='color:#526176'>No upload? The dashboard uses the original UCI data. "
        "After uploading, click Evaluate Dataset to switch all dashboard results to your file.</span></div>",
        unsafe_allow_html=True
    )

if submit_clicked and current_upload_bytes is not None:
    st.session_state["active_uploaded_bytes"] = current_upload_bytes
    st.session_state["active_uploaded_name"] = uploaded_file.name

saved_upload_bytes = st.session_state.get("active_uploaded_bytes")
saved_upload_name = st.session_state.get("active_uploaded_name")

# If the user has selected a different file but has not submitted it yet,
# keep showing the previous active dataset and clearly request submission.
if current_upload_bytes is not None and current_upload_bytes != saved_upload_bytes:
    st.info("A new CSV is selected but not yet evaluated. Click **Evaluate Dataset** to apply it.")
    current_upload_bytes = None

if saved_upload_bytes is not None:
    try:
        active_data = read_csv_bytes(saved_upload_bytes)
        active_source = f"Uploaded dataset: {saved_upload_name}"
        is_uploaded_dataset = True
    except Exception as exc:
        st.error(f"Could not read the uploaded CSV: {exc}")
        st.stop()
else:
    default_path = ROOT / "data" / "online_shoppers_intention.csv"
    if not default_path.exists():
        st.error("The default UCI dataset was not found. Expected location: data/online_shoppers_intention.csv")
        st.stop()
    active_data = read_csv_path(default_path)
    active_source = "Original UCI dataset · data/online_shoppers_intention.csv"
    is_uploaded_dataset = False

try:
    active_data = validate_dataset(active_data, active_source)
except Exception as exc:
    st.error(f"Dataset validation failed: {exc}")
    st.stop()

# ---------------------------------------------------------------------------

st.markdown("### 02 · Dataset overview")
if is_uploaded_dataset:
    st.success(f"All dashboard sections are using: {active_source}")
else:
    st.info(
        "Default view: these details and all model results are based on the original UCI dataset "
        "stored in the project's data/ directory. Upload a labelled CSV and click Evaluate Dataset "
        "to replace the active dataset throughout this page."
    )

st.markdown(
    """<div class="glass" style="padding:16px 18px;margin:10px 0 16px 0">
    <b>About this dataset</b>
    <p style="color:#526176;margin:8px 0 10px 0">
    The <i>Online Shoppers Purchasing Intention</i> dataset records 12,330 e-commerce
    browsing sessions. It describes visitor behaviour such as the types of pages viewed
    and time spent on them, browsing and exit patterns, page values, visitor type,
    traffic source, device/browser details, and visit timing.
    </p>
    <p style="color:#263247;margin:0 0 8px 0">
    <b>Target value: <code>Revenue</code></b> is the binary outcome the models predict:
    <code>True (1)</code> means the session ended in a purchase, while
    <code>False (0)</code> means no purchase was made.
    </p>
    <a href="https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset"
       target="_blank" style="color:#c94747;font-weight:800;text-decoration:underline">
       View and download the original dataset from the UCI Machine Learning Repository ↗
    </a>
    </div>""",
    unsafe_allow_html=True
)

feature_cols, numerical_cols, categorical_cols, feature_info = feature_dictionary(active_data, TARGET)
actual_y = active_data[TARGET].astype(int)

overview_cols = st.columns(4)
overview_cols[0].metric("Records", f"{len(active_data):,}")
overview_cols[1].metric("Input features", f"{len(feature_cols)}")
overview_cols[2].metric("Numerical features", f"{len(numerical_cols)}")
overview_cols[3].metric("Categorical features", f"{len(categorical_cols)}")

target_cols = st.columns(3)
target_cols[0].metric("Purchases", f"{int((actual_y == 1).sum()):,}")
target_cols[1].metric("No purchases", f"{int((actual_y == 0).sum()):,}")
target_cols[2].metric("Purchase rate", f"{actual_y.mean() * 100:.1f}%")

with st.expander("Explore feature dictionary"):
    st.dataframe(feature_info, use_container_width=True, hide_index=True)
with st.expander("Preview active dataset"):
    st.dataframe(active_data.head(20), use_container_width=True, hide_index=True)

# ---------------------------------------------------------------------------

X = active_data[meta["feature_columns"]].copy()
y = active_data[TARGET].astype(int)
all_predictions = {}
comparison_rows = []

with st.spinner("Running all five classifiers on the active dataset..."):
    for model_name, model_file in MODELS.items():
        estimator = load_model(str(MD / model_file))
        predictions = estimator.predict(X)
        probabilities = estimator.predict_proba(X)[:, 1]
        all_predictions[model_name] = {
            "model": estimator,
            "predictions": predictions,
            "probabilities": probabilities,
            "metrics": calculate_metrics(y, predictions, probabilities)
        }
        comparison_rows.append({"Model": model_name, **all_predictions[model_name]["metrics"]})

comparison = pd.DataFrame(comparison_rows)

st.markdown("### 03 · Individual model evaluation")
st.markdown(
    "<p style='color:#526176'>Choose a classifier to inspect its performance, confusion matrix, "
    "classification report and row-level predictions on the active dataset.</p>",
    unsafe_allow_html=True
)
selected_model_name = st.selectbox("Choose a classifier", list(MODELS.keys()))
selected_result = all_predictions[selected_model_name]
selected_predictions = selected_result["predictions"]
selected_probabilities = selected_result["probabilities"]

metric_cols = st.columns(6)
for metric_col, metric_name in zip(metric_cols, METRICS):
    metric_value = selected_result["metrics"][metric_name]
    metric_col.metric(metric_name, f"{metric_value:.3f}" if pd.notna(metric_value) else "N/A")

matrix_col, report_col = st.columns([1, 1.25])
with matrix_col:
    st.pyplot(make_confusion_matrix(y, selected_predictions), use_container_width=True)
with report_col:
    st.markdown("#### Classification report")
    report = classification_report(
        y, selected_predictions,
        labels=[0, 1],
        target_names=["No purchase", "Purchase"],
        output_dict=True,
        zero_division=0
    )
    st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

st.markdown("### 04 · Predicted vs actual")
st.markdown(
    "<p style='color:#526176'>Inspect each session's actual outcome, predicted outcome, "
    "purchase probability and whether the classifier was correct.</p>",
    unsafe_allow_html=True
)
prediction_view = X.copy()
prediction_view["Actual"] = y.map({0: "No purchase", 1: "Purchase"}).values
prediction_view["Predicted"] = pd.Series(selected_predictions).map(
    {0: "No purchase", 1: "Purchase"}
).values
prediction_view["Purchase probability"] = np.round(selected_probabilities, 4)
prediction_view["Result"] = np.where(
    selected_predictions == y.to_numpy(), "Correct", "Incorrect"
)

filter_choice = st.selectbox(
    "Filter prediction rows",
    ["All rows", "Incorrect only", "Correct only"]
)
if filter_choice == "Incorrect only":
    displayed_predictions = prediction_view[prediction_view["Result"] == "Incorrect"]
elif filter_choice == "Correct only":
    displayed_predictions = prediction_view[prediction_view["Result"] == "Correct"]
else:
    displayed_predictions = prediction_view

st.caption(
    f"Showing {len(displayed_predictions):,} of {len(prediction_view):,} rows · "
    f"Selected model: {selected_model_name}"
)
st.dataframe(
    displayed_predictions,
    use_container_width=True,
    hide_index=True,
    height=390
)
st.download_button(
    "⬇ Download actual-vs-predicted results",
    prediction_view.to_csv(index=False).encode("utf-8"),
    "predictions.csv",
    "text/csv"
)

st.markdown("### 05 · Five-model comparison")
st.markdown(
    "<p style='color:#526176'>All five saved classifiers are evaluated against exactly the same active dataset. "
    "Choose a metric to rank the models and select metrics for the comparison chart.</p>",
    unsafe_allow_html=True
)
rank_metric = st.selectbox("Rank models by metric", METRICS, index=4)
ranked_comparison = comparison.sort_values(rank_metric, ascending=False).reset_index(drop=True)
ranked_comparison.insert(0, "Rank", range(1, len(ranked_comparison) + 1))
st.dataframe(
    ranked_comparison.style.format({metric: "{:.3f}" for metric in METRICS}),
    use_container_width=True,
    hide_index=True
)

chart_metrics = st.multiselect(
    "Metrics to include in comparison chart",
    METRICS,
    default=["Accuracy", "AUC", "F1-score", "MCC"]
)
if chart_metrics:
    st.bar_chart(
        comparison.set_index("Model")[chart_metrics],
        use_container_width=True
    )

st.markdown("#### Precision, Recall and F1 comparison")
st.line_chart(
    comparison.set_index("Model")[["Precision", "Recall", "F1-score"]],
    use_container_width=True
)
st.download_button(
    "⬇ Download five-model comparison",
    comparison.to_csv(index=False).encode("utf-8"),
    "model_comparison.csv",
    "text/csv"
)

best_model = ranked_comparison.iloc[0]
st.markdown(
    f"<div class='glass'><b>Leader by {rank_metric}:</b> {best_model['Model']} "
    f"({best_model[rank_metric]:.3f})<br><span style='color:#526176'>"
    "The leader depends on the selected metric. For an imbalanced dataset, consider "
    "Recall, F1 and MCC alongside Accuracy.</span></div>",
    unsafe_allow_html=True
)

with st.expander("What do these evaluation metrics mean?"):
    st.markdown("""
    - **Accuracy:** proportion of all predictions that are correct.
    - **AUC:** how well the model ranks purchasers above non-purchasers across thresholds.
    - **Precision:** among predicted purchasers, how many actually purchased.
    - **Recall:** among actual purchasers, how many were identified.
    - **F1-score:** harmonic balance between precision and recall.
    - **MCC:** correlation-style score using all four confusion-matrix outcomes. 1 is perfect,
      0 is chance-level, and -1 is total disagreement.
    """)

st.markdown(
    "<div style='text-align:center;padding:18px;color:#64748b'>"
    "Five pre-trained classifiers · One active dataset · Full evaluation dashboard</div>",
    unsafe_allow_html=True
)
