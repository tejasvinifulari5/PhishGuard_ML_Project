"""Streamlit interface for the local PhishGuard URL classifier."""

import ipaddress
import json
import logging
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import streamlit as st

from phishguard_pipeline import normalize_url, transform_urls


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_DIR / "models" / "phishing_model.joblib"
METADATA_PATH = PROJECT_DIR / "models" / "model_metadata.json"
SHORTENER_HOSTS = {
    "bit.ly",
    "buff.ly",
    "is.gd",
    "ow.ly",
    "t.co",
    "tinyurl.com",
    "t.ly",
    "rb.gy",
    "shorturl.at",
}
UNUSUAL_CHARACTERS = set("!$^*{}[]|\\`<>")
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title="PhishGuard | URL Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_theme() -> None:
    st.markdown(
        """
        <style>
        :root { color-scheme: light; }
        .stApp { background: #f4f7fb; color: #172b4d; }
        [data-testid="stHeader"] { background: rgba(255, 255, 255, .96); }
        [data-testid="stToolbar"] { right: 1rem; }
        .block-container { max-width: 1480px; padding: 1.7rem clamp(1.1rem, 2.5vw, 2.5rem) 3rem; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #e2e8f0; }
        [data-testid="stSidebar"] > div:first-child { background: #ffffff; }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: 1.25rem 1rem; }
        .sidebar-brand { display: flex; align-items: center; gap: .7rem; margin: .2rem 0 .35rem; }
        .sidebar-brand-mark { display: inline-flex; width: 2.35rem; height: 2.35rem;
            align-items: center; justify-content: center; flex: 0 0 auto;
            border-radius: 10px; color: #1769d2; background: #eaf2ff; font-size: 1.35rem; }
        .sidebar-brand-name { color: #142b4a; font-size: 1.2rem; font-weight: 750; line-height: 1.2; }
        .sidebar-subtitle { margin: 0 0 1rem .1rem; color: #52667f; font-size: .9rem; }
        [data-testid="stSidebar"] [data-testid="stRadio"] label {
            min-height: 2.65rem; margin: .16rem 0; padding: .42rem .65rem;
            border: 1px solid transparent; border-radius: 10px;
            color: #263d59; font-size: .98rem; font-weight: 550;
            line-height: 1.35; transition: background-color .15s ease, border-color .15s ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {
            background: #f2f6fc; border-color: #e2eaf4;
        }
        [data-testid="stSidebar"] [data-testid="stRadioOption"]:has(input:checked) {
            background: #eaf2ff; border-color: #c8dcfb; color: #124f9e; font-weight: 700;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label p {
            color: inherit; font-size: inherit; line-height: inherit;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details {
            border: 1px solid #e2e8f0; border-radius: 11px; background: #ffffff;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            min-height: 2.8rem; color: #263d59; font-weight: 650;
        }
        h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {
            color: #172b4d;
        }
        h1 { letter-spacing: -.035em; }
        .hero { display: flex; align-items: center; gap: 1rem; padding: 1.55rem 1.8rem;
                border: 1px solid #dce6f2; border-radius: 18px; background: #ffffff;
                margin-bottom: 1.4rem; box-shadow: 0 5px 20px rgba(21, 48, 79, .05); }
        .hero h1 { margin: 0; color: #142b4a; font-size: 2.25rem; }
        .hero p { color: #53677f; margin: .35rem 0 0; }
        .shield { display: inline-flex; align-items: center; justify-content: center;
                  min-width: 3.7rem; height: 3.7rem; border-radius: 14px;
                  color: #1769d2; background: #eaf2ff; font-size: 2rem; }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff; border-color: #dce5ef; border-radius: 16px;
            box-shadow: 0 4px 16px rgba(21, 48, 79, .035);
        }
        [data-testid="stMetric"] { background: #ffffff; padding: .8rem 1rem;
            border: 1px solid #e2e8f0; border-radius: 12px; }
        [data-testid="stMetricLabel"] { color: #52667f; }
        [data-testid="stMetricValue"] { color: #173b66; }
        .stButton > button, [data-testid="stFormSubmitButton"] > button {
            min-height: 3.15rem; color: #ffffff; background: #1769d2;
            border: 1px solid #1769d2; border-radius: 10px; font-size: 1rem;
            font-weight: 700; box-shadow: 0 3px 8px rgba(23, 105, 210, .16);
        }
        .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
            color: #ffffff; background: #1057b4; border-color: #1057b4;
            box-shadow: 0 5px 12px rgba(23, 105, 210, .22);
        }
        [data-testid="stFormSubmitButton"] > button {
            width: 100%; white-space: nowrap;
        }
        [data-baseweb="input"] > div, [data-baseweb="textarea"] > div {
            min-height: 3.15rem; background: #ffffff; border-color: #cbd6e2;
        }
        [data-testid="stTextInput"] label {
            color: #263d59; font-size: .98rem; font-weight: 650;
        }
        .result-card { padding: 1.25rem 1.45rem; border-radius: 14px; margin: .9rem 0;
                       border: 1px solid; background: #ffffff; }
        .result-card h2 { margin: 0 0 .25rem; font-size: 1.45rem; }
        .result-card p { margin: 0; color: #40556e; }
        .result-good { border-color: #8bc8a1; background: #f0faf3; }
        .result-good h2 { color: #187342; }
        .result-alert { border-color: #e4a0a0; background: #fff4f3; }
        .result-alert h2 { color: #b42318; }
        .section-eyebrow { color: #1769d2; font-size: .82rem; font-weight: 700;
            letter-spacing: .08em; text-transform: uppercase; margin-bottom: .2rem; }
        .muted-copy { color: #52667f; }
        [data-testid="stExpander"] details {
            border: 1px solid #dce5ef; border-radius: 12px; background: #ffffff;
        }
        [data-testid="stExpander"] summary {
            min-height: 2.9rem; color: #263d59; font-weight: 650;
        }
        [data-testid="stAlert"] { border-radius: 11px; }
        @media (max-width: 700px) {
            .block-container { padding: 1rem .75rem 2rem; }
            [data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding: .85rem .7rem; }
            [data-testid="stSidebar"] [data-testid="stRadio"] label {
                min-height: 2.8rem; font-size: 1rem;
            }
            .hero { padding: 1.1rem; align-items: flex-start; gap: .8rem; }
            .hero h1 { font-size: 1.75rem; }
            .shield { min-width: 3rem; height: 3rem; font-size: 1.6rem; }
            .stButton > button, [data-testid="stFormSubmitButton"] > button {
                min-height: 3.2rem;
            }
        }
        @media (min-width: 701px) {
            section[data-testid="stSidebar"][aria-expanded="true"] {
                width: 18rem !important; min-width: 18rem !important;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def load_model_resources() -> tuple[dict, dict]:
    """Load the trusted local artifact and its metadata once per app process."""
    bundle = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    if bundle.get("label_mapping") != {"0": "Legitimate", "1": "Potential Phishing"}:
        raise ValueError("The saved model has an unexpected label mapping.")
    if metadata.get("label_mapping") != bundle["label_mapping"]:
        raise ValueError("The model and metadata label mappings do not match.")
    return bundle, metadata


def validate_and_normalize_url(raw_url: str) -> tuple[str | None, str | None]:
    """Apply the shared URL policy without making any network request."""
    try:
        return normalize_url(raw_url), None
    except ValueError as error:
        return None, str(error)


def predict_url(url: str, bundle: dict) -> int:
    """Run the shared trained feature pipeline and return the model class."""
    matrix = transform_urls([url], bundle["vectorizer"], bundle["lexical_scaler"])
    return int(bundle["estimator"].predict(matrix)[0])


def get_manual_characteristics(url: str) -> dict[str, str]:
    """Calculate visible URL clues; these are not an explanation of model weights."""
    parsed = urlsplit(url)
    hostname = parsed.hostname or ""
    host_lower = hostname.casefold()
    try:
        is_ip = ipaddress.ip_address(hostname) is not None
    except ValueError:
        is_ip = False
    host_parts = [part for part in hostname.split(".") if part]
    shortener = host_lower in SHORTENER_HOSTS
    unusual_count = sum(character in UNUSUAL_CHARACTERS for character in url)

    return {
        "URL length": f"{len(url)} characters",
        "Dots in URL": str(url.count(".")),
        "Estimated Subdomains": str(max(len(host_parts) - 2, 0)),
        "Uses an IP Address as Hostname": "Yes" if is_ip else "No",
        "HTTPS scheme": "Yes" if parsed.scheme.casefold() == "https" else "No",
        "Contains @ symbol": "Yes" if "@" in url else "No",
        "Recognized URL Shortener": "Yes" if shortener else "No",
        "Unusual Punctuation Count": str(unusual_count),
    }


def show_result(label: int) -> None:
    if label == 1:
        st.markdown(
            '<div class="result-card result-alert"><h2>⚠ Potential Phishing</h2>'
            "<p>The prediction reflects patterns learned from training data; it is an "
            "estimate, not a guarantee of safety.</p></div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="result-card result-good"><h2>✓ Classified as Legitimate</h2>'
            "<p>The prediction reflects patterns learned from training data; it is an "
            "estimate, not a guarantee of safety.</p></div>",
            unsafe_allow_html=True,
        )
    _ = st.warning(
        "This is a machine-learning estimate, not a guarantee that a URL is safe. "
        "Do not enter credentials on suspicious websites."
    )


def render_model_metrics(metadata: dict) -> None:
    evaluation = metadata.get("evaluation", {})
    metric_names = [
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1", "F1-score"),
        ("roc_auc", "ROC-AUC"),
    ]
    available = [(key, title) for key, title in metric_names if key in evaluation]
    if not available:
        st.info("No saved test-split evaluation metrics are available.")
        return

    st.caption(
        f"Saved holdout results · {metadata.get('test_rows', 'unknown')} stratified test URLs. "
        "These scores are not a guarantee of real-world performance."
    )
    columns = st.columns(len(available))
    for column, (key, title) in zip(columns, available):
        column.metric(title, f"{float(evaluation[key]):.1%}")


def render_model_information(metadata: dict) -> None:
    estimator = metadata.get("estimator", {})
    features = metadata.get("features", {})
    tfidf = features.get("character_tfidf", {})
    rows_used = metadata.get("experiment_rows")
    feature_limit = tfidf.get("max_features")
    first, second = st.columns(2)
    with first:
        st.markdown("#### Model")
        st.write(f"**Estimator:** {metadata.get('model_name', 'Unknown')}")
        st.write(f"**Training data:** {metadata.get('dataset', 'Unknown')}")
        formatted_rows = f"{rows_used:,}" if isinstance(rows_used, int) else "Unknown"
        st.write(f"**Rows used:** {formatted_rows}")
        st.write(f"**Training date (UTC):** {metadata.get('training_date_utc', 'Unknown')}")
    with second:
        st.markdown("#### Feature representation")
        formatted_limit = f"{feature_limit:,}" if isinstance(feature_limit, int) else "unknown"
        st.write(
            f"**Character TF-IDF:** {tfidf.get('ngram_range', 'Unknown')} character n-grams, "
            f"up to {formatted_limit} features"
        )
        st.write("**Lexical features:** 22 URL-only measurements")
        st.write("**Target mapping:** 0 = Legitimate · 1 = Potential Phishing")
        st.write(
            "**Probability:** Not available for "
            + ("LinearSVC" if not estimator.get("supports_probability", False) else "this model")
        )


def render_prediction_characteristics(url: str) -> None:
    st.markdown("#### URL characteristics")
    st.caption(
        "These are visible string observations, not an explanation of which features "
        "caused the prediction."
    )
    characteristics = get_manual_characteristics(url)
    columns = st.columns(4)
    for index, (name, value) in enumerate(characteristics.items()):
        columns[index % len(columns)].metric(name, value)


def render_scanner(bundle: dict) -> None:
    st.markdown(
        '<p class="section-eyebrow">Private · Local URL Analysis</p>',
        unsafe_allow_html=True,
    )
    st.header("URL scanner")
    st.write("Enter a URL to check whether it shows patterns associated with phishing.")

    with st.container(border=True):
        with st.form("url_analysis_form", clear_on_submit=False):
            input_column, button_column = st.columns(
                [5, 1.5], gap="medium", vertical_alignment="bottom"
            )
            with input_column:
                raw_url = st.text_input(
                    "URL to analyze",
                    placeholder="example.com/login or https://example.com/login",
                    help="HTTP and HTTPS are supported. If you omit the scheme, https:// is added.",
                )
            with button_column:
                submitted = st.form_submit_button(
                    "Analyze URL", type="primary", width="stretch"
                )

    if not submitted:
        st.info(
            "The model returns a class label only. LinearSVC does not provide a calibrated "
            "probability, so PhishGuard does not show a confidence percentage."
        )
        return

    normalized_url, validation_error = validate_and_normalize_url(raw_url)
    if validation_error:
        st.error(validation_error)
        return

    try:
        prediction = predict_url(normalized_url, bundle)
    except Exception:
        LOGGER.exception("Failed to classify a submitted URL.")
        st.error("The model could not analyze this input. Check the URL and try again.")
        return

    _ = show_result(prediction)
    _ = render_prediction_characteristics(normalized_url)


def render_performance(metadata: dict) -> None:
    st.markdown('<p class="section-eyebrow">Held-out evaluation</p>', unsafe_allow_html=True)
    st.header("Model performance")
    _ = render_model_metrics(metadata)

    evaluation = metadata.get("evaluation", {})
    matrix = evaluation.get("confusion_matrix")
    labels = evaluation.get("confusion_matrix_labels")
    if matrix and labels:
        left, right = st.columns([1, 1.25])
        with left:
            st.markdown("#### Confusion matrix")
            st.caption("Rows are actual classes; columns are predicted classes.")
            st.dataframe(
                {
                    "Actual / predicted": labels,
                    labels[0]: matrix[0],
                    labels[1]: matrix[1],
                },
                hide_index=True,
                width="stretch",
            )
        with right:
            st.markdown("#### Reading the scores")
            st.write(
                "Precision measures how many phishing alerts were correct. Recall measures "
                "how many phishing URLs were found. F1 balances precision and recall; "
                "ROC-AUC measures ranking from the model's decision scores."
            )
    _ = st.warning(
        "These results come from a held-out split of this dataset. They do not guarantee "
        "performance on new domains or future phishing campaigns."
    )


def render_how_it_works() -> None:
    st.markdown('<p class="section-eyebrow">Behind the prediction</p>', unsafe_allow_html=True)
    st.header("How PhishGuard works")
    steps = st.columns(4)
    for column, title, detail in zip(
        steps,
        ["1 · URL input", "2 · Features", "3 · Trained model", "4 · Prediction"],
        [
            "A pasted URL is validated locally. Missing schemes are normalized to HTTPS.",
            "Character n-grams and 22 lexical URL measurements are extracted.",
            "A saved Linear SVM and its training-fitted preprocessing are loaded from disk.",
            "The app labels the URL Legitimate or Potential Phishing; it does not claim certainty.",
        ],
    ):
        with column.container(border=True):
            st.markdown(f"#### {title}")
            st.write(detail)
    st.info(
        "The saved LinearSVC does not support probabilities. No probability or confidence "
        "percentage is calculated or displayed."
    )


def render_security_tips() -> None:
    st.markdown('<p class="section-eyebrow">Stay safer online</p>', unsafe_allow_html=True)
    st.header("Security tips")
    st.markdown(
        """
        - **Check the hostname carefully.** A brand name in the URL path is not proof that
          the site belongs to that brand.
        - **Be cautious with unexpected login links.** Navigate to the service directly
          instead of following an unsolicited message.
        - **Treat shortened URLs carefully.** Their final destination is not obvious.
        - **Never enter credentials on a site you do not trust.**
        - **Use this prediction as a clue, not a verdict.** A “Legitimate” result does
          not guarantee safety.
        """
    )
    st.divider()
    st.subheader("About the project")
    st.write(
        "PhishGuard is an educational machine-learning project for phishing URL "
        "classification using the locally stored UCI PhiUSIIL dataset. It uses only the "
        "URL string and does not inspect or visit the website."
    )


def main() -> None:
    _ = apply_theme()
    st.markdown(
        '<div class="hero"><span class="shield">&#128737;</span><div>'
        "<h1>PhishGuard</h1><p><strong>Intelligent Phishing URL Detection</strong></p>"
        "<p>Submitted URLs are processed locally and are never opened or sent to an external service.</p>"
        "</div></div>",
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.is_file() or not METADATA_PATH.is_file():
        st.error(
            "The PhishGuard model is not ready yet. In the project folder, run "
            "`python train_model.py`, then refresh this page."
        )
        st.stop()

    try:
        bundle, metadata = load_model_resources()
    except Exception:
        LOGGER.exception("Failed to load the local PhishGuard model artifacts.")
        st.error(
            "The saved model could not be loaded. Please rerun `python train_model.py` "
            "from the project folder, then refresh the app."
        )
        st.stop()

    with st.sidebar:
        st.markdown(
            '<div class="sidebar-brand"><span class="sidebar-brand-mark">&#128737;</span>'
            '<span class="sidebar-brand-name">PhishGuard</span></div>'
            '<p class="sidebar-subtitle">Phishing Protection Dashboard</p>',
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            ["URL Scanner", "Model Performance", "How It Works", "Security Tips"],
            label_visibility="collapsed",
        )
        st.divider()
        with st.expander("Model information", expanded=False):
            st.write(f"**Estimator:** {metadata.get('model_name', 'Unknown')}")
            st.write(f"**Training data:** {metadata.get('dataset', 'Unknown')}")
            st.write(f"**Rows used:** {metadata.get('experiment_rows', 'Unknown'):,}")
            st.write("**Labels:** 0 = Legitimate · 1 = Potential Phishing")
            st.write("**Probability:** Not supported by LinearSVC")

    if page == "URL Scanner":
        _ = render_scanner(bundle)
    elif page == "Model Performance":
        _ = render_performance(metadata)
        with st.expander("Model and feature details"):
            _ = render_model_information(metadata)
    elif page == "How It Works":
        _ = render_how_it_works()
        with st.expander("Model and feature details"):
            _ = render_model_information(metadata)
    else:
        _ = render_security_tips()
        with st.expander("Model and feature details"):
            _ = render_model_information(metadata)


if __name__ == "__main__":
    _ = main()
