"""Shared URL feature pipeline used by model training and the Streamlit app."""

from ipaddress import ip_address
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler


LABEL_NAMES = {0: "Legitimate", 1: "Potential Phishing"}
LEXICAL_FEATURE_COLUMNS = [
    "url_length",
    "host_length",
    "num_letters",
    "num_digits",
    "digit_ratio",
    "letter_ratio",
    "num_dots",
    "num_hyphens",
    "num_at_symbols",
    "num_slashes",
    "num_question_marks",
    "num_equals",
    "num_ampersands",
    "num_percent",
    "num_underscores",
    "num_subdomains",
    "has_https",
    "has_ip_address",
    "has_punycode",
    "has_port",
    "path_length",
    "query_length",
]


def normalize_url(url: object) -> str:
    """Preserve HTTP(S) URLs; add HTTPS to scheme-less URLs and reject malformed input."""
    if url is None or pd.isna(url):
        raise ValueError("Enter a URL to analyze.")

    value = str(url).strip()
    if not value:
        raise ValueError("Enter a URL to analyze.")
    if any(character.isspace() or ord(character) < 32 for character in value):
        raise ValueError("The URL cannot contain spaces or control characters.")

    if value.startswith("//"):
        value = f"https:{value}"
    elif "://" not in value:
        value = f"https://{value}"

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        _ = parsed.port
    except ValueError as error:
        raise ValueError(
            "That URL appears malformed. Check its hostname, brackets, or port."
        ) from error

    if parsed.scheme.casefold() not in {"http", "https"}:
        raise ValueError("Only HTTP and HTTPS URLs can be analyzed.")
    try:
        is_ip_literal = bool(hostname) and ip_address(hostname) is not None
    except ValueError:
        is_ip_literal = False
    if not hostname or ("." not in hostname and not is_ip_literal):
        raise ValueError("Enter a valid URL with a hostname, such as example.com.")
    if hostname.startswith(".") or hostname.endswith(".") or ".." in hostname:
        raise ValueError("Check the hostname; it appears malformed.")

    return value


def extract_lexical_features(url: object) -> dict[str, int | float]:
    """Compute the notebook's 22 deterministic lexical URL features."""
    text = normalize_url(url)
    parse_text = text if "://" in text else f"//{text}"
    try:
        parsed = urlsplit(parse_text)
        host = parsed.hostname or ""
    except ValueError:
        parsed = urlsplit("")
        host = ""

    try:
        is_ip_address = int(bool(host) and ip_address(host) is not None)
    except ValueError:
        is_ip_address = 0

    length = max(len(text), 1)
    letters = sum(character.isalpha() for character in text)
    digits = sum(character.isdigit() for character in text)
    host_parts = [part for part in host.split(".") if part]
    try:
        has_port = int(parsed.port is not None)
    except ValueError:
        has_port = 0

    return {
        "url_length": len(text),
        "host_length": len(host),
        "num_letters": letters,
        "num_digits": digits,
        "digit_ratio": digits / length,
        "letter_ratio": letters / length,
        "num_dots": text.count("."),
        "num_hyphens": text.count("-"),
        "num_at_symbols": text.count("@"),
        "num_slashes": text.count("/"),
        "num_question_marks": text.count("?"),
        "num_equals": text.count("="),
        "num_ampersands": text.count("&"),
        "num_percent": text.count("%"),
        "num_underscores": text.count("_"),
        "num_subdomains": max(len(host_parts) - 2, 0),
        "has_https": int(parsed.scheme.casefold() == "https"),
        "has_ip_address": is_ip_address,
        "has_punycode": int("xn--" in host.casefold()),
        "has_port": has_port,
        "path_length": len(parsed.path),
        "query_length": len(parsed.query),
    }


def build_lexical_frame(urls: list[str] | pd.Series) -> pd.DataFrame:
    """Return lexical features in the same stable order for train and predict."""
    frame = pd.DataFrame([extract_lexical_features(url) for url in urls])
    return frame.reindex(columns=LEXICAL_FEATURE_COLUMNS)


def create_vectorizer() -> TfidfVectorizer:
    """Create the character TF-IDF vectorizer used in the notebook."""
    return TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=100_000,
        dtype=np.float32,
        sublinear_tf=True,
    )


def fit_feature_pipeline(
    training_urls: list[str] | pd.Series,
) -> tuple[TfidfVectorizer, MaxAbsScaler, csr_matrix]:
    """Fit preprocessing on training URLs only and return its feature matrix."""
    vectorizer = create_vectorizer()
    url_text = [normalize_url(url) for url in training_urls]
    tfidf_matrix = vectorizer.fit_transform(url_text)

    lexical_frame = build_lexical_frame(url_text)
    scaler = MaxAbsScaler()
    lexical_matrix = csr_matrix(scaler.fit_transform(lexical_frame).astype(np.float32))
    combined = hstack([tfidf_matrix, lexical_matrix], format="csr")
    return vectorizer, scaler, combined


def transform_urls(
    urls: list[str] | pd.Series,
    vectorizer: TfidfVectorizer,
    scaler: MaxAbsScaler,
) -> csr_matrix:
    """Transform URLs with already-fitted training preprocessors."""
    url_text = [normalize_url(url) for url in urls]
    tfidf_matrix = vectorizer.transform(url_text)
    lexical_frame = build_lexical_frame(url_text)
    lexical_matrix = csr_matrix(scaler.transform(lexical_frame).astype(np.float32))
    return hstack([tfidf_matrix, lexical_matrix], format="csr")
