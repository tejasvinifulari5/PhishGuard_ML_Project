"""Compare random and group-aware URL phishing evaluations.

This diagnostic reproduces the notebook's sampled random split and compares
it with exact-URL-grouped and hostname-grouped splits. It does not save models
or other artifacts.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.svm import LinearSVC

from phishguard_pipeline import (
    LABEL_NAMES,
    build_lexical_frame,
    fit_feature_pipeline,
    normalize_url,
    transform_urls,
)


RANDOM_STATE = 42
SAMPLE_SIZE = 30_000
TEST_SIZE = 0.20
DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "phiusiil.csv"


def normalize_url_for_grouping(url: str) -> str:
    """Normalize a URL consistently before grouping identical URL strings."""
    return normalize_url(url).casefold()


def extract_hostname(url: str) -> str | None:
    """Parse hostname with or without scheme; return normalized hostname."""
    try:
        parsed = urlsplit(normalize_url(url))
        hostname = parsed.hostname
    except ValueError:
        return None
    if not hostname:
        return None
    return hostname.rstrip(".").casefold()


def choose_group_split(
    urls: pd.Series, labels: pd.Series, groups: np.ndarray, split_name: str
) -> tuple[np.ndarray, np.ndarray]:
    """Choose the SGKF fold closest to the target size and class balance."""
    splitter = StratifiedGroupKFold(
        n_splits=5, shuffle=True, random_state=RANDOM_STATE
    )
    candidates = list(splitter.split(urls, labels, groups))
    target_positive_rate = float(labels.mean())

    def score(fold: tuple[np.ndarray, np.ndarray]) -> tuple[float, float, int]:
        train_indices, test_indices = fold
        test_labels = labels.iloc[test_indices]
        size_error = abs(len(test_indices) / len(labels) - TEST_SIZE)
        balance_error = abs(float(test_labels.mean()) - target_positive_rate)
        return size_error, balance_error, int(test_indices.min())

    if not candidates:
        raise RuntimeError(f"{split_name}: StratifiedGroupKFold produced no splits.")
    train_indices, test_indices = min(candidates, key=score)
    if set(labels.iloc[train_indices].unique()) != {0, 1}:
        raise RuntimeError(f"{split_name}: training partition does not contain both classes.")
    if set(labels.iloc[test_indices].unique()) != {0, 1}:
        raise RuntimeError(f"{split_name}: test partition does not contain both classes.")
    return train_indices, test_indices


def make_splits(
    urls: pd.Series, labels: pd.Series
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    random_train, random_test = train_test_split(
        np.arange(len(urls)),
        test_size=TEST_SIZE,
        stratify=labels,
        random_state=RANDOM_STATE,
    )
    exact_groups = urls.map(normalize_url_for_grouping).to_numpy()

    hostname_values = urls.map(extract_hostname)
    # Unparseable hostnames remain separate groups instead of being joined.
    hostname_groups = np.array(
        [
            f"host:{hostname}" if hostname is not None else f"unparseable-row:{index}"
            for index, hostname in enumerate(hostname_values)
        ],
        dtype=object,
    )
    exact_train, exact_test = choose_group_split(
        urls, labels, exact_groups, "Exact-URL grouped split"
    )
    host_train, host_test = choose_group_split(
        urls, labels, hostname_groups, "Hostname grouped split"
    )
    return {
        "Random": (random_train, random_test),
        "Exact-URL grouped": (exact_train, exact_test),
        "Hostname grouped": (host_train, host_test),
    }


def describe_split(
    name: str,
    urls: pd.Series,
    labels: pd.Series,
    indices: tuple[np.ndarray, np.ndarray],
) -> dict[str, object]:
    train_indices, test_indices = indices
    train_urls = urls.iloc[train_indices]
    test_urls = urls.iloc[test_indices]
    train_labels = labels.iloc[train_indices]
    test_labels = labels.iloc[test_indices]

    train_exact = set(train_urls.map(normalize_url_for_grouping))
    test_exact = set(test_urls.map(normalize_url_for_grouping))
    exact_overlap = train_exact & test_exact
    train_hosts = set(train_urls.map(extract_hostname).dropna())
    test_host_series = test_urls.map(extract_hostname)
    test_hosts = set(test_host_series.dropna())
    host_overlap = train_hosts & test_hosts
    test_seen_host_rows = int(test_host_series.isin(train_hosts).sum())
    hostname_parse_failures = int(test_host_series.isna().sum())

    exact_ok = not exact_overlap
    host_ok = not host_overlap
    if name == "Exact-URL grouped":
        intended_constraint = f"exact-URL={exact_ok}; hostname=not enforced"
    elif name == "Hostname grouped":
        intended_constraint = f"hostname={host_ok}; exact-URL={exact_ok}"
    else:
        intended_constraint = "none (random split baseline)"
    print(f"\n[{name}]")
    print(
        f"Rows: train={len(train_indices):,}, test={len(test_indices):,}; "
        f"class counts train={dict(Counter(train_labels))}, "
        f"test={dict(Counter(test_labels))}"
    )
    print(
        f"Class proportions train="
        f"{train_labels.value_counts(normalize=True).sort_index().round(4).to_dict()}, "
        f"test={test_labels.value_counts(normalize=True).sort_index().round(4).to_dict()}"
    )
    print(
        f"Exact URL overlap={len(exact_overlap):,}; "
        f"hostname overlap={len(host_overlap):,}; "
        f"test rows on train-seen hostnames={test_seen_host_rows:,}/"
        f"{len(test_indices):,} ({test_seen_host_rows / len(test_indices):.2%}); "
        f"unparseable test hostnames={hostname_parse_failures:,}"
    )
    print(f"Intended grouping constraint: {intended_constraint}")
    return {
        "train_indices": train_indices,
        "test_indices": test_indices,
        "exact_url_overlap": len(exact_overlap),
        "hostname_overlap": len(host_overlap),
        "exact_group_constraint_met": exact_ok,
        "hostname_group_constraint_met": host_ok,
    }


def build_features(
    train_urls: pd.Series, test_urls: pd.Series
) -> tuple[object, object, pd.DataFrame, pd.DataFrame]:
    """Fit TF-IDF and scaling on training rows only."""
    vectorizer, scaler, train_combined = fit_feature_pipeline(train_urls)
    test_combined = transform_urls(test_urls, vectorizer, scaler)
    train_lexical = build_lexical_frame(train_urls)
    test_lexical = build_lexical_frame(test_urls)
    return (
        train_combined,
        test_combined,
        train_lexical,
        test_lexical,
    )


def make_models() -> dict[str, object]:
    """Return the notebook's predeclared model configurations."""
    return {
        "Logistic Regression": LogisticRegression(
            C=2.0,
            solver="liblinear",
            max_iter=500,
            random_state=RANDOM_STATE,
        ),
        "Linear SVM": LinearSVC(
            C=1.0, max_iter=2_000, random_state=RANDOM_STATE
        ),
        "Random Forest (lexical only)": RandomForestClassifier(
            n_estimators=120,
            max_depth=18,
            min_samples_leaf=2,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
    }


def evaluate_split(
    split_name: str,
    urls: pd.Series,
    labels: pd.Series,
    split_info: dict[str, object],
) -> list[dict[str, object]]:
    train_indices = split_info["train_indices"]
    test_indices = split_info["test_indices"]
    train_urls = urls.iloc[train_indices].reset_index(drop=True)
    test_urls = urls.iloc[test_indices].reset_index(drop=True)
    train_labels = labels.iloc[train_indices].reset_index(drop=True)
    test_labels = labels.iloc[test_indices].reset_index(drop=True)

    train_combined, test_combined, train_lexical, test_lexical = build_features(
        train_urls, test_urls
    )
    result_rows = []
    for model_name, model in make_models().items():
        if model_name == "Random Forest (lexical only)":
            train_matrix, test_matrix = train_lexical, test_lexical
        else:
            train_matrix, test_matrix = train_combined, test_combined
        model.fit(train_matrix, train_labels)
        predictions = model.predict(test_matrix)
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(test_matrix)[:, 1]
            score_type = "probability"
        else:
            scores = model.decision_function(test_matrix)
            score_type = "decision score"

        matrix = confusion_matrix(test_labels, predictions, labels=[0, 1])
        tn, fp, fn, tp = matrix.ravel()
        row = {
            "split": split_name,
            "model": model_name,
            "n_test": len(test_labels),
            "accuracy": accuracy_score(test_labels, predictions),
            "precision": precision_score(
                test_labels, predictions, pos_label=1, zero_division=0
            ),
            "recall": recall_score(
                test_labels, predictions, pos_label=1, zero_division=0
            ),
            "f1": f1_score(test_labels, predictions, pos_label=1, zero_division=0),
            "roc_auc": (
                roc_auc_score(test_labels, scores)
                if test_labels.nunique() == 2
                else np.nan
            ),
            "score_type": score_type,
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }
        result_rows.append(row)
        print(
            f"  {model_name}: accuracy={row['accuracy']:.4f} "
            f"({int(round(row['accuracy'] * len(test_labels)))}/{len(test_labels)}), "
            f"precision={row['precision']:.4f}, recall={row['recall']:.4f}, "
            f"F1={row['f1']:.4f}, ROC-AUC={row['roc_auc']:.4f} "
            f"({score_type})"
        )
        print(
            "    Confusion matrix (rows=actual [Legitimate (0), Potential Phishing (1)]; "
            "columns=predicted [Legitimate (0), Potential Phishing (1)]): "
            f"[[TN={tn}, FP={fp}], [FN={fn}, TP={tp}]]"
        )
    return result_rows


def main() -> None:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, usecols=["URL", "label"])
    if df[["URL", "label"]].isna().any().any():
        raise ValueError("Dataset contains missing URL or label values.")
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype("int8")
    if set(df["label"].unique()) != {0, 1}:
        raise ValueError(f"Expected internal labels {{0, 1}}, got {set(df['label'].unique())}.")

    print(f"Dataset: {DATA_PATH}")
    print(f"Rows={len(df):,}; labels={df['label'].value_counts().sort_index().to_dict()}")
    print(
        f"Label mapping: 0={LABEL_NAMES[0]}, 1={LABEL_NAMES[1]}; "
        "labels are not remapped."
    )
    print(
        "Configurations copied from notebook: sample=30,000, seed=42, "
        "test fraction=0.20, char TF-IDF (3,5), 100,000 max features, "
        "22 lexical features, LogisticRegression C=2, LinearSVC C=1, "
        "RandomForest 120 trees/max_depth=18."
    )
    experiment_df, _ = train_test_split(
        df,
        train_size=SAMPLE_SIZE,
        stratify=df["label"],
        random_state=RANDOM_STATE,
    )
    urls = experiment_df["URL"].astype(str).reset_index(drop=True)
    labels = experiment_df["label"].reset_index(drop=True)
    print(f"Sample rows={len(urls):,}; sample labels={labels.value_counts().sort_index().to_dict()}")

    splits = make_splits(urls, labels)
    integrity = {}
    all_results = []
    for split_name, indices in splits.items():
        info = describe_split(split_name, urls, labels, indices)
        integrity[split_name] = info
        all_results.extend(evaluate_split(split_name, urls, labels, info))

    result_df = pd.DataFrame(all_results)
    print("\nCOMPARISON TABLE (all metrics on each split's held-out test partition)")
    display_columns = [
        "split",
        "model",
        "n_test",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "score_type",
    ]
    print(
        result_df[display_columns].to_string(
            index=False,
            formatters={
                "accuracy": "{:.4f}".format,
                "precision": "{:.4f}".format,
                "recall": "{:.4f}".format,
                "f1": "{:.4f}".format,
                "roc_auc": "{:.4f}".format,
            },
        )
    )

    print("\nLIMITATIONS")
    print(
        "- Group splits use the same stratified 30,000-row sample but different "
        "partition composition; compare them as separate generalization diagnostics, "
        "not identical test sets."
    )
    print(
        "- For each group-aware split, the selected fold is the one closest to 20% "
        "test size, then closest to the sample phishing rate; this selection uses "
        "only group assignments and labels, not model metrics."
    )
    print(
        "- Hostnames are parsed syntactically from URL strings. Subdomains are treated "
        "as distinct hostnames; this is not registrable-domain grouping."
    )
    print(
        "- No hyperparameter tuning or model selection is performed. Every split uses "
        "the same model configurations; all learned preprocessing is fit on that "
        "split's training partition only."
    )


if __name__ == "__main__":
    main()
