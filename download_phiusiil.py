"""Load the UCI PhiUSIIL phishing URL dataset, caching it locally."""

from pathlib import Path

import pandas as pd
from ucimlrepo import DatasetNotFoundError, fetch_ucirepo


DATA_PATH = Path(__file__).resolve().parent / "data" / "raw" / "phiusiil.csv"
UCI_DATASET_ID = 967
UCI_TO_INTERNAL_LABEL = {
    0: 1,  # phishing
    1: 0,  # legitimate
}


def _validate_internal_labels(data: pd.DataFrame) -> None:
    if "label" not in data.columns:
        raise ValueError("PhiUSIIL CSV must contain a 'label' target column.")

    labels = set(data["label"].dropna().unique())
    if data["label"].isna().any() or labels != {0, 1}:
        raise ValueError(
            "PhiUSIIL labels must be non-null internal binary labels "
            "(1=phishing, 0=legitimate)."
        )


def load_phiusiil(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load cached PhiUSIIL data, or download, convert, and cache it once."""
    data_path = Path(data_path)
    if data_path.exists():
        data = pd.read_csv(data_path)
        _validate_internal_labels(data)
        return data

    try:
        dataset = fetch_ucirepo(id=UCI_DATASET_ID)
    except (ConnectionError, DatasetNotFoundError) as exc:
        raise RuntimeError(
            f"Could not download UCI dataset {UCI_DATASET_ID} (PhiUSIIL). "
            "Check your network connection and try again."
        ) from exc

    print("UCI dataset metadata:")
    print(dataset.metadata)

    target_variables = dataset.variables.loc[
        dataset.variables["role"].str.casefold().eq("target")
    ]
    if len(target_variables) != 1:
        raise ValueError(
            "Expected exactly one target in PhiUSIIL metadata; "
            f"found {len(target_variables)}."
        )

    target_name = target_variables.iloc[0]["name"]
    print("UCI target metadata:")
    print(target_variables[["name", "role", "description"]].to_string(index=False))

    targets = dataset.data.targets
    if targets is None or target_name not in targets.columns:
        raise ValueError(
            f"Target {target_name!r} from UCI metadata is missing from the data."
        )

    source_labels = pd.to_numeric(targets[target_name], errors="raise")
    observed_labels = set(source_labels.dropna().unique())
    if source_labels.isna().any() or observed_labels != set(UCI_TO_INTERNAL_LABEL):
        raise ValueError(
            "Expected UCI PhiUSIIL target labels {0, 1}; "
            f"observed {observed_labels}."
        )

    source_counts = source_labels.value_counts().sort_index().to_dict()
    expected_counts = {0: 100_945, 1: 134_850}
    if source_counts != expected_counts:
        raise ValueError(
            "UCI PhiUSIIL label counts do not match the official dataset "
            f"(expected {expected_counts}, observed {source_counts})."
        )

    print(f"Observed UCI target labels: {sorted(observed_labels)}")
    print(f"UCI target counts: {source_counts}")
    print(
        "Verified target mapping: UCI 1=legitimate -> internal 0; "
        "UCI 0=phishing -> internal 1."
    )

    features = dataset.data.features
    if features is None or len(features) != len(source_labels):
        raise ValueError("UCI PhiUSIIL features and target have different row counts.")

    data = features.copy()
    data["label"] = source_labels.map(UCI_TO_INTERNAL_LABEL).astype("int64")
    _validate_internal_labels(data)

    data_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_path, index=False)
    print(f"Saved {len(data)} rows to {data_path}")
    return data


if __name__ == "__main__":
    try:
        phiusiil = load_phiusiil()
        print(f"Loaded PhiUSIIL dataset: {phiusiil.shape[0]} rows, {phiusiil.shape[1]} columns")
    except (OSError, RuntimeError, ValueError) as exc:
        raise SystemExit(f"Failed to load PhiUSIIL dataset: {exc}") from exc
