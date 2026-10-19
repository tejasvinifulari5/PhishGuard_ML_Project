# PhishGuard — Intelligent Phishing URL Detection

PhishGuard is an educational Streamlit application that classifies a URL as legitimate-like or phishing-like using a local, saved machine-learning model. Submitted URLs are not opened and are not sent to external services.

## Features

- Local PhiUSIIL dataset use; no data download during app startup or model training.
- Character-level URL TF-IDF plus 22 lexical URL features.
- Stratified, reproducible train/test split with train-only fitted preprocessing.
- Linear Support Vector Classifier (LinearSVC).
- Streamlit dashboard with input validation, manual URL characteristics, and held-out test metrics.
- Model artifacts cached in memory by Streamlit; the application never retrains on page load.

## Dataset and labels

The source is the official UCI PhiUSIIL Phishing URL (Website) dataset (ID 967), stored locally as `data/raw/phiusiil.csv`. Training reads only its `URL` and `label` columns.

The local CSV has already been converted by `download_phiusiil.py`. The internal mapping is:

- `0` = Legitimate
- `1` = Potential Phishing

Neither the trainer nor the app remaps or inverts these labels. Do not use the dataset's other precomputed website/page columns for this URL-only experiment.
The verified CSV contains 235,795 rows: 134,850 legitimate and 100,945 potential phishing URLs.

If the CSV is not present, run `.\.venv\Scripts\python.exe download_phiusiil.py` from the project root. The script fetches UCI dataset ID 967 and saves the CSV to `data/raw/phiusiil.csv`; this first download requires network access. If the local CSV already exists, the script reuses it.

## Model and feature extraction

The model follows the feature pipeline in `PhishGuard_Intelligent_Phishing_URL_Detection.ipynb`:

- `TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=2, max_features=100000, dtype=float32, sublinear_tf=True)`.
- 22 deterministic lexical features, including URL/host length, character counts and ratios, punctuation counts, estimated subdomain count, HTTPS, IP-host and punycode indicators, port, path length, and query length.
- `MaxAbsScaler` fitted only on the training rows; sparse TF-IDF and lexical features are combined.
- `LinearSVC(C=1.0, max_iter=2000, random_state=42)`, selected from the notebook's evaluated models.

The notebook baseline compares Logistic Regression and Linear SVM using character TF-IDF plus all 22 lexical features. Its Random Forest uses only the 22 lexical features. Since the feature inputs differ, these results are not a controlled algorithm-only comparison.

LinearSVC does not provide calibrated class probabilities. PhishGuard therefore displays the predicted class but does not invent a probability or confidence score. Saved metrics are calculated from a held-out stratified test split and are not guarantees of real-world performance.

`phishguard_pipeline.py` is shared by training and inference so that lexical feature order, TF-IDF settings, scaling, and sparse feature combination remain consistent. The estimator and preprocessing objects are saved together in `models/phishing_model.joblib`; evaluation and run metadata are stored in `models/model_metadata.json`.

## Installation (Windows PowerShell)

Open PowerShell in the project root. Use the Python installation available in your environment. To create and activate a project virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies (including NumPy):

```powershell
python -m pip install -r requirements.txt
```

## Train the model

Use the default stratified 30,000-row experiment:

```powershell
python train_model.py
```

Use a different stratified sample size:

```powershell
python train_model.py --sample-size 60000
```

Use every row in the local CSV:

```powershell
python train_model.py --sample-size 0
```

Training writes `models/phishing_model.joblib` and `models/model_metadata.json`. It reads the existing CSV and does not alter it or the notebook.

## Run the Streamlit app

From the project folder, after training:

```powershell
python -m streamlit run app.py
```

Streamlit prints a local address to open in your browser. The app does not request the submitted URL; it only processes the string locally.

## Run the notebook

Open `PhishGuard_Intelligent_Phishing_URL_Detection.ipynb` in VS Code and select the project's `.venv` Python kernel. Run the notebook from the project root so its relative dataset and model paths resolve correctly. The model-comparison cell fits Logistic Regression, Linear SVM, and Random Forest for the notebook experiment. The final custom-URL cell instead loads the existing saved model bundle and can run without the notebook training cells.

The model-comparison cell's saved output includes accuracy, each model's feature inputs, confusion matrices, and a Linear SVM confusion-matrix plot. Rerunning that cell refits the three notebook comparison models but does not overwrite the saved deployment model. The custom URL example is an unverified model demonstration, not a ground-truth label.

## Validation

The saved metrics describe performance on a held-out split from the PhiUSIIL dataset; they are not a guarantee of detection accuracy on real-world, future, or previously unseen phishing URLs. Random splits may share hostnames across training and test data.

In the full CSV, 425 URL rows are duplicates among 235,795 rows (235,370 unique URL strings), with no URL having conflicting labels. In the baseline's seed-42 30,000-row sample split, two exact URL strings occur across train and test. The split also has 123 shared hostnames; 263 of 6,000 test URLs (4.38%) use a hostname present in training. These overlaps can make a random-split score optimistic. Near-duplicate URL analysis has not been performed.

To run the separate random, exact-URL-grouped, and hostname-grouped evaluation from the project root:

```powershell
python audit_group_split.py
```

This diagnostic fits models on its own sampled splits but does not replace or overwrite the saved model artifacts. Its group-aware scores use different test examples from the random split and should not be treated as a direct same-example comparison. Hostname grouping uses parsed hostnames, treats subdomains separately, and does not analyze near-duplicate URLs.

For a split-integrity check on the same sample and seed, the exact-URL-grouped partition had 0 shared exact URLs and 122 shared hostnames; the hostname-grouped partition had 0 shared hostnames and 0 shared exact URLs. Both had 24,000 training rows and 6,000 test rows, with 13,726/10,274 training labels and 3,431/2,569 test labels (legitimate/potential phishing). These are split checks only, not group-aware model scores.

The recorded baseline comparison uses the 6,000-row held-out test set:

| Model | Test features | Precision | Recall | F1-score | ROC-AUC |
|---|---|---:|---:|---:|---:|
| Logistic Regression | Character TF-IDF + 22 lexical | 1.0000 | 0.9833 | 0.9916 | 0.9990 |
| Linear SVM | Character TF-IDF + 22 lexical | 1.0000 | 0.9895 | 0.9947 | 0.9991 |
| Random Forest | 22 lexical only | 0.9969 | 0.9903 | 0.9936 | 0.9973 |

These are the notebook's recorded dataset-split results, not a guarantee of performance on unseen websites or future phishing attacks.

Run the lightweight URL pipeline tests from the project root:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## Troubleshooting

- **Dataset not found:** Confirm `data/raw/phiusiil.csv` exists and run commands from the project folder.
- **Model not found in the app:** Run `python train_model.py` first, then refresh Streamlit.
- **Model cannot be loaded:** Install the project's requirements in the same Python environment used to launch Streamlit, rerun training, and restart the app. Only load the locally generated trusted joblib artifact.
- **Missing Streamlit or package import:** Activate the intended environment and run `python -m pip install -r requirements.txt`.
- **PowerShell blocks environment activation:** You can skip activation and call the environment executable directly, for example `.\.venv\Scripts\python.exe -m pip install -r requirements.txt`.
- **A URL is rejected:** Supply a hostname using HTTP or HTTPS. If the scheme is omitted, PhishGuard normalizes it by adding `https://`.

## Known limitations

- This is an educational classifier, not a browser protection product or guarantee of safety.
- The separate `audit_group_split.py` diagnostic compares random, exact-URL-grouped, and hostname-grouped splits. These dataset-based results still do not guarantee performance on future domains or campaigns.
- URL-only features cannot inspect page contents, redirects, certificates, or live reputation.
- The displayed subdomain count is a simple heuristic and does not use a public suffix list.
- Manual URL characteristics are observations, not a per-prediction explanation of the model.
- No probability is available from the saved LinearSVC classifier.
