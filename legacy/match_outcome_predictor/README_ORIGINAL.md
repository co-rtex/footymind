# FootyMind ⚽  
Premier League Match Outcome Predictor

FootyMind is a fully end-to-end machine learning system that predicts **Premier League match outcomes** (Home Win / Draw / Away Win) using historical performance data and recent form.

It’s designed as a **portfolio-grade** project:

- Realistic **data pipeline** (raw stats → engineered features).
- Reproducible **ML training & evaluation**.
- **FastAPI** backend with a prediction endpoint.
- **Streamlit** UI for interactive exploration and explainability.
- Basic **pytest** tests for core components.

---

## 🚀 Features

- **Data pipeline (ETL)**  
  - Starts from a CSV of historical matches (`data/raw/sample_matches.csv`).
  - Cleans and validates raw data.
  - Builds rolling team features:
    - Recent average goals scored/conceded.
    - xG (expected goals).
    - Possession, shots, shots on target.
    - Cards, corners, and points per match.
  - Produces model-ready features + labels in `data/processed/train.csv`.
  - Also saves `data/processed/matches_with_features.csv` for API/UI use.

- **Modeling & evaluation**
  - Trains:
    - Multinomial **Logistic Regression** (with scaling).
    - **Gradient Boosting** classifier.
  - Compares models on a validation set and chooses the best by **log loss**.
  - Saves the best model + metadata with `joblib` to `models/footymind_model.joblib`.
  - Evaluation script prints:
    - Accuracy, log_loss, baseline accuracy.
    - Confusion matrix.
  - Saves:
    - `plots/confusion_matrix.png`
    - `plots/feature_importances.png`

- **Explainability**
  - Stores **feature importances** and **feature means** in the model artifact.
  - For each prediction:
    - Computes a simple contribution score per feature:
      \[
      \text{contribution}_i = (\text{value}_i - \text{mean}_i) \times \text{importance}_i
      \]
    - Returns top features with direction (positive / negative / neutral).

- **API (FastAPI)**
  - `GET /health` – health check.
  - `GET /matches` – list of matches with `match_id`, teams, date, outcome.
  - `POST /predict_by_match_id` – run prediction + explanation for a match.

- **UI (Streamlit)**
  - Filter matches by season, home team, away team.
  - Select a specific match via dropdown.
  - Displays:
    - Predicted outcome (from home team’s perspective).
    - Probability distribution (Home Win / Draw / Away Win).
    - Table of key contributing features.
    - Data table of filtered matches.

- **Testing**
  - Basic tests with `pytest`:
    - Data loading.
    - Feature engineering.
    - End-to-end ETL + training.

---

## 🧱 Tech Stack

- **Language:** Python 3.11 (tested)
- **Data & ML:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Backend:** `FastAPI`, `uvicorn`
- **UI:** `Streamlit`
- **Serialization:** `joblib`
- **Testing:** `pytest`

---

## 📁 Project Structure

```text
footymind/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ raw/
│  │  └─ sample_matches.csv               # sample dataset (~20 matches)
│  └─ processed/
│     ├─ train.csv                        # model-ready features + labels (generated)
│     └─ matches_with_features.csv        # enriched matches with features (generated)
├─ models/
│  └─ footymind_model.joblib              # trained model artifact (generated)
├─ plots/
│  ├─ confusion_matrix.png                # saved by evaluate_model.py
│  └─ feature_importances.png             # saved by evaluate_model.py
├─ src/
│  └─ footymind/
│     ├─ __init__.py
│     ├─ config.py
│     ├─ utils/
│     │  ├─ logging_utils.py
│     │  └─ paths.py
│     ├─ data/
│     │  └─ data_loader.py
│     ├─ features/
│     │  ├─ feature_builder.py
│     │  └─ etl_pipeline.py
│     ├─ models/
│     │  ├─ train_model.py
│     │  ├─ evaluate_model.py
│     │  └─ metrics.py
│     ├─ api/
│     │  └─ main.py
│     └─ ui/
│        └─ app.py
└─ tests/
   ├─ test_data_layer.py
   ├─ test_features.py
   └─ test_training.py
