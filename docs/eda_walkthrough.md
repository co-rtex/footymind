# FootyMind EDA & Demo Walkthrough

This guide explains how to use the notebook-style script:

- `notebooks/footymind_eda_and_demo.py`

to explore the FootyMind data and model.

---

## 1. Requirements

Make sure you have already:

1. Created and activated the virtual environment.
2. Installed dependencies.
3. Run the ETL pipeline and trained a model at least once:

```bash
cd /path/to/footymind
source .venv/bin/activate

cd src
python -m footymind.features.etl_pipeline
python -m footymind.models.train_model
