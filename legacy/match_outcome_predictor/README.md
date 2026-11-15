# Legacy: FootyMind v1 match-outcome predictor (archived)

This directory preserves FootyMind v1 (Nov 2025): a Premier League
home-win / draw / away-win classifier with a FastAPI endpoint and a Streamlit
UI. It is kept as a reproducible historical experiment. It is **not** part of
the FootyMind 2.0 product and has no dependencies on, or from, the `footymind`
package at the repository root.

The original README is preserved verbatim in `README_ORIGINAL.md`.

## How to run (historical reproduction)

```bash
cd legacy/match_outcome_predictor
python -m venv .venv && . .venv/bin/activate
pip install pandas numpy scikit-learn joblib fastapi pytest matplotlib seaborn
PYTHONPATH=. pytest -q tests
```

Running the tests regenerates `data/processed/`, `models/` and `plots/` locally.
These outputs are git-ignored.

Changes from the original code (made during the 2.0 migration, Nov 2026):

* The package was renamed `footymind` → `footymind_legacy`, and imports were updated accordingly.
* `api/__initII.py` was renamed to `api/__init__.py`.
* `LogisticRegression(multi_class="multinomial")` was removed. The argument was
  dropped in recent scikit-learn, and multinomial is now the default.
* `config.PROJECT_ROOT` now points at this directory.
* Generated artifacts (`models/*.joblib`, `data/processed/*.csv`, `plots/*.png`),
  `__pycache__` and `.DS_Store` files are no longer tracked. They remain
  recoverable from git history at commit `f08f147`.

## Audit (2026-11-13)

Verified by reading the code and running it, not by trusting the README.

| Finding | Evidence | Severity |
|---|---|---|
| Data is synthetic. Team names are real, but the 20 rows are invented ("synthetic but realistic", `docs/datasets.md`). | `data/raw/sample_matches.csv`, 20 rows | Blocks any real-world claim |
| Only 9 rows survive feature building: 8 home wins, 1 away win, 0 draws. | `train.csv` shape `(9, 27)` | No generalization evidence |
| Validation is a random `train_test_split` with 25% held out (about 3 rows) on time-ordered matches. Model selection uses the same 3 rows. | `train_model.py` | Metrics are meaningless |
| A 3-class model is trained on data containing no draws, so the draw probability is never learned. | class counts | Invalid probabilities |
| The "explanation" `(value - mean) * importance` is computed with means over train+val. It is not a valid attribution method (no sign for tree importances, no interaction handling). | `train_model.py`, `explain_model.py` | Misleading explanations |
| The API and UI call `joblib.load` on a pickle. Unpickling an untrusted file executes arbitrary code. | `api/main.py`, `ui/app.py` | Security (if artifacts are untrusted) |
| The end-to-end test overwrote the committed model artifact. | `tests/test_training.py` | Reproducibility |
| Rolling form features use `shift(1)` before `rolling`, so no same-match leakage. | `feature_builder.py` | OK: retained idea |
| `requirements.txt` pins (numpy 1.26, pandas 2.2) do not install on Python 3.13. On current scikit-learn, training failed with `TypeError: multi_class`. | baseline test run: 3 passed, 1 failed | Fixed minimally |

### Component classification

| Component | Decision | Reason |
|---|---|---|
| Rolling pre-match form features | ARCHIVE | Correct leakage-avoidance pattern, but not relevant to tracking-based research |
| ETL/train/serve layout | ARCHIVE (pattern informs 2.0) | 2.0 uses typed configs and versioned manifests instead |
| Synthetic sample CSV | ARCHIVE, clearly labeled synthetic | Must never be presented as observed data |
| joblib model artifact | REMOVE from tracking | Generated, unsafe to load from untrusted sources |
| Streamlit UI | ARCHIVE | Replaced by the tactical replay frontend |
| FastAPI `/predict_by_match_id` | ARCHIVE | The 2.0 API is tracking-centric |
| Feature-contribution explainer | ARCHIVE, do not reuse | Not a valid attribution |
| Scraper example | ARCHIVE | Unused, and its target site terms were never assessed |
