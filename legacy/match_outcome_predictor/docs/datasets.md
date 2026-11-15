# FootyMind Datasets Guide

FootyMind ships with a small demo dataset:

- `data/raw/sample_matches.csv` (~20 matches, synthetic but realistic).

This is enough to run the full pipeline, but for more serious modeling you
will want a **larger real dataset** of Premier League matches.

This guide explains how to plug in a bigger dataset while keeping the existing
code and features.

---

## 1. Expected schema

FootyMind assumes a tabular CSV with **one row per match** and at least the
following columns:

- `date` – match date (e.g., `2023-08-19`)
- `season` – season label (e.g., `2023-2024`)
- `home_team` – home team name (string)
- `away_team` – away team name (string)
- `home_goals` – integer goals scored by home team
- `away_goals` – integer goals scored by away team

Optional but recommended continuous stats:

- `home_xG`, `away_xG` – expected goals
- `home_possession`, `away_possession` – percent possession (0–100)
- `home_shots`, `away_shots`
- `home_shots_on_target`, `away_shots_on_target`
- `home_corners`, `away_corners`
- `home_yellow_cards`, `away_yellow_cards`
- `home_red_cards`, `away_red_cards`

If your dataset uses different names (e.g., `HomeTeam`, `FTHG`, `FTAG` from
some public sources), you can:

- Either rename the columns in the CSV itself, or
- Add a small pre-processing script to map them to the expected names.

The current ETL and feature builder scripts expect the **FootyMind schema**.

---

## 2. Replacing `sample_matches.csv` with a larger dataset

1. Find or prepare a Premier League dataset with at least the columns listed
   above. Common sources include:
   - Open football statistics sites
   - Kaggle datasets for the Premier League

2. Make a backup of the demo file (optional):

   ```bash
   cd /path/to/footymind
   mv data/raw/sample_matches.csv data/raw/sample_matches_demo_backup.csv

3. Save your larger dataset as:
   data/raw/sample_matches.csv
   Make sure it is UTF-8 encoded and uses a comma (,) as separator
   (or adjust data_loader.py accordingly).

4. Check the header row to ensure it matches the expected schema. If not,
   you can quickly fix it in Python, for example:
   import pandas as pd
   
   df = pd.read_csv("your_raw_file.csv")
   df = df.rename(
      columns={
         "Date": "date",
         "Season": "season",
         "HomeTeam": "home_team",
         "AwayTeam": "away_team",
         "FTHG": "home_goals",
         "FTAG": "away_goals",
     }
   )

   df.to_csv("data/raw/sample_matches.csv", index=False)
