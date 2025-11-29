"""
Streamlit UI for FootyMind – Premier League Match Outcome Predictor.

Run from project root:

    streamlit run src/footymind/ui/app.py
"""

from __future__ import annotations

from typing import Any, Dict, List
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Ensure the project src/ directory is on sys.path so that:
#   from footymind.utils.paths import ...
# works when running via "streamlit run src/footymind/ui/app.py"
# from the project root.
# ---------------------------------------------------------------------------
SRC_ROOT = Path(__file__).resolve().parents[2]  # .../footymind/src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from footymind.utils.paths import get_model_path, get_processed_data_path  # noqa: E402


@st.cache_data(show_spinner=False)
def load_matches_with_features() -> pd.DataFrame:
    """
    Load the processed matches_with_features dataset.

    Requires that you have already run:
        python -m footymind.features.etl_pipeline
    """
    path = get_processed_data_path("matches_with_features.csv")
    df = pd.read_csv(path)
    # Ensure date is a string for display
    if "date" in df.columns:
        df["date"] = df["date"].astype(str)
    return df


@st.cache_resource(show_spinner=False)
def load_model_artifact() -> Dict[str, Any]:
    """
    Load the trained model artifact.

    Requires that you have already run:
        python -m footymind.models.train_model
    """
    path = get_model_path()
    artifact = joblib.load(path)
    return artifact


def build_explanation(
    feature_names: List[str],
    values: np.ndarray,
    feature_means: np.ndarray | None,
    feature_importances: np.ndarray | None,
    top_k: int = 8,
) -> List[Dict[str, Any]]:
    """
    Build a simple explanation by combining feature importance with deviation
    from mean:

        contribution_i = (value_i - mean_i) * importance_i

    Returns a list of up to top_k features, sorted by absolute contribution.
    """
    if feature_means is None or feature_importances is None:
        return []

    values = np.asarray(values, dtype=float)
    feature_means = np.asarray(feature_means, dtype=float)
    feature_importances = np.asarray(feature_importances, dtype=float)

    if (
        values.shape[0] != feature_means.shape[0]
        or values.shape[0] != feature_importances.shape[0]
    ):
        return []

    contributions = (values - feature_means) * feature_importances
    idx_sorted = np.argsort(np.abs(contributions))[::-1]
    top_k = min(top_k, len(idx_sorted))
    idx_top = idx_sorted[:top_k]

    explanation: List[Dict[str, Any]] = []
    for i in idx_top:
        contrib = float(contributions[i])
        explanation.append(
            {
                "feature_name": feature_names[i],
                "value": float(values[i]),
                "mean": float(feature_means[i]),
                "importance": float(feature_importances[i]),
                "contribution": contrib,
                "direction": "positive"
                if contrib > 0
                else "negative" if contrib < 0 else "neutral",
            }
        )

    return explanation


def predict_for_match(
    match_id: int,
    matches_df: pd.DataFrame,
    artifact: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run prediction + explanation for a single match_id, using the trained model.
    """
    mask = matches_df["match_id"] == match_id
    if not mask.any():
        raise ValueError(f"No match found with match_id={match_id}")

    row = matches_df.loc[mask].iloc[0]

    model = artifact["model"]
    scaler = artifact.get("scaler")
    feature_names: List[str] = artifact["feature_names"]
    class_labels: List[str] = artifact["class_labels"]
    feature_means = artifact.get("feature_means")
    feature_importances = artifact.get("feature_importances")

    x_values = row[feature_names].astype(float).values
    X = np.asarray([x_values], dtype=float)

    if scaler is not None:
        X_input = scaler.transform(X)
    else:
        X_input = X

    proba = model.predict_proba(X_input)[0]
    pred_idx = int(np.argmax(proba))
    pred_label = class_labels[pred_idx]
    prob_dict = {label: float(p) for label, p in zip(class_labels, proba)}

    explanation = build_explanation(
        feature_names=feature_names,
        values=x_values,
        feature_means=feature_means,
        feature_importances=feature_importances,
        top_k=8,
    )

    result: Dict[str, Any] = {
        "match_id": int(row["match_id"]),
        "home_team": row.get("home_team"),
        "away_team": row.get("away_team"),
        "season": row.get("season"),
        "date": str(row.get("date")),
        "predicted_class": pred_label,
        "class_probabilities": prob_dict,
        "explanation": explanation,
    }
    return result


def render_past_match_mode(matches_df: pd.DataFrame, artifact: Dict[str, Any]) -> None:
    """
    UI mode for exploring and predicting on past matches directly.
    """
    st.sidebar.header("Match selection (past data)")

    seasons = sorted(matches_df["season"].dropna().unique().tolist())
    home_teams = sorted(matches_df["home_team"].dropna().unique().tolist())
    away_teams = sorted(matches_df["away_team"].dropna().unique().tolist())

    selected_season = st.sidebar.selectbox(
        "Season (optional filter)", options=["All"] + seasons, index=0
    )

    selected_home_team = st.sidebar.selectbox(
        "Home team (optional filter)", options=["All"] + home_teams, index=0
    )

    selected_away_team = st.sidebar.selectbox(
        "Away team (optional filter)", options=["All"] + away_teams, index=0
    )

    df_filtered = matches_df.copy()
    if selected_season != "All":
        df_filtered = df_filtered[df_filtered["season"] == selected_season]
    if selected_home_team != "All":
        df_filtered = df_filtered[df_filtered["home_team"]
                                  == selected_home_team]
    if selected_away_team != "All":
        df_filtered = df_filtered[df_filtered["away_team"]
                                  == selected_away_team]

    if df_filtered.empty:
        st.warning("No matches found for the selected filters.")
        return

    # Build nice labels for each match
    df_filtered = df_filtered.sort_values("date")
    options = []
    for _, r in df_filtered.iterrows():
        label = (
            f"{r['date']} – {r['home_team']} vs {r['away_team']} "
            f"({r['season']}, actual: {r['outcome']}, id={r['match_id']})"
        )
        options.append((label, int(r["match_id"])))

    option_labels = [lbl for lbl, _ in options]
    label_to_id = {lbl: mid for lbl, mid in options}

    selected_label = st.selectbox(
        "Select a match", options=option_labels, index=len(option_labels) - 1
    )
    selected_match_id = label_to_id[selected_label]

    st.markdown("---")

    col_left, col_right = st.columns([2, 3])

    with col_left:
        st.subheader("Selected match")
        row = matches_df[matches_df["match_id"] == selected_match_id].iloc[0]
        st.write(
            f"**{row['home_team']}** vs **{row['away_team']}** "
            f"({row['season']}, {row['date']})"
        )
        st.write(f"Actual outcome: `{row['outcome']}`")
        st.caption(f"Match ID: {selected_match_id}")

        if st.button("🔮 Predict outcome", key="predict_past"):
            try:
                result = predict_for_match(
                    selected_match_id, matches_df, artifact
                )
            except Exception as e:  # noqa: BLE001
                st.error(f"Prediction failed: {e}")
            else:
                st.subheader("Model prediction")
                st.write(
                    f"**Predicted outcome:** `{result['predicted_class']}`")

                probs = result["class_probabilities"]
                prob_df = pd.DataFrame(
                    {
                        "Outcome": list(probs.keys()),
                        "Probability": list(probs.values()),
                    }
                ).set_index("Outcome")
                st.bar_chart(prob_df)

                with col_right:
                    st.subheader("Key contributing features")
                    explanation = result["explanation"]
                    if not explanation:
                        st.write(
                            "No explanation available (feature importances not present)."
                        )
                    else:
                        exp_df = pd.DataFrame(explanation)
                        exp_df["abs_contribution"] = exp_df[
                            "contribution"
                        ].abs()
                        exp_df = exp_df.sort_values(
                            "abs_contribution", ascending=False
                        )
                        st.dataframe(
                            exp_df[
                                [
                                    "feature_name",
                                    "value",
                                    "mean",
                                    "importance",
                                    "contribution",
                                    "direction",
                                ]
                            ],
                            use_container_width=True,
                        )

    with col_right:
        st.subheader("Matches in dataset (filtered)")
        st.dataframe(
            df_filtered[
                [
                    "match_id",
                    "date",
                    "home_team",
                    "away_team",
                    "season",
                    "outcome",
                ]
            ],
            use_container_width=True,
            height=400,
        )


def render_quick_team_mode(matches_df: pd.DataFrame, artifact: Dict[str, Any]) -> None:
    """
    UI mode for quickly predicting by teams.

    Under the hood, this finds the most recent historical match between the
    selected home and away team in the dataset, and runs the prediction on
    that match's feature vector.
    """
    st.sidebar.header("Quick prediction by teams")

    home_teams = sorted(matches_df["home_team"].dropna().unique().tolist())
    away_teams = sorted(matches_df["away_team"].dropna().unique().tolist())

    selected_home = st.sidebar.selectbox(
        "Home team", options=home_teams, index=0)
    selected_away = st.sidebar.selectbox(
        "Away team", options=away_teams, index=0)

    if selected_home == selected_away:
        st.warning("Home and away team must be different.")
        return

    st.markdown("---")
    st.subheader("Simulated fixture (based on latest historical matchup)")

    # Find all historical matches between these teams (home/away in this order)
    df_pair = matches_df[
        (matches_df["home_team"] == selected_home)
        & (matches_df["away_team"] == selected_away)
    ].copy()

    if df_pair.empty:
        st.info(
            "No historical matches found between these teams in this dataset "
            "with this home/away configuration."
        )
        return

    df_pair = df_pair.sort_values("date")
    latest_row = df_pair.iloc[-1]
    latest_match_id = int(latest_row["match_id"])

    st.write(
        f"Using latest recorded match between **{selected_home}** and "
        f"**{selected_away}** in the dataset:"
    )
    st.write(
        f"- Date: `{latest_row['date']}`  "
        f"- Season: `{latest_row['season']}`  "
        f"- Actual outcome: `{latest_row['outcome']}`  "
        f"- Match ID: `{latest_match_id}`"
    )

    if st.button("🔮 Predict for this pairing", key="predict_quick"):
        try:
            result = predict_for_match(latest_match_id, matches_df, artifact)
        except Exception as e:  # noqa: BLE001
            st.error(f"Prediction failed: {e}")
        else:
            st.subheader("Model prediction")
            st.write(
                f"If these teams played under **similar conditions**, "
                f"the model predicts: **{result['predicted_class']}**"
            )

            probs = result["class_probabilities"]
            prob_df = pd.DataFrame(
                {
                    "Outcome": list(probs.keys()),
                    "Probability": list(probs.values()),
                }
            ).set_index("Outcome")
            st.bar_chart(prob_df)

            explanation = result["explanation"]
            st.subheader("Key contributing features")
            if not explanation:
                st.write(
                    "No explanation available (feature importances not present)."
                )
            else:
                exp_df = pd.DataFrame(explanation)
                exp_df["abs_contribution"] = exp_df["contribution"].abs()
                exp_df = exp_df.sort_values(
                    "abs_contribution", ascending=False
                )
                st.dataframe(
                    exp_df[
                        [
                            "feature_name",
                            "value",
                            "mean",
                            "importance",
                            "contribution",
                            "direction",
                        ]
                    ],
                    use_container_width=True,
                )


def main() -> None:
    st.set_page_config(
        page_title="FootyMind – Premier League Match Predictor",
        layout="wide",
    )

    st.title("⚽ FootyMind – Premier League Match Outcome Predictor")

    st.markdown(
        """
Use FootyMind in two ways:

1. **Explore past matches** – pick a specific match from the dataset and see what the model predicts.
2. **Quick prediction by teams** – select home & away teams and get a prediction based on their latest historical matchup in the data.
"""
    )

    # Load data and model
    try:
        matches_df = load_matches_with_features()
    except FileNotFoundError:
        st.error(
            "Could not find `data/processed/matches_with_features.csv`.\n\n"
            "Run the ETL pipeline first:\n\n"
            "```bash\n"
            "cd src\n"
            "python -m footymind.features.etl_pipeline\n"
            "```"
        )
        return

    try:
        artifact = load_model_artifact()
    except FileNotFoundError:
        st.error(
            "Could not find trained model artifact `models/footymind_model.joblib`.\n\n"
            "Train the model first:\n\n"
            "```bash\n"
            "cd src\n"
            "python -m footymind.models.train_model\n"
            "```"
        )
        return

    mode = st.radio(
        "Mode",
        options=["Explore past matches", "Quick prediction by teams"],
        horizontal=True,
    )

    if mode == "Explore past matches":
        render_past_match_mode(matches_df, artifact)
    else:
        render_quick_team_mode(matches_df, artifact)


if __name__ == "__main__":
    main()
