#!/usr/bin/env bash
# Rebuild every committed research artefact from raw data, in dependency order.
#   scripts/reproduce.sh            # run everything (~2 GB download; several hours)
#   scripts/reproduce.sh --dry-run  # print the steps only
#   PROVIDERS=metrica scripts/reproduce.sh   # Metrica only (~180 MB; 1-2 hours)
# (The recorded run times of the committed results sum to about 1.6 h for Metrica and 4.4 h for
# SkillCorner on 4 vCPU, partly under load.)
# Each results.json records the code revision it was produced at; run from a clean tree.
set -euo pipefail
cd "$(dirname "$0")/.."
DRY=0; [[ "${1:-}" == "--dry-run" ]] && DRY=1
PROVIDERS="${PROVIDERS:-metrica skillcorner}"
run() { echo "+ $*"; [[ $DRY -eq 1 ]] || "$@"; }

if [[ $DRY -eq 0 && -n "$(git status --porcelain -- src)" ]]; then
  echo "refusing: src/ has uncommitted changes (results would record a dirty revision)" >&2
  exit 1
fi

ids() { uv run python -c "from footymind.data.sources import SOURCES; print(' '.join(m for m in SOURCES if m.startswith('$1-')))"; }
for p in $PROVIDERS; do
  run uv run footymind download $(ids "$p")
  run uv run footymind ingest $(ids "$p")
done

# Metrica experiments (configs pin dataset rules; v1/v2 configs use rules v1).
for c in pass_outcome_v1 pass_outcome_v1_ablation pass_outcome_v1_sens_no_ball_out \
         pass_outcome_v1_sens_full_release pass_outcome_v2_3match pass_outcome_v2_sens_no_after_carry \
         option_model_v2_lead pass_outcome_v3; do
  run uv run footymind experiment "experiments/configs/$c.json"
done
run uv run python -m footymind.research.failure_modes pass_outcome_v3
run uv run python -m footymind.research.guard_calibration
for c in temporal_v1 temporal_guard_v1_02s temporal_guard_v1_04s option_model_guard_v1; do
  run uv run footymind experiment "experiments/configs/$c.json"
done
run uv run python -m footymind.research.temporal_strata temporal_v1
run uv run python -m footymind.research.calibrate_physics
# The packaged option model is the value model of the Defensive Shadow, run-value and
# movement-decomposition studies below, so it is trained before them.
run uv run footymind train-option-model --provider metrica
run uv run python -m footymind.research.shadow_stability
run uv run python -m footymind.research.shadow_validation metrica
run uv run python -m footymind.research.run_value_validation metrica
run uv run python -m footymind.research.movement_decomposition_study metrica
run uv run python -m footymind.research.rule_response_check metrica 150
run uv run python -m footymind.research.similarity_validation metrica
# Defender response forecasts (validation first: the trainer copies its verdict into the model).
run uv run python -m footymind.research.response_validation metrica
run uv run footymind train-response-model --provider metrica
run uv run python -m footymind.research.response_sensitivity metrica
# Session 4: envelope v2 (pre-registered) and the innovation response model.
run uv run python -m footymind.research.envelope_validation metrica
run uv run python -m footymind.research.response_diagnostics metrica
run uv run python -m footymind.research.train_response_model innovation metrica
run uv run python -m footymind.research.response_sensitivity metrica 300 2
run uv run python -m footymind.research.rule_response_check metrica 150 3
run uv run python -m footymind.research.rule_response_check metrica 150 4
run uv run python -m footymind.research.offball_study metrica

if [[ " $PROVIDERS " == *" skillcorner "* ]]; then
  for c in pass_outcome_skillcorner_v2 pass_outcome_skillcorner_v2_importance \
           temporal_skillcorner_v1 temporal_guard_skillcorner_v1_02s \
           temporal_guard_skillcorner_v2_ballcensor option_model_guard_skillcorner_v1; do
    run uv run footymind experiment "experiments/configs/$c.json"
  done
  run uv run python -m footymind.research.temporal_strata temporal_skillcorner_v1
  run uv run python -m footymind.research.temporal_strata temporal_guard_skillcorner_v1_02s --match-level-only
  run uv run python -m footymind.research.temporal_strata temporal_guard_skillcorner_v2_ballcensor --match-level-only
  run uv run python -m footymind.research.ball_censor_check skillcorner
  run uv run footymind train-option-model --provider skillcorner
  run uv run python -m footymind.research.shadow_validation skillcorner
  run uv run python -m footymind.research.data_quality_audit
  run uv run python -m footymind.research.run_value_validation skillcorner
  run uv run python -m footymind.research.movement_decomposition_study skillcorner
  run uv run python -m footymind.research.rule_response_check skillcorner 60
  run uv run python -m footymind.research.similarity_validation skillcorner
  run uv run python -m footymind.research.response_validation skillcorner
  run uv run footymind train-response-model --provider skillcorner
  run uv run python -m footymind.research.response_sensitivity skillcorner
  # Session 4: envelope v2 (pre-registered) and the innovation response model.
  run uv run python -m footymind.research.envelope_validation skillcorner
  run uv run python -m footymind.research.response_diagnostics skillcorner
  run uv run python -m footymind.research.train_response_model innovation skillcorner
  run uv run python -m footymind.research.response_sensitivity skillcorner 300 2
  run uv run python -m footymind.research.rule_response_check skillcorner 60 3
  run uv run python -m footymind.research.rule_response_check skillcorner 60 4
  run uv run python -m footymind.research.offball_study skillcorner
fi

run uv run --group research python -m footymind.research.figures
echo "done. Compare with committed results: git diff --stat experiments/ src/footymind/spatial/*.json src/footymind/counterfactual/*.json docs/figures"
