import type { InterventionIn, ResponsePolicy, ScenarioRequest, ValueModel } from "../api/types";

/** Reachability horizon and ensemble size used for every scenario the app requests. */
export const HORIZON_S = 1.5;
export const ENSEMBLE = 24;
/** Defensive Shadow request settings (the lab layer and the featured cards use the same). */
export const SHADOW_OPTS: { horizonS: number; valueModel: ValueModel } = { horizonS: 1.0, valueModel: "option_model" };

export function scenarioRequest(
  matchId: string,
  frame: number,
  interventions: InterventionIn[],
  policy: ResponsePolicy,
  valueModel: ValueModel,
): ScenarioRequest {
  return {
    match_id: matchId,
    frame,
    interventions,
    response_policy: policy,
    horizon_s: HORIZON_S,
    ensemble: ENSEMBLE,
    value_model: valueModel,
  };
}
