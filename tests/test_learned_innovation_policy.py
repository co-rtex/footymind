"""The learned_innovation response policy on SYNTHETIC frame states and a hand-set model."""

from __future__ import annotations

import numpy as np
import pytest

from footymind.counterfactual.response import (
    ALL_POLICY_NAMES,
    POLICY_NAMES,
    ResponseContext,
    policy_info,
)
from footymind.counterfactual.response_innovation import (
    INNOVATION_FEATURES,
    InnovationResponseModel,
)
from footymind.tracking.model import PitchSpec, Provenance
from tests.test_response_policies import CARRIER, NAN_POS, RECEIVER, mirror, run, shift
from tests.test_response_policies import synthetic_state as state

NF = len(INNOVATION_FEATURES)


def hand_model(near: float = 0.5, zone: float = 0.3) -> InnovationResponseModel:
    """Known gains in metres per metre (x and y alike), offsets 0.5 s and 1.0 s."""
    g = np.zeros((2, NF))
    g[:, 0], g[:, 2] = near, zone
    return InnovationResponseModel(
        offsets_s=[0.5, 1.0],
        x_mean=np.zeros((2, NF)),
        x_scale=np.ones((2, NF)),
        coef_x=g.copy(),
        intercept_x=np.array([0.05, 0.1]),
        y_scale=np.ones((2, NF)),
        coef_y=g.copy(),
        meta={"synthetic": True},
    )


def ictx(**kw: object) -> ResponseContext:
    base: dict[str, object] = {
        "pitch": PitchSpec(),
        "horizon_s": 1.5,
        "innovation_model": hand_model(),
    }
    base.update(kw)
    return ResponseContext(**base)  # type: ignore[arg-type]


def test_served_names_unchanged_and_new_policy_accepted() -> None:
    assert "learned_innovation" not in POLICY_NAMES  # the API schema enumerates POLICY_NAMES
    assert (*POLICY_NAMES, "learned_innovation") == ALL_POLICY_NAMES


@pytest.mark.parametrize("who", [CARRIER, RECEIVER])
def test_moves_only_outfield_defenders_by_a_visible_amount(who: int) -> None:
    s = state()
    hyp, moves, _ = run(s, [shift(s, who, 4.0, 3.0)], "learned_innovation", ictx())
    assert moves
    assert max(np.hypot(m.to_xy[0] - m.from_xy[0], m.to_xy[1] - m.from_xy[1]) for m in moves) > 0.5
    for m in moves:
        j = s.index(m.player_id)
        assert s.teams[j] == "away" and not s.is_gk[j] and j != NAN_POS
        assert m.provenance == Provenance.SIMULATED and "innovation model" in m.reason
        assert hyp.position_provenance[j] == Provenance.SIMULATED


def test_mirror_invariance() -> None:
    s = state(seed=3)
    m = mirror(s)
    _, a, _ = run(s, [shift(s, CARRIER, 3.0, -2.0)], "learned_innovation", ictx())
    _, b, _ = run(m, [shift(m, CARRIER, -3.0, -2.0)], "learned_innovation", ictx())
    assert [x.player_id for x in a] == [x.player_id for x in b]
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(y.to_xy, (-x.to_xy[0], x.to_xy[1]), atol=1e-6)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_small_intervention_small_response(seed: int) -> None:
    s = state(seed=seed)
    _, moves, _ = run(s, [shift(s, RECEIVER, 0.06, 0.08)], "learned_innovation", ictx())
    for mv in moves:
        assert np.hypot(mv.to_xy[0] - mv.from_xy[0], mv.to_xy[1] - mv.from_xy[1]) <= 0.1 + 1e-6


def test_no_attacker_moved_no_model_and_unvalidated_status() -> None:
    s = state()
    _, moves, notes = run(s, [shift(s, 13, 2.0, 0.0)], "learned_innovation", ictx())
    assert moves == [] and any("no outfield attacker" in n for n in notes)
    with pytest.raises(ValueError):
        run(s, [shift(s, CARRIER, 2.0, 0.0)], "learned_innovation", ictx(innovation_model=None))
    info = policy_info("learned_innovation", "no-such-provider")
    assert not info.validated and info.status.startswith("experimental")
    assert any("not a causal effect" in a for a in info.assumptions)
    assert policy_info("learned", "metrica").description.startswith("Forecaster")
