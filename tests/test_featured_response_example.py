"""The featured 'response changes the conclusion' example (real Metrica data; skipped if absent).

Guards the curated example shown in the guided tour: moving home #5 at Game 2 frame 31360 into
space raises the option-model estimate with static defenders, but with the marker_follows
response (a rule consistent with real marker movement, C37) the gain disappears.
"""

import pytest

from footymind.counterfactual.scenario import Intervention, run_scenario
from footymind.data.store import available_matches

pytestmark = pytest.mark.skipif(
    "metrica-sample-2" not in available_matches(), reason="Metrica Game 2 not ingested"
)
IV = [Intervention("home_5", -29.04, 9.47)]


def _delta(policy: str) -> tuple[float, float]:
    s = run_scenario("metrica-sample-2", 31360, IV, policy)
    assert s.supported
    d = next(o for o in s.option_deltas if o.receiver_id == "home_5")
    return float(d.delta), float(d.delta_sign_agreement)


def test_static_gain_disappears_when_the_marker_follows() -> None:
    static, agree0 = _delta("none")
    marked, agree1 = _delta("marker_follows")
    assert static > 0.05 and marked < 0 and agree0 == 1.0 and agree1 == 1.0
