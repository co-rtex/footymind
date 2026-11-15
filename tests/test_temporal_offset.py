"""The temporal leakage guard computes features strictly before the release frame."""

import pytest

from footymind.data.store import available_matches, load_match
from footymind.research.pass_dataset import DatasetConfig, build
from footymind.spatial.state import frame_state
from footymind.spatial.temporal import temporal_context

MET = [m for m in available_matches() if m.startswith("metrica-")]


@pytest.mark.skipif(not MET, reason="Metrica data not ingested")
def test_offset_moves_the_temporal_cutoff_back() -> None:
    mid = MET[0]
    df = build([mid], DatasetConfig(temporal_offset_s=0.2)).head(15)
    td = load_match(mid).tracking
    for r in df.iter_rows(named=True):
        i0 = td.index_of_frame(int(r["frame"]))
        i_t = i0 - 5  # 0.2 s at 25 Hz
        s = frame_state(mid, int(r["frame"]))
        direct = temporal_context(td, i_t, r["team"], td.player_index(r["passer_id"]), s.is_gk)
        for k, v in direct.items():
            assert r[k] == pytest.approx(v, nan_ok=True), k


@pytest.mark.skipif(not MET, reason="Metrica data not ingested")
def test_static_offset_reads_the_earlier_frame() -> None:
    import numpy as np

    mid = MET[0]
    df = build([mid], DatasetConfig(static_offset_s=0.2, temporal_offset_s=0.2)).head(10)
    td = load_match(mid).tracking
    for r in df.iter_rows(named=True):
        i0 = td.index_of_frame(int(r["frame"]))
        s = frame_state(mid, int(td.frame[i0 - 5]))
        p = td.player_index(r["passer_id"])
        dfn = (~s.team_mask(r["team"])) & s.on_pitch()
        d = np.linalg.norm(s.xy[dfn] - s.xy[p][None, :], axis=1)
        assert r["press_nearest_def_m"] == pytest.approx(float(d.min()))
