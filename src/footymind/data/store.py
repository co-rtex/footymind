"""Processed-dataset storage: write and read canonical match bundles.

Layout per match (``data/processed/<match_id>/``)::

    tracking.npz        frame, period, time_s, xy (T,P,2) float32, ball_xy (T,2) float32
    players.parquet     player_id, team, jersey, provider_id
    events.parquet      canonical events + possession_id
    possessions.parquet possession summary
    manifest.json       provenance, source hashes, code revision, quality report

``PIPELINE_VERSION`` must be bumped when the processed representation changes
meaning. Loaders refuse bundles from a different pipeline version, so stale
caches are not silently used.

Freshness: in-process caches of a bundle (or of anything derived from it) are keyed on
``bundle_token``, the manifest file's modification time. ``ingest`` writes the manifest
last, so a re-ingest while a server runs is picked up on the next call, including a
pipeline-version change (the reload then refuses the bundle). The data root is resolved
at call time (``footymind.paths.PROCESSED_DIR``), not at import time.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import polars as pl

from footymind import paths
from footymind.data import sources
from footymind.data.quality import validate
from footymind.paths import REPO_ROOT
from footymind.tracking import metrica, metrica_epts, skillcorner
from footymind.tracking.model import DatasetProvenance, PitchSpec, PlayerInfo, TrackingData
from footymind.tracking.possessions import assign_possessions

# 2: SkillCorner adapter v2 events (restarts, recoveries, ball-outs).
# 3: event and possession end frames never leave their period. A SkillCorner event's end was
#    the next possession's start even across half-time, so a few possessions spanned the break
#    (release review, 2026-11-14). Such ends are now null, and possession ends are clipped to
#    the period's last tracked frame. Bundles from v2 carry those cross-period ends.
PIPELINE_VERSION = "3"


def git_revision() -> str:
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--", "src"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return rev + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _processed(root: Path | None) -> Path:
    return paths.PROCESSED_DIR if root is None else root


def ingest(match_id: str, raw_root: Path | None = None, out_root: Path | None = None) -> Path:
    raw_root = paths.RAW_DIR if raw_root is None else raw_root
    src = sources.SOURCES[match_id]
    hashes = sources.verify(src, raw_root)
    d = src.local_dir(raw_root)
    game = match_id.rsplit("-", 1)[-1]
    sync_gate = 1.5
    if src.provider == "skillcorner":
        prov = DatasetProvenance(
            provider="skillcorner",
            source_name=f"SkillCorner open data, A-League 2024/25 match {game}",
            source_url="https://github.com/SkillCorner/opendata",
            license_note=sources.SKILLCORNER_LICENSE_NOTE,
            files=hashes,
        )
    else:
        prov = DatasetProvenance(
            provider="metrica",
            source_name=f"Metrica Sports Sample Game {game}",
            source_url="https://github.com/metrica-sports/sample-data",
            license_note=sources.METRICA_LICENSE_NOTE,
            files=hashes,
        )
    if src.fmt == "skillcorner":
        td, sc_meta = skillcorner.load_tracking(
            match_id, d / f"{game}_tracking_extrapolated.jsonl", d / f"{game}_match.json", prov
        )
        events = skillcorner.load_events(d / f"{game}_dynamic_events.csv", sc_meta, td)
        # Broadcast tracking at 10 fps: the release frame catches a ball already leaving
        # the foot, so the sync gate is wider than for 25 Hz optical tracking.
        sync_gate = 2.5
    elif src.fmt == "metrica_csv":
        td = metrica.load_tracking(
            match_id,
            d / f"Sample_Game_{game}_RawTrackingData_Home_Team.csv",
            d / f"Sample_Game_{game}_RawTrackingData_Away_Team.csv",
            prov,
        )
        events = metrica.load_events(d / f"Sample_Game_{game}_RawEventsData.csv")
    elif src.fmt == "metrica_epts":
        td, meta = metrica_epts.load_tracking(
            match_id,
            d / f"Sample_Game_{game}_tracking.txt",
            d / f"Sample_Game_{game}_metadata.xml",
            prov,
        )
        events = metrica_epts.load_events(d / f"Sample_Game_{game}_events.json", meta)
    else:
        raise NotImplementedError(f"format {src.fmt}")
    return write_bundle(match_id, td, events, prov, sync_gate, out_root)


def write_bundle(
    match_id: str,
    td: TrackingData,
    events: pl.DataFrame,
    prov: DatasetProvenance,
    sync_gate: float = 1.5,
    out_root: Path | None = None,
) -> Path:
    """Validate, segment possessions and write one bundle (the manifest is written last)."""
    report = validate(td, events, sync_gate)
    period_last = {int(p): int(td.frame[td.period == p].max()) for p in np.unique(td.period)}
    events, possessions = assign_possessions(events, match_id, period_last)

    out = _processed(out_root) / match_id
    out.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "frame": td.frame,
        "period": td.period,
        "time_s": td.time_s,
        "xy": td.xy,
        "ball_xy": td.ball_xy,
    }
    if td.detected is not None:
        arrays["detected"] = td.detected
    if td.ball_detected is not None:
        arrays["ball_detected"] = td.ball_detected
    np.savez_compressed(out / "tracking.npz", **arrays)  # type: ignore[arg-type]
    pl.DataFrame([asdict(p) for p in td.players]).write_parquet(out / "players.parquet")
    events.write_parquet(out / "events.parquet")
    possessions.write_parquet(out / "possessions.parquet")
    manifest = {
        "match_id": match_id,
        "pipeline_version": PIPELINE_VERSION,
        "code_revision": git_revision(),
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "pitch": asdict(td.pitch),
        "fps": td.fps,
        "home_attacks_positive_x": {str(k): v for k, v in td.home_attacks_positive_x.items()},
        "provenance": asdict(prov),
        "coordinate_system": "canonical-v1: metres, origin centre spot, y up, provider orientation",
        "counts": {"events": events.height, "possessions": possessions.height},
        "quality": report,
    }
    # Written last and atomically: readers key their caches on this file (bundle_token), and
    # a reader never sees a half-written manifest.
    tmp = out / "manifest.json.tmp"
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    os.replace(tmp, out / "manifest.json")
    return out


class StaleBundleError(RuntimeError):
    pass


def bundle_token(match_id: str, root: Path | None = None) -> tuple[int, int, int] | None:
    """Cheap freshness token: the manifest's (mtime_ns, size, inode); None if it is absent.

    ``write_bundle`` replaces the manifest atomically (a new inode), so even two writes within
    one file-system timestamp tick give different tokens.
    """
    try:
        st = os.stat(_processed(root) / match_id / "manifest.json")
    except OSError:
        return None
    return (st.st_mtime_ns, st.st_size, st.st_ino)


@lru_cache(maxsize=256)
def _read_manifest(path: Path, token: tuple[int, int, int] | None) -> dict[str, Any]:
    out: dict[str, Any] = json.loads(path.read_text())
    return out


def read_manifest(match_id: str, root: Path | None = None) -> dict[str, Any]:
    """The bundle's manifest, cached until the file changes. Raises if it is missing.

    The returned dict is shared between callers: do not mutate it.
    """
    root = _processed(root)
    return _read_manifest(root / match_id / "manifest.json", bundle_token(match_id, root))


def load_match(match_id: str, root: Path | None = None) -> MatchBundle:
    """Load a processed bundle. Cached per manifest version (see ``bundle_token``)."""
    root = _processed(root)
    return _load_match(match_id, root, bundle_token(match_id, root))


def _clear_caches() -> None:
    _load_match.cache_clear()
    _read_manifest.cache_clear()


# kept for callers that used the former ``functools.lru_cache`` API
load_match.cache_clear = _clear_caches  # type: ignore[attr-defined]


T = TypeVar("T")


def cached(fn: Callable[[str], T]) -> Callable[[str], T]:
    """Memoise a per-match function of a bundle, invalidated when the bundle changes."""

    @lru_cache(maxsize=4)
    def inner(match_id: str, _token: tuple[int, int, int] | None) -> T:
        return fn(match_id)

    def wrapper(match_id: str) -> T:
        return inner(match_id, bundle_token(match_id))

    wrapper.__doc__ = fn.__doc__
    wrapper.__name__ = fn.__name__
    wrapper.cache_clear = inner.cache_clear  # type: ignore[attr-defined]
    return wrapper


@lru_cache(maxsize=4)
def _load_match(match_id: str, root: Path, token: tuple[int, int, int] | None) -> MatchBundle:
    d = root / match_id
    manifest = json.loads((d / "manifest.json").read_text())
    if manifest["pipeline_version"] != PIPELINE_VERSION:
        raise StaleBundleError(
            f"{d} built by pipeline v{manifest['pipeline_version']}, expected v{PIPELINE_VERSION}."
            f" Re-run `footymind ingest {match_id}`."
        )
    z = np.load(d / "tracking.npz")
    players = [
        PlayerInfo(**r) for r in pl.read_parquet(d / "players.parquet").iter_rows(named=True)
    ]
    prov = DatasetProvenance(**manifest["provenance"])
    td = TrackingData(
        match_id=match_id,
        pitch=PitchSpec(**manifest["pitch"]),
        fps=float(manifest["fps"]),
        frame=z["frame"],
        period=z["period"],
        time_s=z["time_s"],
        players=players,
        xy=z["xy"],
        ball_xy=z["ball_xy"],
        home_attacks_positive_x={int(k): v for k, v in manifest["home_attacks_positive_x"].items()},
        provenance=prov,
        detected=z["detected"] if "detected" in z.files else None,
        ball_detected=z["ball_detected"] if "ball_detected" in z.files else None,
    )
    return MatchBundle(
        tracking=td,
        events=pl.read_parquet(d / "events.parquet"),
        possessions=pl.read_parquet(d / "possessions.parquet"),
        manifest=manifest,
    )


def available_matches(root: Path | None = None) -> list[str]:
    root = _processed(root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if (p / "manifest.json").exists())


class MatchBundle:
    __slots__ = ("events", "manifest", "possessions", "tracking")

    def __init__(
        self,
        tracking: TrackingData,
        events: pl.DataFrame,
        possessions: pl.DataFrame,
        manifest: dict[str, Any],
    ) -> None:
        self.tracking = tracking
        self.events = events
        self.possessions = possessions
        self.manifest = manifest
