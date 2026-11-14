"""Possession segmentation from synchronised event data.

Definition (v1, documented in docs/DATA_SOURCES.md): a *possession* is a maximal
run of consecutive on-ball events credited to one team, within one period. A
possession ends when:

* the next possession-bearing event belongs to the other team, or
* the period changes, or
* the event is ``BALL OUT`` or ``SHOT`` (play stops, or the attack ends). A
  following set piece starts a new possession, even for the same team.

Events that do not bear possession (``CHALLENGE``, ``CARD``, ``FAULT
RECEIVED``) are attached to the possession that is open when they occur and
never open or switch one. This is a deliberately simple, transparent rule. It
splits short scrambles into micro-possessions, which the ``n_events`` and
``duration_s`` fields make easy to filter.
"""

from __future__ import annotations

from collections.abc import Mapping

import polars as pl

NON_POSSESSION_TYPES = frozenset({"CHALLENGE", "CARD", "FAULT RECEIVED"})
TERMINAL_TYPES = frozenset({"BALL OUT", "SHOT"})


def assign_possessions(
    events: pl.DataFrame, match_id: str, period_last_frame: Mapping[int, int] | None = None
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return ``(events_with_possession_id, possessions)``.

    Possession ids are stable strings ``"{match_id}:p{index:04d}"``. The index
    is the order of appearance, which is deterministic for a given event file.

    ``period_last_frame`` (period -> last tracked frame) clips each possession's
    ``end_frame`` to its own period, so a provider event whose end lies past the period
    (for example in the next half) can never make a possession span half-time.
    """
    ev = events.sort(["start_frame", "event_id"])
    pos_ids: list[str | None] = []
    current_team: str | None = None
    current_period: int | None = None
    idx = -1
    close_after = False
    for team, etype, period in ev.select("team", "type", "period").iter_rows():
        if etype in NON_POSSESSION_TYPES:
            pos_ids.append(f"{match_id}:p{idx:04d}" if idx >= 0 and not close_after else None)
            continue
        if close_after or team != current_team or period != current_period or etype == "SET PIECE":
            idx += 1
            current_team, current_period = team, period
            close_after = False
        pos_ids.append(f"{match_id}:p{idx:04d}")
        if etype in TERMINAL_TYPES:
            close_after = True

    ev = ev.with_columns(pl.Series("possession_id", pos_ids, dtype=pl.Utf8))
    possessions = (
        ev.filter(
            pl.col("possession_id").is_not_null() & ~pl.col("type").is_in(NON_POSSESSION_TYPES)
        )
        .group_by("possession_id", maintain_order=True)
        .agg(
            pl.col("team").first(),
            pl.col("period").first(),
            pl.col("start_frame").min().alias("start_frame"),
            pl.max_horizontal(pl.col("end_frame").max(), pl.col("start_frame").max()).alias(
                "end_frame"
            ),
            pl.col("start_time_s").min().alias("start_time_s"),
            pl.len().alias("n_events"),
            (pl.col("type") == "PASS").sum().alias("n_passes"),
            pl.col("type").first().alias("first_type"),
            pl.col("subtype").first().alias("first_subtype"),
            pl.col("type").last().alias("last_type"),
            pl.col("subtype").last().alias("last_subtype"),
            (pl.col("type") == "SHOT").any().alias("has_shot"),
            (pl.col("subtype").fill_null("").str.contains("GOAL") & (pl.col("type") == "SHOT"))
            .any()
            .alias("has_goal"),
        )
        .with_columns(
            pl.lit(match_id).alias("match_id"),
        )
    )
    if period_last_frame:
        last = pl.col("period").replace_strict(
            {int(k): int(v) for k, v in period_last_frame.items()},
            default=None,
            return_dtype=pl.Int64,
        )
        possessions = possessions.with_columns(
            pl.when(last.is_not_null() & (pl.col("end_frame") > last))
            .then(pl.max_horizontal(last, pl.col("start_frame")))
            .otherwise(pl.col("end_frame"))
            .alias("end_frame")
        )
    return ev, possessions
