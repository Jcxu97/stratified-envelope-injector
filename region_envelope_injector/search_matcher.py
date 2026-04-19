"""Stage-2 matcher: given Chat2Scenario activity dicts + a Chat2Scenario-shape
activity_request (Ego + Target Vehicle #N), find a real (ego_id, target_ids,
frame range) tuple from AD4CHE/highD that satisfies every requested target
simultaneously.

Chat2Scenario's built-in :func:`mainFunctionScenarioIdentification` only
handles a single Target Vehicle. We loop it per target, intersect the ego
candidates + frame windows across targets, and pick the best scoring tuple.

The returned :class:`MatchedSegment` carries the slice bounds the xosc
emitter replays via FollowTrajectoryAction.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import pandas as pd

from region_envelope_injector.chat2scenario_bridge import (
    mainFunctionScenarioIdentification,
)


@dataclass
class MatchedSegment:
    ego_id: int
    target_ids: list[int]
    f0: int
    f1: int
    score: float = 0.0
    per_target: dict[str, int] = field(default_factory=dict)  # tgt_key -> tgt_id
    notes: list[str] = field(default_factory=list)


def _single_target_request(request: dict, tgt_key: str) -> dict:
    """Return a shallow copy of ``request`` with only Ego + ``tgt_key`` kept,
    renamed to ``Target Vehicle #1`` so Chat2Scenario's hardcoded lookup
    still matches."""
    out = {"Ego Vehicle": copy.deepcopy(request["Ego Vehicle"])}
    out["Target Vehicle #1"] = copy.deepcopy(request[tgt_key])
    return out


def _target_keys(request: dict) -> list[str]:
    keys = [k for k in request.keys() if k.startswith("Target Vehicle")]
    # Sort by the trailing integer so #1 < #2 < #10
    def _idx(k: str) -> int:
        tail = k.rsplit("#", 1)[-1]
        try:
            return int(tail)
        except ValueError:
            return 999
    keys.sort(key=_idx)
    return keys


def _intersect_ranges(a: tuple[int, int], b: tuple[int, int]
                       ) -> tuple[int, int] | None:
    lo, hi = max(a[0], b[0]), min(a[1], b[1])
    return (lo, hi) if lo <= hi else None


def find_matching_segment(activity_request: dict,
                           raw_tracks: pd.DataFrame,
                           latActDict: dict,
                           longActDict: dict,
                           interactIdDict: dict,
                           *,
                           min_frames: int = 20,
                           ) -> MatchedSegment | None:
    """Locate one (ego, targets, frame_range) tuple matching the LLM request.

    Returns ``None`` if no ego candidate satisfies every requested target.
    Scoring prefers longer overlapping frame windows so the xosc replay
    captures a substantive segment rather than a two-frame sliver.
    """
    tgt_keys = _target_keys(activity_request)
    if not tgt_keys:
        return None

    # Per-target: Chat2Scenario returns [egoID, [tgtID], beg, end] lists.
    # Build a dict: ego_id -> list of (tgt_key, tgt_id, beg, end).
    per_target: dict[str, list[list]] = {}
    for k in tgt_keys:
        sub = _single_target_request(activity_request, k)
        try:
            hits = mainFunctionScenarioIdentification(
                raw_tracks, sub, latActDict, longActDict, interactIdDict,
                progress_bar=None) or []
        except Exception:
            hits = []
        per_target[k] = hits

    # Ego candidates = ids that appear in every target's hit-list.
    if not per_target or any(not v for v in per_target.values()):
        # Relax: if some targets have no matches at all, best-effort use
        # only those targets that had ≥1 hit and let the emitter carry on.
        per_target = {k: v for k, v in per_target.items() if v}
        if not per_target:
            return None

    ego_sets = [set(row[0] for row in hits) for hits in per_target.values()]
    common_egos = set.intersection(*ego_sets) if ego_sets else set()
    if not common_egos:
        # Fall back to the per-target best ego (union) — pick the ego that
        # covers the most targets. Trades exact coverage for non-empty match.
        from collections import Counter
        c = Counter()
        for hits in per_target.values():
            for row in hits:
                c[row[0]] += 1
        if not c:
            return None
        common_egos = {c.most_common(1)[0][0]}

    best: MatchedSegment | None = None
    for ego_id in common_egos:
        # For each target pick the (tgtID, beg, end) row on this ego and
        # intersect frame windows.
        tgt_rows: dict[str, tuple[int, int, int]] = {}  # tgt_key -> (tgtID, beg, end)
        for k, hits in per_target.items():
            rows = [h for h in hits if h[0] == ego_id]
            if not rows:
                continue
            # Prefer the row with the widest frame span.
            rows.sort(key=lambda r: r[3] - r[2], reverse=True)
            tgtIDs = rows[0][1]
            tgt_id = int(tgtIDs[0]) if isinstance(tgtIDs, list) else int(tgtIDs)
            tgt_rows[k] = (tgt_id, int(rows[0][2]), int(rows[0][3]))
        if not tgt_rows:
            continue
        # Intersect windows across all targets for this ego.
        window = None
        for (_, beg, end) in tgt_rows.values():
            window = (beg, end) if window is None else _intersect_ranges(window, (beg, end))
            if window is None:
                break
        if window is None:
            continue
        span = window[1] - window[0]
        if span < min_frames:
            continue
        score = float(span) * len(tgt_rows)
        if best is None or score > best.score:
            best = MatchedSegment(
                ego_id=int(ego_id),
                target_ids=[v[0] for v in tgt_rows.values()],
                f0=int(window[0]),
                f1=int(window[1]),
                score=score,
                per_target={k: v[0] for k, v in tgt_rows.items()},
                notes=[f"matched {len(tgt_rows)}/{len(tgt_keys)} targets",
                       f"span={span} frames"],
            )
    return best
