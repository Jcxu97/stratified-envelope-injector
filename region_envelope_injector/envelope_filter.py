"""Stage-1 candidate narrower: trim the `DatasetClip.raw_tracks` pool down
to vehicles whose speed / headway / density profile matches the scenario
stratum (F / S / J) and the region envelope before the Chat2Scenario
activity classifier runs over it.

This is a coarse filter on purpose. The expensive stage is
:mod:`chat2scenario_bridge.main_fcn_veh_activity` (O(N_vehicles) × per-vehicle
pandas transforms), so cutting obviously-wrong candidates up front saves
meaningful time on big highD/AD4CHE recordings.

The LLM-derived activity labels do the fine matching in stage 2
(:mod:`search_matcher`); envelope_filter does not enforce exact percentiles.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from region_envelope_injector.dataset_loader import DatasetClip
from region_envelope_injector.envelope_loader import EnvelopeStore


# Stratum speed bands (m/s). Broad on purpose — the envelope percentiles
# refine within the band when available, the band just discards cars that are
# obviously in the wrong phase of flow.
_STRATUM_SPEED = {
    "F": (22.0, 40.0),   # free flow
    "S": (12.0, 28.0),   # synchronised flow
    "J": (0.0, 18.0),    # wide jam / stop-and-go
}


@dataclass
class FilterResult:
    raw_tracks: pd.DataFrame    # filtered, Chat2Scenario-column-named
    kept_ids: list[int]
    dropped_ids: list[int]
    speed_band: tuple[float, float]
    notes: list[str]


def _speed_band_for(store: EnvelopeStore, region: str,
                     stratum: str) -> tuple[float, float]:
    """Prefer D2_thw-derived speed range if present; fall back to static
    :data:`_STRATUM_SPEED` bands. The envelope store's dimension records
    carry p10/p90 for headway, not speed, so we translate THW×speed back to
    a speed band only when the paper locked it explicitly."""
    lo, hi = _STRATUM_SPEED.get(stratum, (0.0, 40.0))
    # If the envelope file carries a "speed_mps" dimension we can use that.
    for dim in ("speed_mps", "D12_speed", "D13_speed_mps"):
        if dim in store.dimensions:
            env = store.lookup(dim, stratum, region)
            if env and "p10" in env and "p90" in env:
                return float(env["p10"]), float(env["p90"])
    return lo, hi


def filter_tracks_by_envelope(clip: DatasetClip,
                               store: EnvelopeStore,
                               *,
                               region: str,
                               stratum: str,
                               min_frames: int = 20,
                               ) -> FilterResult:
    """Return a subset of ``clip.raw_tracks`` (same column schema) whose
    vehicles spend at least ``min_frames`` of their life with mean speed
    inside the stratum/region speed band.

    Also drops vehicles whose trajectories are too short to yield a
    meaningful lateral/longitudinal activity classification.
    """
    raw = clip.raw_tracks
    if raw is None or raw.empty:
        return FilterResult(raw_tracks=raw.iloc[0:0].copy() if raw is not None
                             else pd.DataFrame(),
                            kept_ids=[], dropped_ids=[],
                            speed_band=(0.0, 0.0),
                            notes=["raw_tracks empty"])

    notes: list[str] = []
    speed_lo, speed_hi = _speed_band_for(store, region, stratum)
    notes.append(f"speed band = [{speed_lo:.1f}, {speed_hi:.1f}] m/s "
                 f"(region={region}, stratum={stratum})")

    # Per-vehicle mean speed + frame count (use xVelocity since raw_tracks
    # carries native column names).
    agg = raw.groupby("id").agg(
        mean_v=("xVelocity", lambda s: float(np.mean(np.abs(s)))),
        n_frames=("frame", "size"),
    )
    in_band = (agg["mean_v"] >= speed_lo) & (agg["mean_v"] <= speed_hi)
    long_enough = agg["n_frames"] >= min_frames
    kept_mask = in_band & long_enough

    kept_ids = [int(i) for i in agg.index[kept_mask].tolist()]
    dropped_ids = [int(i) for i in agg.index[~kept_mask].tolist()]
    notes.append(f"kept {len(kept_ids)} / {len(agg)} vehicles "
                 f"({len(dropped_ids)} dropped)")

    if not kept_ids:
        # If the filter is too aggressive (tight envelope, sparse clip)
        # keep the top-N longest-lived vehicles so the matcher has *something*
        # to chew on. Paper-locked envelopes are strict; real clips can be
        # sparse; we'd rather return a degenerate match than zero results.
        fallback = agg.sort_values("n_frames", ascending=False).head(20).index
        kept_ids = [int(i) for i in fallback.tolist()]
        dropped_ids = [i for i in dropped_ids if i not in kept_ids]
        notes.append(f"speed filter produced 0 matches; fell back to "
                     f"top-{len(kept_ids)} longest-lived vehicles")

    subset = raw[raw["id"].isin(kept_ids)].copy().reset_index(drop=True)
    return FilterResult(raw_tracks=subset, kept_ids=kept_ids,
                         dropped_ids=dropped_ids,
                         speed_band=(speed_lo, speed_hi),
                         notes=notes)
