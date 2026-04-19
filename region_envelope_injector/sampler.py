"""Sample concrete parameter values from a percentile envelope.

All draws are deterministic under a provided seed so generated scenarios are
replayable.
"""
from __future__ import annotations

import random
from typing import Any


def _piecewise_linear(u: float, breakpoints: list[tuple[float, float]]) -> float:
    """Invert a piecewise-linear CDF defined at percentiles (prob, value)."""
    breakpoints = sorted(breakpoints, key=lambda t: t[0])
    if u <= breakpoints[0][0]:
        return breakpoints[0][1]
    if u >= breakpoints[-1][0]:
        return breakpoints[-1][1]
    for (p1, v1), (p2, v2) in zip(breakpoints, breakpoints[1:]):
        if p1 <= u <= p2:
            t = (u - p1) / (p2 - p1)
            return v1 + t * (v2 - v1)
    return breakpoints[-1][1]


def sample_envelope(envelope: dict, key_prefix: str = "", rng: random.Random | None = None) -> float:
    """Sample a single scalar from {p10, p50, p90} via piecewise-linear inverse CDF."""
    rng = rng or random.Random()
    p10 = envelope[f"{key_prefix}p10"]
    p50 = envelope[f"{key_prefix}p50"]
    p90 = envelope[f"{key_prefix}p90"]
    return _piecewise_linear(rng.random(), [(0.1, p10), (0.5, p50), (0.9, p90)])


def sample_scenario_params(envelope: dict, seed: int = 42) -> dict[str, float]:
    """Sample every scalar triple in the envelope.

    Envelopes can hold multiple named parameter families (e.g. D7 carries
    lat_v_max, lat_a_max, T_LC); this helper discovers them by scanning for
    keys that end in '_p10' and returns a flat dict of sampled values.
    """
    rng = random.Random(seed)
    out: dict[str, float] = {}
    prefixes: set[str] = set()
    for k in envelope.keys():
        if k.endswith("_p10"):
            prefixes.add(k[:-3])
        if k in ("p10", "p50", "p90"):
            prefixes.add("")
    for prefix in prefixes:
        try:
            out[prefix.rstrip("_") or "value"] = sample_envelope(envelope, prefix, rng)
        except KeyError:
            continue
    return out
