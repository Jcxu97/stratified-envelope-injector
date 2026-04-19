"""Parse a free-text scenario request into (template_id, stratum, num_lanes,
relevant_dimensions) using lightweight keyword matching.

For production, swap the heuristic with a Databricks-Claude call whose system
prompt is the scenario taxonomy. The heuristic is sufficient for demo and as a
deterministic fallback when the LLM is unavailable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from region_envelope_injector.envelope_loader import EnvelopeStore


@dataclass
class ScenarioRequest:
    raw_text: str
    region: str
    template_id: str
    stratum: str
    num_lane_changes: int
    relevant_dimensions: list[str]
    notes: list[str]


_NUM_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "single": 1, "double": 2, "triple": 3, "quadruple": 4,
}


def _extract_num_lane_changes(text: str) -> int:
    t = text.lower()
    m = re.search(r"(\d+)\s*(?:consecutive\s*)?lane\s*change", t)
    if m:
        return int(m.group(1))
    m = re.search(r"across\s*(\d+)\s*lane", t)
    if m:
        return int(m.group(1)) - 1
    m = re.search(r"(\d+)\s*-?\s*lane", t)
    if m and "consecutive" in t:
        return int(m.group(1)) - 1
    for word, n in _NUM_WORDS.items():
        if re.search(rf"\b{word}\b\s+(?:consecutive\s+)?lane\s*change", t):
            return n
    if "consecutive" in t and "lane" in t:
        return 2
    if "lane change" in t or "cut-in" in t or "cut in" in t:
        return 1
    return 0


def _infer_stratum(text: str, region: str) -> tuple[str, list[str]]:
    t = text.lower()
    notes: list[str] = []
    if any(k in t for k in ["jam", "congest", "dense", "high-density", "high density", "stop-and-go", "queue"]):
        return "J", ["jam keyword detected"]
    if any(k in t for k in ["synchron", "moderate density", "moderate flow", "dense but flowing"]):
        return "S", ["synchronized keyword detected"]
    if any(k in t for k in ["free flow", "free-flow", "light traffic", "low density"]):
        return "F", ["free-flow keyword detected"]
    default = "J" if region == "CN" else "S"
    notes.append(f"no explicit traffic-state keyword; defaulting to {default} (typical for {region} expressway).")
    return default, notes


def _infer_template(text: str, num_lc: int) -> str:
    t = text.lower()
    if num_lc >= 2:
        return "consecutive_lane_change"
    if re.search(r"\bcut[\s-]?in\b|\bcuts\s+in\b", t):
        return "cut_in_conflict"
    if num_lc == 1:
        return "consecutive_lane_change"
    if any(k in t for k in ["follow", "following", "car-follow", "headway", "ttc"]):
        return "close_following"
    return "consecutive_lane_change" if num_lc > 0 else "close_following"


def parse_scenario_request(text: str, region: str,
                           store: EnvelopeStore) -> ScenarioRequest:
    region = region.upper()
    if region not in ("CN", "DE"):
        raise ValueError(f"region must be CN or DE, got {region!r}")
    n_lc = _extract_num_lane_changes(text)
    stratum, notes = _infer_stratum(text, region)
    template_id = _infer_template(text, n_lc)
    template = store.scenario_templates.get(template_id, {})
    dims = list(template.get("relevant_dimensions", []))
    return ScenarioRequest(
        raw_text=text,
        region=region,
        template_id=template_id,
        stratum=stratum,
        num_lane_changes=max(n_lc, 1) if template_id == "consecutive_lane_change" else n_lc,
        relevant_dimensions=dims,
        notes=notes,
    )
