"""Mechanical tier decision tree.

Given (dimension, stratum, region), decide which envelope source to use:
  - Tier I           : region-invariant large effect -> use region-specific envelope everywhere
  - Tier II          : stratum-conditional -> use region-specific envelope only in strata where robust
  - Tier III         : shared pooled      -> use pooled envelope (no region split)
  - Mixed            : dimension has tier per-stratum (e.g. D7 Tier III in S, Tier II in J)
  - LLM_DOMINANT     : formula is saturated in this stratum, must defer to LLM output
"""
from __future__ import annotations

from dataclasses import dataclass

from region_envelope_injector.envelope_loader import EnvelopeStore


@dataclass
class TierDecision:
    dim_id: str
    stratum: str
    region: str
    tier: str                        # "I" | "II" | "III" | "LLM_DOMINANT"
    envelope_source: str             # "region_specific" | "pooled" | "llm_safety_net"
    envelope: dict | None
    rationale: str


def route(store: EnvelopeStore, dim_id: str, stratum: str, region: str) -> TierDecision:
    d = store.dimensions[dim_id]
    envs = d.get("envelopes", {})

    if "region_invariant" in envs:
        env = envs["region_invariant"].get(region)
        return TierDecision(
            dim_id=dim_id, stratum=stratum, region=region,
            tier="I", envelope_source="region_specific", envelope=env,
            rationale=f"{dim_id}: Tier I (region-invariant large effect across all strata).",
        )

    if "tier_mixed" in d:
        per_stratum = d["tier_mixed"].get(stratum, "III")
        if per_stratum.startswith("III") and d.get("llm_safety_net"):
            return TierDecision(
                dim_id=dim_id, stratum=stratum, region=region,
                tier="LLM_DOMINANT", envelope_source="llm_safety_net",
                envelope=envs.get(stratum, {}).get("pooled"),
                rationale=f"{dim_id}: formula saturates in stratum {stratum}; LLM dual-track is authoritative.",
            )
        if per_stratum.startswith("II"):
            node = envs.get(stratum, {})
            env = node.get(region) or node.get("pooled")
            return TierDecision(
                dim_id=dim_id, stratum=stratum, region=region,
                tier="II", envelope_source="region_specific" if region in node else "pooled",
                envelope=env,
                rationale=f"{dim_id}: Mixed tier -> Tier II in {stratum}, use region-specific envelope.",
            )

    tier_tag = d.get("tier", "III")
    if tier_tag.startswith("II"):
        node = envs.get(stratum, {})
        if region in node:
            return TierDecision(
                dim_id=dim_id, stratum=stratum, region=region,
                tier="II", envelope_source="region_specific", envelope=node[region],
                rationale=f"{dim_id}: Tier {tier_tag}, region split in stratum {stratum}.",
            )
        env = node.get("pooled") or envs.get("pooled")
        return TierDecision(
            dim_id=dim_id, stratum=stratum, region=region,
            tier="II", envelope_source="pooled", envelope=env,
            rationale=f"{dim_id}: Tier {tier_tag}, but stratum {stratum} not split -> pooled fallback.",
        )

    env = envs.get("pooled") or envs.get(stratum, {}).get("pooled")
    return TierDecision(
        dim_id=dim_id, stratum=stratum, region=region,
        tier="III", envelope_source="pooled", envelope=env,
        rationale=f"{dim_id}: Tier III, pooled envelope (no region split).",
    )
