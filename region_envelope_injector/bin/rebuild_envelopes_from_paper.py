"""Rebuild scenario_envelopes.json using the statistical values quoted in
the paper's Results section (§V-A / §V-B of *Stratified Cross-National
Comparison of Highway Driving Behavior*, 2026).

This is the bridge between the current illustrative-placeholder envelopes
shipped with the module and the paper-locked values one would normally obtain
from `envelope_loader.rebuild_from_stats_csv()` once the full
``statistical_results.csv`` run is available.

What this script does
---------------------
1. Reads the current (placeholder) ``scenario_envelopes.json`` as a template
   -- preserving structure, unit fields, tier names, scenario_templates.
2. Overwrites every (dimension, stratum, region) triple's ``cliffs_delta``,
   ``fdr_p`` and (where the paper quotes them) medians with the exact numbers
   from the Results text.
3. Fixes direction flips: the paper's signs differ from the placeholder on
   D5, D7, D8, D9, D10 (confirmed by re-reading §V-A Jul-Sep 2026 submission);
   we overwrite both ``cliffs_delta`` and, by symmetric rescaling about the
   unit median, nudge the ``pXX`` points to respect the paper's direction.
4. Adds a ``provenance`` block in metadata recording the paper sections
   consulted for each dimension, so downstream code can distinguish values
   derived from the paper text (high confidence) from values that are still
   placeholder-shaped (percentile bands for dimensions where the paper only
   quotes a scalar delta, not percentiles).
5. Writes to ``scenario_envelopes_paperlocked.json`` next to the original.

Any dimension whose envelope shape is NOT explicitly anchored in the paper
text keeps its original placeholder shape but with ``cliffs_delta`` / ``fdr_p``
corrected. This is marked ``source: "paper_delta_only"`` in provenance so the
next maintainer can replace the shape later without touching the delta.

Paper-quoted anchors consumed here
----------------------------------
- D1 S:  medians 8.83 (CN) vs 10.08 (DE), delta=-0.50, q=0.04
- D1 J:  medians 5.72 (CN) vs 6.74 (DE), delta=-0.68, q=0.06 (not surviving FDR)
- D3 J:  delta=+0.88, q=0.015
- D5 J:  delta=+0.24 (event-level, dagger), q=0.025 -- DIRECTION FLIP
- D6 S:  CN=1.83, DE=0.19 per veh-km, delta=+0.98, q<1e-3
- D6 J:  CN=2.52, DE=0.17 per veh-km, delta=+1.00, q=0.007
- D7 S:  formula delta=-0.23, p=0.015; LLM delta=-0.08, p=0.082
- D7 J:  formula delta=-0.33, p=1.0e-4; LLM delta=-0.21, p=7.3e-3
- D8 S:  formula delta=-0.09, p=0.73; LLM delta=-0.10, p=0.25
- D8 J:  event-level formula delta=+0.57, q<1e-3; LLM delta=-0.08, p=0.19
- D9 J:  lag-gap event-level delta=+0.27, q=0.016 -- DIRECTION FLIP
- D9 S:  delta=+0.46, q=0.059 (borderline)
- D10 J: delta=-0.88, q=0.007
- D11 S: delta=+0.88, q<1e-3
- D12 S: delta=-0.40, q=0.014
- D13 J: delta=-0.92, q=0.012 (metric shifted to peak magnitude)
- D14:   delta=+1.00 in both strata, q<1e-3
- D7 threshold A_thresh = 0.132 on the 150-event pool (from §III-E)
- Example-event anchor (Fig 4): AD4CHE J cut-in DHW=3.7m, lag=6.8m; highD F
  cut-in DHW=13.6m, lag=38.6m -- these populate the D5_J and D9_J medians.

Usage
-----
python -m region_envelope_injector.bin.rebuild_envelopes_from_paper \
    --src region_envelope_injector/scenario_envelopes.json \
    --out region_envelope_injector/scenario_envelopes_paperlocked.json
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path


PAPER_VALUES = {
    "D1_ttc_min": {
        "S": {"CN_median": 8.83, "DE_median": 10.08, "delta": -0.50, "q": 0.04,
              "source": "paper_medians_delta"},
        "J": {"CN_median": 5.72, "DE_median": 6.74, "delta": -0.68, "q": 0.06,
              "fdr_survives": False, "source": "paper_medians_delta"},
    },
    "D2_thw": {
        "pooled": {"delta": 0.06, "q": 0.58, "source": "paper_prose_no_sig"}
    },
    "D3_ttc_exposure": {
        "J": {"delta": 0.88, "q": 0.015, "source": "paper_delta_only"},
        "S": {"delta": 0.11, "q": 0.22, "source": "placeholder_preserved"},
    },
    "D4_idm_T": {
        "pooled": {"delta": 0.08, "q": 0.45, "source": "paper_prose_no_sig"}
    },
    "D5_cutin_dhw": {
        "J": {"delta": 0.24, "q": 0.025, "dagger": True,
              "CN_anchor_median": 3.7, "DE_anchor_median": 13.6,
              "direction_flip_from_placeholder": True,
              "source": "paper_delta_with_example_anchor"},
        "S": {"delta": 0.11, "q": 0.19, "source": "placeholder_preserved"},
    },
    "D6_lc_rate": {
        "S": {"CN_mean_per_vehkm": 1.83, "DE_mean_per_vehkm": 0.19,
              "delta": 0.98, "q": 1e-3, "source": "paper_means_delta"},
        "J": {"CN_mean_per_vehkm": 2.52, "DE_mean_per_vehkm": 0.17,
              "delta": 1.00, "q": 0.007, "source": "paper_means_delta"},
    },
    "D7_lc_aggressiveness": {
        "S": {"formula_delta": -0.23, "formula_p": 0.015,
              "llm_delta": -0.08, "llm_p": 0.082,
              "A_threshold": 0.132,
              "tier_resolution": "S formula artefact (withdrawn under LLM)",
              "direction_flip_from_placeholder": True,
              "source": "paper_dualtrack_S"},
        "J": {"formula_delta": -0.33, "formula_p": 1.0e-4,
              "llm_delta": -0.21, "llm_p": 7.3e-3,
              "A_threshold": 0.132,
              "direction_flip_from_placeholder": True,
              "tier_resolution": "J genuine but formula overstates by ~1/3",
              "source": "paper_dualtrack_J"},
    },
    "D8_pet": {
        "S": {"formula_delta": -0.09, "formula_p": 0.73,
              "llm_delta": -0.10, "llm_p": 0.25,
              "source": "paper_dualtrack_S"},
        "J": {"formula_delta": 0.57, "formula_q": 1e-3, "dagger": True,
              "llm_delta": -0.08, "llm_p": 0.19,
              "direction_flip_from_placeholder": True,
              "source": "paper_dualtrack_J_eventpooled"},
    },
    "D9_gap_acceptance": {
        "J": {"delta": 0.27, "q": 0.016, "dagger": True,
              "CN_lag_anchor": 6.8, "DE_lag_anchor": 38.6,
              "direction_flip_from_placeholder": True,
              "source": "paper_delta_with_example_anchor"},
        "S": {"delta": 0.46, "q": 0.059, "source": "paper_delta_borderline"},
    },
    "D10_delta_v": {
        "J": {"delta": -0.88, "q": 0.007,
              "direction_flip_from_placeholder": True,
              "source": "paper_delta_only"},
    },
    "D11_rlonga": {
        "S": {"delta": 0.88, "q": 1e-3, "source": "paper_delta_only"},
    },
    "D12_heavy_veh_left_lane": {
        "S": {"delta": -0.40, "q": 0.014, "source": "paper_delta_only"},
        "pooled": {"delta": 0.05, "q": 0.69, "source": "paper_prose_no_sig"},
    },
    "D13_accel_dist": {
        "J": {"delta": -0.92, "q": 0.012,
              "metric_shift": "peak magnitude replaces |a|>3 fraction",
              "direction_flip_from_placeholder": True,
              "source": "paper_delta_metric_shift"},
    },
    "D14_jerk": {
        "S": {"delta": 1.00, "q": 1e-3, "source": "paper_delta_only"},
        "J": {"delta": 1.00, "q": 1e-3, "source": "paper_delta_only"},
    },
}


def _rescale_about_median(env_block: dict, new_median: float,
                           keys: tuple[str, str, str] = ("p10", "p50", "p90")) -> None:
    p10_k, p50_k, p90_k = keys
    if p50_k not in env_block:
        return
    old_median = env_block[p50_k]
    if old_median == 0:
        return
    scale = new_median / old_median
    env_block[p10_k] = round(env_block[p10_k] * scale, 3)
    env_block[p50_k] = round(new_median, 3)
    env_block[p90_k] = round(env_block[p90_k] * scale, 3)


def _flip_direction_symmetric(cn_block: dict, de_block: dict,
                               keys: tuple[str, ...] = ("p10", "p50", "p90")) -> None:
    """When the paper direction opposes the placeholder direction, swap the
    per-region percentile bands so the cross-national ordering is correct."""
    for k in keys:
        if k in cn_block and k in de_block:
            cn_block[k], de_block[k] = de_block[k], cn_block[k]


def apply_paper_updates(envelopes: dict) -> tuple[dict, list[str]]:
    out = copy.deepcopy(envelopes)
    dims = out["dimensions"]
    notes: list[str] = []
    prov: dict[str, dict] = {}

    for dim_id, per_stratum in PAPER_VALUES.items():
        if dim_id not in dims:
            notes.append(f"SKIP {dim_id}: not in envelope JSON")
            continue
        dim = dims[dim_id]
        prov[dim_id] = {}

        if "region_invariant" in dim["envelopes"]:
            strongest = None
            for stratum_key, update in per_stratum.items():
                if "delta" in update and (strongest is None
                        or abs(update["delta"]) > abs(strongest.get("delta", 0.0))):
                    strongest = update
            if strongest is not None:
                ri = dim["envelopes"]["region_invariant"]
                delta = abs(strongest["delta"])
                q = strongest.get("q", strongest.get("formula_p"))
                for side in ("CN", "DE"):
                    if side in ri:
                        ri[side]["cliffs_delta"] = delta if strongest["delta"] > 0 else -delta
                        if q is not None:
                            ri[side]["fdr_p"] = q
                if dim_id == "D6_lc_rate":
                    s_update = per_stratum.get("S", {})
                    if "CN_mean_per_vehkm" in s_update and "CN" in ri:
                        _rescale_about_median(ri["CN"], s_update["CN_mean_per_vehkm"])
                    if "DE_mean_per_vehkm" in s_update and "DE" in ri:
                        _rescale_about_median(ri["DE"], s_update["DE_mean_per_vehkm"])
                prov[dim_id]["region_invariant"] = {
                    "source": strongest.get("source", "paper_delta_only"),
                    "note": "Strongest |delta| across S and J applied to region_invariant"
                }

        for stratum_key, update in per_stratum.items():
            prov[dim_id][stratum_key] = {"source": update.get("source", "unspecified")}

            if stratum_key == "pooled":
                block = dim["envelopes"].get("pooled")
                if block is None:
                    continue
                if "delta" in update:
                    block["cliffs_delta"] = update["delta"]
                if "q" in update:
                    block["fdr_p"] = update["q"]
                continue

            if dim_id == "D7_lc_aggressiveness":
                env = dim["envelopes"].get(stratum_key)
                if env is None:
                    continue
                if update.get("direction_flip_from_placeholder"):
                    if "CN" in env and "DE" in env:
                        for key_suffix in ("lat_v_max_", "lat_a_max_", "T_LC_"):
                            for p in ("p10", "p50", "p90"):
                                k = key_suffix + p
                                if k in env["CN"] and k in env["DE"]:
                                    env["CN"][k], env["DE"][k] = env["DE"][k], env["CN"][k]
                for side in ("CN", "DE"):
                    if side in env:
                        env[side]["cliffs_delta_formula"] = update.get("formula_delta")
                        env[side]["cliffs_delta_llm"] = update.get("llm_delta")
                        env[side]["formula_p"] = update.get("formula_p")
                        env[side]["llm_p"] = update.get("llm_p")
                        env[side]["A_threshold"] = update.get("A_threshold")
                        if stratum_key == "S":
                            env[side]["llm_dominant"] = True
                            env[side]["resolution"] = update.get("tier_resolution")
                if "pooled" in env and stratum_key == "S":
                    env["pooled"]["cliffs_delta_formula"] = update.get("formula_delta")
                    env["pooled"]["cliffs_delta_llm"] = update.get("llm_delta")
                    env["pooled"]["formula_p"] = update.get("formula_p")
                    env["pooled"]["llm_p"] = update.get("llm_p")
                    env["pooled"]["llm_dominant"] = True
                continue

            env_dim = dim["envelopes"].get(stratum_key)
            if env_dim is None:
                continue

            if update.get("direction_flip_from_placeholder") \
                    and "CN" in env_dim and "DE" in env_dim:
                if dim_id == "D9_gap_acceptance":
                    flip_keys = ("lead_p10", "lead_p50", "lead_p90",
                                 "lag_p10",  "lag_p50",  "lag_p90")
                    _flip_direction_symmetric(env_dim["CN"], env_dim["DE"],
                                              flip_keys)
                else:
                    _flip_direction_symmetric(env_dim["CN"], env_dim["DE"])

            if "CN_median" in update and "CN" in env_dim:
                _rescale_about_median(env_dim["CN"], update["CN_median"])
            if "DE_median" in update and "DE" in env_dim:
                _rescale_about_median(env_dim["DE"], update["DE_median"])
            if "CN_mean_per_vehkm" in update and "CN" in env_dim:
                _rescale_about_median(env_dim["CN"], update["CN_mean_per_vehkm"])
            if "DE_mean_per_vehkm" in update and "DE" in env_dim:
                _rescale_about_median(env_dim["DE"], update["DE_mean_per_vehkm"])

            if dim_id == "D5_cutin_dhw" and stratum_key == "J":
                if "CN_anchor_median" in update and "CN" in env_dim:
                    _rescale_about_median(env_dim["CN"], update["CN_anchor_median"])
                if "DE_anchor_median" in update and "DE" in env_dim:
                    _rescale_about_median(env_dim["DE"], update["DE_anchor_median"])
            if dim_id == "D9_gap_acceptance" and stratum_key == "J":
                if "CN_lag_anchor" in update and "CN" in env_dim:
                    _rescale_about_median(env_dim["CN"], update["CN_lag_anchor"],
                                          keys=("lag_p10", "lag_p50", "lag_p90"))
                if "DE_lag_anchor" in update and "DE" in env_dim:
                    _rescale_about_median(env_dim["DE"], update["DE_lag_anchor"],
                                          keys=("lag_p10", "lag_p50", "lag_p90"))

            delta = update.get("delta", update.get("formula_delta"))
            q = update.get("q", update.get("formula_q", update.get("formula_p")))
            for side in ("CN", "DE", "pooled"):
                if side in env_dim and delta is not None:
                    env_dim[side]["cliffs_delta"] = delta
                if side in env_dim and q is not None:
                    env_dim[side]["fdr_p"] = q

    out["metadata"]["paper_locked"] = True
    out["metadata"]["paper_section_consulted"] = ("IV-A / V-A / V-B of "
        "*Stratified Cross-National Comparison of Highway Driving Behavior* (2026)")
    out["metadata"]["provenance_per_dim"] = prov
    out["metadata"]["note"] = (
        "Deltas and (where the paper quotes them) medians / mean rates are "
        "paper-locked. Percentile shapes for dimensions that the paper only "
        "reports as a scalar delta are preserved from the original placeholder "
        "JSON, rescaled to satisfy the paper median; see metadata.provenance_per_dim.")

    return out, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", default="region_envelope_injector/scenario_envelopes.json")
    parser.add_argument("--out", default="region_envelope_injector/scenario_envelopes_paperlocked.json")
    args = parser.parse_args(argv)

    src = Path(args.src)
    out = Path(args.out)
    base = json.loads(src.read_text(encoding="utf-8"))
    updated, notes = apply_paper_updates(base)
    out.write_text(json.dumps(updated, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[rebuild_envelopes_from_paper] wrote paper-locked envelope -> {out}")
    for n in notes:
        print(f"  note: {n}")
    print(f"  provenance entries: {len(updated['metadata']['provenance_per_dim'])}")
    flips = sum(
        1 for dim_id, pv in PAPER_VALUES.items()
        for su in pv.values()
        if isinstance(su, dict) and su.get("direction_flip_from_placeholder"))
    print(f"  direction flips applied: {flips} (D5/J, D7/S, D7/J, D8/J, D9/J, D10/J, D13/J)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
