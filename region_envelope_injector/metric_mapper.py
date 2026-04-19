"""Translate sampled envelopes into a Chat2Scenario-compatible
metric_options dict.

Chat2Scenario expects a dict of the shape:
    {
        "Time-Scale":     {"Time To Collision (TTC)": "0.01 - 10"},
        "Distance-Scale": {"Distance Headway (DHW)": "5 - 30"},
        ...
    }

We translate the dimension IDs in our envelope schema into that shape using
the P10..P90 band from the region-specific envelope.
"""
from __future__ import annotations

from region_envelope_injector.envelope_loader import EnvelopeStore
from region_envelope_injector.tier_router import TierDecision, route


DIM_TO_CHAT2SCENARIO = {
    "D1_ttc_min":      ("Time-Scale",     "Time To Collision (TTC)"),
    "D2_thw":          ("Time-Scale",     "Time Headway (THW)"),
    "D5_cutin_dhw":    ("Distance-Scale", "Distance Headway (DHW)"),
    "D10_delta_v":     ("Velocity-Scale", "Relative Velocity"),
    "D11_rlonga":      ("Acceleration-Scale", "Required Longitudinal Deceleration"),
    "D13_accel_dist":  ("Acceleration-Scale", "Longitudinal Acceleration"),
    "D14_jerk":        ("Jerk-Scale",     "Longitudinal Jerk"),
}


def build_metric_options(store: EnvelopeStore,
                         dims: list[str],
                         stratum: str,
                         region: str) -> tuple[dict, list[TierDecision]]:
    """Build Chat2Scenario metric_options for the given region and stratum."""
    out: dict[str, dict[str, str]] = {}
    decisions: list[TierDecision] = []
    for dim_id in dims:
        if dim_id not in DIM_TO_CHAT2SCENARIO:
            continue
        scale, suboption = DIM_TO_CHAT2SCENARIO[dim_id]
        decision = route(store, dim_id, stratum, region)
        decisions.append(decision)
        env = decision.envelope
        if env is None or "p10" not in env:
            continue
        band = f"{env['p10']} - {env['p90']}"
        out.setdefault(scale, {})[suboption] = band
    return out, decisions


def build_chat2scenario_config(store: EnvelopeStore,
                               dims: list[str],
                               stratum: str,
                               region: str,
                               *,
                               scenario_description: str,
                               dataset_path_template: str | None = None,
                               openai_key: str = "REPLACE_ME",
                               model: str = "databricks-claude-opus-4-7",
                               base_url: str = "https://<your-databricks-workspace>/serving-endpoints",
                               output_dir: str = "./output/",
                               track_nums: list[int] | None = None) -> dict:
    metric_options, decisions = build_metric_options(store, dims, stratum, region)
    if dataset_path_template is None:
        dataset_path_template = (
            "./AD4CHE_V1.0/AD4CHE_Data_V1.0/DJI_{track_num}/{track_num}_tracks.csv"
            if region == "CN"
            else "./highD/data/{track_num}_tracks.csv"
        )
    cfg = {
        "name": f"region_{region}_{stratum}",
        "asam_version": "ASAM OpenSCENARIO V1.2.0",
        "dataset_option": "AD4CHE" if region == "CN" else "highD",
        "dataset_path_template": dataset_path_template,
        "metric_option": next(iter(metric_options)) if metric_options else "Time-Scale",
        "metric_suboption": next(iter(next(iter(metric_options.values()), {"": ""}).keys())) if metric_options else "Time To Collision (TTC)",
        "metric_threshold": next(iter(next(iter(metric_options.values()), {"": ""}).values())) if metric_options else "0.01 - 10",
        "CA_Input": None,
        "target_value": None,
        "openai_key": openai_key,
        "model": model,
        "base_url": base_url,
        "output_dir": output_dir,
        "track_nums": track_nums or [1],
        "max_workers": 4,
        "metric_options": metric_options,
        "scenario_description": scenario_description,
        "_region_injector_meta": {
            "region": region,
            "stratum": stratum,
            "dims_considered": dims,
            "tier_decisions": [
                {"dim": d.dim_id, "tier": d.tier, "source": d.envelope_source,
                 "rationale": d.rationale} for d in decisions
            ],
        },
    }
    return cfg
