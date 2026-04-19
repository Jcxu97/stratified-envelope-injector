"""End-to-end orchestrator: NL + region -> (sampled params, Chat2Scenario
config, OpenSCENARIO file)."""
from __future__ import annotations

import json
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from region_envelope_injector.envelope_loader import EnvelopeStore, load_envelopes
from region_envelope_injector.metric_mapper import build_chat2scenario_config, build_metric_options
from region_envelope_injector.nl_region_parser import parse_scenario_request, ScenarioRequest
from region_envelope_injector.sampler import sample_envelope, sample_scenario_params
from region_envelope_injector.tier_router import TierDecision, route
from region_envelope_injector.xosc_emitter import emit_xosc

try:
    from region_envelope_injector.nl_llm_parser import parse_scenario_request_llm
except Exception:
    parse_scenario_request_llm = None  # type: ignore


@dataclass
class GenerationResult:
    request: ScenarioRequest
    sampled_params: dict[str, float]
    tier_decisions: list[dict]
    chat2scenario_config: dict
    xosc_path: str | None = None
    provenance: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps({
            "request": {
                "raw_text": self.request.raw_text,
                "region": self.request.region,
                "template_id": self.request.template_id,
                "stratum": self.request.stratum,
                "num_lane_changes": self.request.num_lane_changes,
                "relevant_dimensions": self.request.relevant_dimensions,
                "notes": self.request.notes,
            },
            "sampled_params": self.sampled_params,
            "tier_decisions": self.tier_decisions,
            "chat2scenario_config": self.chat2scenario_config,
            "xosc_path": self.xosc_path,
            "provenance": self.provenance,
        }, indent=2, ensure_ascii=False)


def _collect_sampled_params(store: EnvelopeStore,
                            dims: list[str],
                            stratum: str,
                            region: str,
                            seed: int) -> tuple[dict[str, float], list[TierDecision]]:
    """Walk each relevant dimension, route via tier tree, sample a scalar."""
    params: dict[str, float] = {}
    decisions: list[TierDecision] = []
    if region == "CN":
        params["ego_init_speed"] = 15.0 if stratum == "J" else 22.0
    else:
        params["ego_init_speed"] = 28.0 if stratum != "J" else 10.0
    for dim_id in dims:
        decision = route(store, dim_id, stratum, region)
        decisions.append(decision)
        env = decision.envelope
        if env is None:
            continue
        sampled = sample_scenario_params(env, seed=seed + zlib.crc32(dim_id.encode("utf-8")) % 10_000)
        if dim_id == "D7_lc_aggressiveness":
            params["lat_v_max"] = sampled.get("lat_v_max", 0.8)
            params["lat_a_max"] = sampled.get("lat_a_max", 1.2)
            params["T_LC"] = sampled.get("T_LC", 4.0)
        elif dim_id == "D5_cutin_dhw":
            params["cutin_dhw"] = sampled.get("value", 8.0)
        elif dim_id == "D8_pet":
            params["pet"] = sampled.get("value", 1.5)
        elif dim_id == "D9_gap_acceptance":
            params["lag_gap"] = sampled.get("lag", sampled.get("value", 5.0))
            params["lead_gap"] = sampled.get("lead", sampled.get("value", 8.0))
        elif dim_id == "D10_delta_v":
            params["delta_v"] = sampled.get("value", 2.0)
        elif dim_id == "D1_ttc_min":
            params["ttc_target"] = sampled.get("value", 2.5)
        elif dim_id == "D2_thw":
            params["thw"] = sampled.get("value", 1.5)
        elif dim_id == "D3_ttc_exposure":
            params["ttc_exposure"] = sampled.get("value", 0.5)
        elif dim_id == "D4_idm_T":
            params["idm_T"] = sampled.get("value", 1.2)
        elif dim_id == "D11_rlonga":
            params["brake_decel"] = sampled.get("brake", sampled.get("value", 4.0))
            params["accel_max"] = sampled.get("accel", sampled.get("value", 2.0))
    if "lead_gap" not in params and "thw" in params:
        params["lead_gap"] = params["thw"] * params["ego_init_speed"]
    return params, decisions


def generate_region_scenario(scenario_description: str,
                             region: str,
                             *,
                             envelopes_path: str | Path = "scenario_envelopes.json",
                             out_dir: str | Path = "output",
                             template_path: str | Path | None = None,
                             seed: int = 42,
                             write_xosc: bool = True,
                             write_chat2scenario_config: bool = True,
                             use_llm_parser: bool = False) -> GenerationResult:
    """One-shot entry point.

    Example
    -------
    >>> r = generate_region_scenario(
    ...     "Ego performs four consecutive lane changes in a dense Chinese "
    ...     "highway to reach the leftmost lane.",
    ...     region="CN")
    >>> print(r.xosc_path, r.sampled_params)
    """
    envelopes_path = Path(envelopes_path)
    if not envelopes_path.exists() and not envelopes_path.is_absolute():
        bundled = Path(__file__).parent / envelopes_path.name
        if bundled.exists():
            envelopes_path = bundled
    store = load_envelopes(envelopes_path.resolve())

    if use_llm_parser and parse_scenario_request_llm is not None:
        request = parse_scenario_request_llm(scenario_description, region, store)
    else:
        request = parse_scenario_request(scenario_description, region, store)
    params, decisions = _collect_sampled_params(
        store, request.relevant_dimensions, request.stratum, request.region, seed)
    cfg = build_chat2scenario_config(
        store,
        request.relevant_dimensions,
        request.stratum,
        request.region,
        scenario_description=scenario_description,
    )
    result = GenerationResult(
        request=request,
        sampled_params=params,
        tier_decisions=[{"dim": d.dim_id, "tier": d.tier, "source": d.envelope_source,
                          "rationale": d.rationale, "envelope": d.envelope}
                         for d in decisions],
        chat2scenario_config=cfg,
        provenance={
            "envelopes_source": str(envelopes_path),
            "seed": seed,
            "paper": "Stratified Cross-National Comparison of Highway Driving Behavior (2026)",
        },
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if write_chat2scenario_config:
        cfg_path = out_dir / f"chat2scenario_{request.region}_{request.stratum}_{request.template_id}.json"
        cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    if write_xosc:
        tmpl_map = {
            "consecutive_lane_change": "consecutive_lane_change.xosc",
            "cut_in_conflict": "cut_in_conflict.xosc",
            "close_following": "close_following.xosc",
        }
        tmpl_name = tmpl_map.get(request.template_id)
        if tmpl_name is not None:
            tmpl = template_path or (Path(__file__).parent / "templates" / tmpl_name)
            if request.template_id == "consecutive_lane_change":
                suffix = f"{request.num_lane_changes}LC"
            else:
                suffix = request.template_id
            xosc_out = out_dir / f"scenario_{request.region}_{request.stratum}_{suffix}.xosc"
            emit_xosc(tmpl, xosc_out, params,
                      num_lc=max(request.num_lane_changes, 1),
                      scenario_description=scenario_description,
                      template_id=request.template_id)
            # Stage the xodr next to the xosc so esmini (and any other
            # consumer) can resolve the relative LogicFile path.
            import shutil
            src_rn = Path(__file__).parent / "templates" / "road_network"
            dst_rn = out_dir / "road_network"
            if src_rn.is_dir():
                dst_rn.mkdir(exist_ok=True)
                for xodr in src_rn.glob("*.xodr"):
                    shutil.copy(xodr, dst_rn / xodr.name)
            result.xosc_path = str(xosc_out)

    trace_path = out_dir / f"trace_{request.region}_{request.stratum}_{request.template_id}.json"
    trace_path.write_text(result.to_json(), encoding="utf-8")

    return result
