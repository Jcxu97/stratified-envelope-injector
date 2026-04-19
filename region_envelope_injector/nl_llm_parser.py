"""Databricks-Claude NL parser. ``parse_scenario_request_llm`` -> legacy
``ScenarioRequest``; ``parse_activity_request_llm`` -> Chat2Scenario-shaped
dict (for ``mainFunctionScenarioIdentification``). Env: ``DATABRICKS_TOKEN`` +
(``DATABRICKS_HOST`` or ``ANTHROPIC_BASE_URL``); optional
``DATABRICKS_LLM_ENDPOINT``. Falls back to a keyword heuristic on missing env
or network failure.
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx

from region_envelope_injector.envelope_loader import EnvelopeStore
from region_envelope_injector.nl_region_parser import (
    ScenarioRequest,
    parse_scenario_request as _heuristic_parse,
)

_log = logging.getLogger(__name__)
DEFAULT_ENDPOINT = "databricks-claude-opus-4-7"
DEFAULT_TEMPLATES = ("consecutive_lane_change", "close_following", "cut_in_conflict")
DEFAULT_STRATA = ("F", "S", "J")
# activity vocabulary (Chat2Scenario classification_framework)
_LON_ACTS = ("keep velocity", "acceleration", "deceleration")
_LAT_ACTS = ("follow lane", "lane change left", "lane change right")
_POS_KEYS = {
    "same lane": ("front", "behind"),
    "adjacent lane": ("left adjacent lane", "right adjacent lane"),
    "lane next to adjacent lane": ("lane next to left adjacent lane",
                                    "lane next to right adjacent lane"),
}

# legacy template-parser prompt (kept for parse_scenario_request_llm)
SYSTEM_PROMPT = (
    "You are a scenario-taxonomy classifier for a cross-national highway-behavior "
    "study (AD4CHE vs. highD). Return ONLY valid JSON with keys: template_id "
    "(one of {templates}); stratum (F=free flow | S=synchronised | J=wide jam); "
    "num_lane_changes (int>=0); relevant_dimensions (array of IDs D1..D14). "
    "Heuristics: dense/congest/jam/queue/stop-and-go->J; synchronised/moderate->S; "
    "free flow/light->F; cut-in->cut_in_conflict; consecutive/N lane changes->"
    "consecutive_lane_change; following/headway/TTC->close_following. "
    "No prose, no markdown."
).format(templates=", ".join(DEFAULT_TEMPLATES))

# activity parser prompt (Chat2Scenario-shaped output)
ACTIVITY_SYSTEM_PROMPT = (
    'You are a driving-scenario classifier. Return ONLY a JSON object (no '
    'markdown fences, no prose). Vocabulary: longitudinal = "keep velocity"|'
    '"acceleration"|"deceleration" (target may use "NA"); lateral = '
    '"follow lane"|"lane change left"|"lane change right"; position = '
    '{"same lane":["front"|"behind"]} | {"adjacent lane":["left adjacent lane"|'
    '"right adjacent lane"]} | {"lane next to adjacent lane":["lane next to '
    'left adjacent lane"|"lane next to right adjacent lane"]}. Schema (>=1 '
    'Ego Vehicle + Target Vehicle #1; add #2,#3... only if text mentions them): '
    '{"Ego Vehicle":{"Ego longitudinal activity":["keep velocity"],'
    '"Ego lateral activity":["follow lane"]},"Target Vehicle #1":'
    '{"Target start position":{"adjacent lane":["left adjacent lane"]},'
    '"Target end position":{"same lane":["front"]},"Target behavior":'
    '{"target longitudinal activity":["acceleration"],'
    '"target lateral activity":["lane change right"]}}}. Example input: '
    '"Ego keeps speed; a vehicle in the left adjacent lane accelerates and '
    'cuts right, ending ahead of ego." -> exactly that schema.'
)


def _candidate_dimensions(store: EnvelopeStore, template_id: str) -> list[str]:
    template = store.scenario_templates.get(template_id, {})
    return list(template.get("relevant_dimensions", []))


def _resolve_endpoint() -> tuple[str, str, str]:
    """Return (mode, url, model_name) where mode is 'anthropic' or 'openai'."""
    anth = os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/")
    host = os.environ.get("DATABRICKS_HOST", "").rstrip("/")
    model = os.environ.get("DATABRICKS_LLM_ENDPOINT", DEFAULT_ENDPOINT)
    if anth:
        return "anthropic", f"{anth}/v1/messages", model
    if host:
        return "openai", f"{host}/serving-endpoints/{model}/invocations", model
    raise RuntimeError("Neither ANTHROPIC_BASE_URL nor DATABRICKS_HOST is set.")


def _call_databricks(text: str, region: str, *, token: str,
                     system_prompt: str = SYSTEM_PROMPT,
                     timeout: float = 30.0) -> tuple[dict[str, Any], str]:
    mode, url, model = _resolve_endpoint()
    user_msg = (f"Scenario description (region={region}):\n{text}\n\n"
                f"Return the JSON classification now.")
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if mode == "anthropic":
        headers["anthropic-version"] = "2023-06-01"
        body = {"model": model, "max_tokens": 600, "system": system_prompt,
                "messages": [{"role": "user", "content": user_msg}]}
    else:
        body = {"max_tokens": 600, "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}]}
    with httpx.Client(timeout=timeout) as cli:
        r = cli.post(url, json=body, headers=headers)
        r.raise_for_status()
        return r.json(), mode


def _extract_text(resp: dict[str, Any], mode: str) -> str:
    def _first(lst):
        if isinstance(lst, list) and lst and isinstance(lst[0], dict):
            return lst[0].get("text") or lst[0].get("content") or ""
        return None
    if mode == "anthropic":
        t = _first(resp.get("content") or [])
        if t is not None:
            return t
    else:
        for ch in (resp.get("choices") or [])[:1]:
            c = (ch.get("message") or {}).get("content")
            if isinstance(c, str):
                return c
            t = _first(c)
            if t is not None:
                return t
        for k in ("content", "output_text", "text"):
            if isinstance(resp.get(k), str):
                return resp[k]
    raise ValueError(f"Cannot extract text from LLM response: {json.dumps(resp)[:500]}")


def _parse_json_object(raw: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object in LLM output: {raw!r}")
    try:  # allow single-quoted JSON (Chat2Scenario example style)
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return json.loads(m.group(0).replace("'", '"'))


def _closest(value: str, vocab: tuple[str, ...]) -> str:
    v = (value or "").strip().lower()
    if v in vocab:
        return v
    return next((opt for opt in vocab if v and (v in opt or opt in v)), vocab[0])


def _coerce_act(val: Any, vocab: tuple[str, ...], *, allow_na: bool = False,
                default: str | None = None) -> list[str]:
    items = [val] if isinstance(val, str) else (val or [])
    if not isinstance(items, list) or not items:
        return [default or vocab[0]]
    out = ["NA" if (allow_na and str(it).strip().upper() == "NA")
           else _closest(str(it), vocab) for it in items]
    return out or [default or vocab[0]]


def _coerce_pos(pos: Any) -> dict[str, list[str]]:
    if not isinstance(pos, dict) or not pos:
        return {"same lane": ["front"]}
    raw_key = next(iter(pos.keys()))
    key = str(raw_key).strip().lower()
    if key not in _POS_KEYS:
        key = next((k for k in _POS_KEYS
                    if key.startswith(k) or k.startswith(key)), "same lane")
    allowed = _POS_KEYS[key]
    vals = pos.get(raw_key) or pos.get(key) or []
    if isinstance(vals, str):
        vals = [vals]
    if not isinstance(vals, list) or not vals:
        return {key: [allowed[0]]}
    return {key: [_closest(str(vals[0]), allowed)]}


def _mk_target(start: dict, end: dict, lon: str, lat: str) -> dict[str, Any]:
    return {"Target start position": start, "Target end position": end,
            "Target behavior": {"target longitudinal activity": [lon],
                                "target lateral activity": [lat]}}


def _validate_activity_request(req: dict[str, Any],
                                notes: list[str]) -> dict[str, Any]:
    ego = req.get("Ego Vehicle") or {}
    out: dict[str, Any] = {"Ego Vehicle": {
        "Ego longitudinal activity": _coerce_act(
            ego.get("Ego longitudinal activity"), _LON_ACTS, default="keep velocity"),
        "Ego lateral activity": _coerce_act(
            ego.get("Ego lateral activity"), _LAT_ACTS, default="follow lane"),
    }}
    tgt_keys = [k for k in req if k.startswith("Target Vehicle")]
    if not tgt_keys:
        notes.append("no Target Vehicle found; default Target Vehicle #1 inserted.")
        sf = {"same lane": ["front"]}
        out["Target Vehicle #1"] = _mk_target(sf, sf, "keep velocity", "follow lane")
    for key in tgt_keys:
        tgt = req.get(key) or {}
        beh = tgt.get("Target behavior") or {}
        out[key] = {
            "Target start position": _coerce_pos(tgt.get("Target start position")),
            "Target end position": _coerce_pos(tgt.get("Target end position")),
            "Target behavior": {
                "target longitudinal activity": _coerce_act(
                    beh.get("target longitudinal activity"), _LON_ACTS,
                    allow_na=True, default="keep velocity"),
                "target lateral activity": _coerce_act(
                    beh.get("target lateral activity"), _LAT_ACTS, default="follow lane"),
            },
        }
    out["_notes"] = notes
    return out


def _heuristic_activity_request(text: str, region: str) -> dict[str, Any]:
    t = (text or "").lower()
    notes = [f"heuristic activity parser (region={region})"]
    cut_in = any(k in t for k in ("cut in", "cuts in", "cut-in"))
    following = any(k in t for k in ("following", "headway", "ttc", "car-follow", "跟车"))
    lc = ("lane change" in t) or ("变道" in t)
    sf = {"same lane": ["front"]}
    if cut_in:
        notes.append("matched cut-in keywords")
        from_left = any(k in t for k in ("from left", "left adjacent", "from the left"))
        side = "right" if from_left else "left"
        start = {"adjacent lane": [("left adjacent lane" if side == "right"
                                     else "right adjacent lane")]}
        ego_lat, tgt_lon, tgt_lat = "follow lane", "acceleration", f"lane change {side}"
    elif following and not lc:
        notes.append("matched following/headway keywords")
        start, ego_lat, tgt_lon, tgt_lat = sf, "follow lane", "keep velocity", "follow lane"
    else:
        notes.append("matched lane-change keywords (or default fallback)")
        start, ego_lat, tgt_lon, tgt_lat = sf, "lane change left", "keep velocity", "follow lane"
    return _validate_activity_request({
        "Ego Vehicle": {"Ego longitudinal activity": ["keep velocity"],
                        "Ego lateral activity": [ego_lat]},
        "Target Vehicle #1": _mk_target(start, sf, tgt_lon, tgt_lat),
    }, notes)


def _env_ready() -> tuple[str | None, bool]:
    return (os.environ.get("DATABRICKS_TOKEN"),
            bool(os.environ.get("DATABRICKS_HOST")
                 or os.environ.get("ANTHROPIC_BASE_URL")))


def parse_scenario_request_llm(text: str, region: str, store: EnvelopeStore, *,
                                endpoint: str | None = None,
                                fallback_to_heuristic: bool = True) -> ScenarioRequest:
    """Legacy Databricks-Claude parser returning the template-based dataclass."""
    region = region.upper()
    token, has_host = _env_ready()
    if not token or not has_host:
        if not fallback_to_heuristic:
            raise RuntimeError("DATABRICKS_TOKEN / host not set.")
        out = _heuristic_parse(text, region, store)
        out.notes.append("LLM parser skipped: env vars missing")
        return out
    if endpoint:
        os.environ["DATABRICKS_LLM_ENDPOINT"] = endpoint
    try:
        resp, mode = _call_databricks(text, region, token=token,
                                      system_prompt=SYSTEM_PROMPT)
        parsed = _parse_json_object(_extract_text(resp, mode))
    except Exception as e:
        if not fallback_to_heuristic:
            raise
        _log.warning("LLM parse failed (%s); heuristic fallback.", e)
        out = _heuristic_parse(text, region, store)
        out.notes.append(f"LLM parser failed: {type(e).__name__}: {e}")
        return out
    template_id = parsed.get("template_id") or "consecutive_lane_change"
    if template_id not in DEFAULT_TEMPLATES:
        template_id = "consecutive_lane_change"
    default_s = "J" if region == "CN" else "S"
    stratum = parsed.get("stratum") or default_s
    if stratum not in DEFAULT_STRATA:
        stratum = default_s
    num_lc = int(parsed.get("num_lane_changes", 1) or 1)
    dims = parsed.get("relevant_dimensions")
    if not isinstance(dims, list):
        dims = _candidate_dimensions(store, template_id)
    canon = {d: d for d in store.dimensions}
    canon.update({d.split("_", 1)[0]: d for d in store.dimensions})
    tdims = set(_candidate_dimensions(store, template_id))
    resolved = [canon[d] for d in dims if d in canon]
    if tdims:
        resolved = [d for d in resolved if d in tdims]
    dims = resolved or _candidate_dimensions(store, template_id)
    ep = endpoint or os.environ.get("DATABRICKS_LLM_ENDPOINT", DEFAULT_ENDPOINT)
    return ScenarioRequest(
        raw_text=text, region=region, template_id=template_id, stratum=stratum,
        num_lane_changes=max(num_lc, 1) if template_id == "consecutive_lane_change" else num_lc,
        relevant_dimensions=dims,
        notes=[f"LLM endpoint={ep}", f"stratum={stratum} (LLM)"],
    )


def parse_activity_request_llm(text: str, region: str, store: EnvelopeStore, *,
                                endpoint: str | None = None,
                                fallback_to_heuristic: bool = True) -> dict[str, Any]:
    """Chat2Scenario ``classification_framework``-shaped dict; drop-in for
    ``mainFunctionScenarioIdentification``. ``_notes`` records the parsing
    path; consumers ignore unknown top-level keys."""
    del store  # reserved for future annotations
    region = region.upper()
    token, has_host = _env_ready()
    if not token or not has_host:
        out = _heuristic_activity_request(text, region)
        out["_notes"].insert(0, "LLM parser skipped: env vars missing")
        return out
    if endpoint:
        os.environ["DATABRICKS_LLM_ENDPOINT"] = endpoint
    try:
        resp, mode = _call_databricks(text, region, token=token,
                                      system_prompt=ACTIVITY_SYSTEM_PROMPT)
        parsed = _parse_json_object(_extract_text(resp, mode))
    except Exception as e:
        if not fallback_to_heuristic:
            raise
        _log.warning("LLM activity parse failed (%s); heuristic fallback.", e)
        out = _heuristic_activity_request(text, region)
        out["_notes"].insert(0, f"LLM parser failed: {type(e).__name__}: {e}")
        return out
    ep = endpoint or os.environ.get("DATABRICKS_LLM_ENDPOINT", DEFAULT_ENDPOINT)
    return _validate_activity_request(
        parsed, [f"LLM endpoint={ep}", f"region={region} (LLM path)"])


def main(argv: list[str] | None = None) -> int:
    import argparse
    from pathlib import Path
    from region_envelope_injector.envelope_loader import load_envelopes
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--region", required=True, choices=["CN", "DE"])
    ap.add_argument("--text", required=True)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--envelopes",
                    default="region_envelope_injector/scenario_envelopes.json")
    args = ap.parse_args(argv)
    store = load_envelopes(Path(args.envelopes).resolve())
    activity = parse_activity_request_llm(args.text, args.region, store,
                                          endpoint=args.endpoint)
    print(json.dumps(activity, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
