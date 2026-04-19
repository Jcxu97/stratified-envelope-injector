"""Load scenario_envelopes.json into a typed-ish EnvelopeStore."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


REGIONS = ("CN", "DE")
STRATA = ("F", "S", "J")
POOLED_KEYS = ("pooled", "region_invariant")


@dataclass
class EnvelopeStore:
    raw: dict
    dimensions: dict[str, dict] = field(default_factory=dict)
    scenario_templates: dict[str, dict] = field(default_factory=dict)

    def dim_ids(self) -> list[str]:
        return list(self.dimensions.keys())

    def tier_of(self, dim_id: str) -> str:
        d = self.dimensions[dim_id]
        if "tier" in d:
            return d["tier"]
        if "tier_mixed" in d:
            return "MIXED"
        return "UNKNOWN"

    def lookup(self, dim_id: str, stratum: str, region: str) -> dict | None:
        """Return the percentile dict for (dim, stratum, region).

        Resolution order:
          1. envelopes[stratum][region]              # Tier II region-split in this stratum
          2. envelopes[stratum]["pooled"]            # Tier III in this stratum
          3. envelopes["region_invariant"][region]   # Tier I
          4. envelopes["pooled"]                     # Tier III fully pooled
        """
        d = self.dimensions[dim_id]
        envs = d.get("envelopes", {})
        if stratum in envs:
            node = envs[stratum]
            if region in node:
                return node[region]
            if "pooled" in node:
                return node["pooled"]
        for key in POOLED_KEYS:
            if key in envs:
                node = envs[key]
                if region in node:
                    return node[region]
                if isinstance(node, dict) and "p10" in node:
                    return node
        return None


def load_envelopes(path: str | Path) -> EnvelopeStore:
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    store = EnvelopeStore(raw=raw)
    store.dimensions = raw.get("dimensions", {})
    store.scenario_templates = raw.get("scenario_templates", {})
    return store


def rebuild_from_stats_csv(stats_csv_path: str | Path,
                           template_json_path: str | Path,
                           out_path: str | Path) -> None:
    """Placeholder: rebuild scenario_envelopes.json from statistical_results.csv.

    Expected CSV columns: dimension, stratum, region, p10, p50, p90,
    cliffs_delta, fdr_p. This is a hook for the paper's full pipeline;
    the demo envelopes ship with illustrative values.
    """
    import csv
    template = json.loads(Path(template_json_path).read_text(encoding="utf-8"))
    out = template
    with open(stats_csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            dim, stratum, region = row["dimension"], row["stratum"], row["region"]
            if dim not in out["dimensions"]:
                continue
            envs = out["dimensions"][dim].setdefault("envelopes", {}).setdefault(stratum, {})
            envs[region] = {
                "p10": float(row["p10"]),
                "p50": float(row["p50"]),
                "p90": float(row["p90"]),
                "cliffs_delta": float(row["cliffs_delta"]),
                "fdr_p": float(row["fdr_p"]),
            }
    Path(out_path).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
