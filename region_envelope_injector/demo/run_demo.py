"""End-to-end demo: produce CN and DE variants of a consecutive-LC scenario
so the two region envelopes can be compared side-by-side.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from region_envelope_injector.injector import generate_region_scenario


DEMO_DESCRIPTION = (
    "Ego vehicle performs four consecutive lane changes across five adjacent "
    "lanes to reach the leftmost lane on a dense, congested highway segment."
)


def main() -> int:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print(" region_envelope_injector: consecutive-lane-change demo")
    print("=" * 72)
    print(f"NL description:\n  {DEMO_DESCRIPTION}\n")

    results = {}
    for region in ("CN", "DE"):
        print(f"\n--- Generating region={region} ---")
        r = generate_region_scenario(
            scenario_description=DEMO_DESCRIPTION,
            region=region,
            out_dir=out_dir,
            seed=42,
        )
        results[region] = r
        print(f"  stratum picked    : {r.request.stratum}")
        print(f"  num lane changes  : {r.request.num_lane_changes}")
        print(f"  dims considered   : {len(r.request.relevant_dimensions)}")
        print(f"  sampled params:")
        for k, v in r.sampled_params.items():
            print(f"       {k:>18s} = {v:.3f}")
        print(f"  xosc              : {r.xosc_path}")

    print("\n" + "=" * 72)
    print(" Side-by-side parameter comparison (CN vs DE):")
    print("=" * 72)
    cn, de = results["CN"].sampled_params, results["DE"].sampled_params
    all_keys = sorted(set(cn) | set(de))
    print(f"  {'parameter':>18s} | {'CN':>10s} | {'DE':>10s} | {'ratio CN/DE':>12s}")
    print(f"  {'-'*18} | {'-'*10} | {'-'*10} | {'-'*12}")
    for k in all_keys:
        vcn, vde = cn.get(k), de.get(k)
        if vcn is not None and vde is not None and vde != 0:
            print(f"  {k:>18s} | {vcn:10.3f} | {vde:10.3f} | {vcn/vde:12.2f}x")
        else:
            vcn_s = f"{vcn:.3f}" if vcn is not None else "-"
            vde_s = f"{vde:.3f}" if vde is not None else "-"
            print(f"  {k:>18s} | {vcn_s:>10s} | {vde_s:>10s} | {'-':>12s}")

    summary = {
        "description": DEMO_DESCRIPTION,
        "CN": {
            "stratum": results["CN"].request.stratum,
            "sampled_params": results["CN"].sampled_params,
            "xosc": results["CN"].xosc_path,
        },
        "DE": {
            "stratum": results["DE"].request.stratum,
            "sampled_params": results["DE"].sampled_params,
            "xosc": results["DE"].xosc_path,
        },
    }
    (out_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nSummary written to {out_dir / 'demo_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
