"""Command-line entry point.

Examples
--------
# Chinese dense-traffic 4-lane consecutive LC
python -m region_envelope_injector.cli \\
    --region CN \\
    --text "Ego performs four consecutive lane changes in dense highway traffic" \\
    --out-dir output

# German synchronized-flow cut-in conflict
python -m region_envelope_injector.cli \\
    --region DE \\
    --text "Target vehicle cuts in front of ego with tight headway" \\
    --out-dir output
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from region_envelope_injector.injector import generate_region_scenario


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--region", required=True, choices=["CN", "DE"],
                        help="Region selector (CN=AD4CHE, DE=highD)")
    parser.add_argument("--text", required=True,
                        help="Free-text scenario description")
    parser.add_argument("--envelopes", default=None,
                        help="Path to scenario_envelopes.json (defaults to bundled)")
    parser.add_argument("--out-dir", default="output",
                        help="Directory for generated .xosc / config / trace files")
    parser.add_argument("--seed", type=int, default=42,
                        help="PRNG seed for reproducible sampling")
    parser.add_argument("--no-xosc", action="store_true",
                        help="Skip OpenSCENARIO emission (only produce Chat2Scenario config)")
    parser.add_argument("--no-config", action="store_true",
                        help="Skip Chat2Scenario config emission")
    parser.add_argument("--llm-parser", action="store_true",
                        help="Use Databricks Claude for NL parsing (requires "
                             "ANTHROPIC_BASE_URL or DATABRICKS_HOST + "
                             "DATABRICKS_TOKEN env vars)")
    args = parser.parse_args(argv)

    envelopes_path = Path(args.envelopes) if args.envelopes else (
        Path(__file__).parent / "scenario_envelopes.json"
    )

    result = generate_region_scenario(
        scenario_description=args.text,
        region=args.region,
        envelopes_path=envelopes_path,
        out_dir=args.out_dir,
        seed=args.seed,
        write_xosc=not args.no_xosc,
        write_chat2scenario_config=not args.no_config,
        use_llm_parser=args.llm_parser,
    )

    print(f"\n[region_envelope_injector] Generated region-{args.region} scenario")
    print(f"  template     : {result.request.template_id}")
    print(f"  stratum      : {result.request.stratum} "
          f"{'(inferred)' if result.request.notes else ''}")
    print(f"  dimensions   : {', '.join(result.request.relevant_dimensions)}")
    print(f"  sampled:")
    for k, v in result.sampled_params.items():
        print(f"      {k:>18s} = {v:.3f}")
    print(f"  tier routing:")
    for d in result.tier_decisions:
        print(f"      {d['dim']:<24s} Tier {d['tier']:<14s} ({d['source']})")
    if result.xosc_path:
        print(f"  xosc written : {result.xosc_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
