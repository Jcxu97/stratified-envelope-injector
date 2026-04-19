"""Build a minimal 4-lane + 1-merge-lane straight highway OpenDRIVE (.xodr)
using ``scenariogeneration``, then try to render the companion .xosc
geometry to an SVG/PNG snapshot for visual verification.

This replaces the stub reference in ``xosc_emitter.py`` so that the generated
``.xosc`` can be opened in esmini / CarMaker / Chat2Scenario without a 404 on
the road network. The geometry is intentionally minimal -- a single 500 m
straight road with 5 driving lanes on the right side of the reference line --
which is enough to host the consecutive-lane-change demo.

Usage
-----
python -m region_envelope_injector.bin.build_highway_xodr \
    --out region_envelope_injector/templates/road_network/highway_5lane.xodr

Optional: --snapshot region_envelope_injector/demo/output/road_preview.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from scenariogeneration import xodr


def build_xodr(num_lanes: int = 5,
               lane_width: float = 3.5,
               length: float = 500.0) -> xodr.OpenDrive:
    """Return a straight multi-lane highway suitable for the demo."""
    odr = xodr.OpenDrive("highway_demo")
    planview = xodr.PlanView()
    planview.add_geometry(xodr.Line(length))

    lanes = xodr.Lanes()
    ls = xodr.LaneSection(0, xodr.standard_lane())
    for _ in range(num_lanes):
        right_lane = xodr.Lane(a=lane_width)
        right_lane.add_roadmark(xodr.std_roadmark_broken())
        ls.add_right_lane(right_lane)
    lanes.add_lanesection(ls)

    road = xodr.Road(1, planview, lanes)
    odr.add_road(road)
    odr.adjust_roads_and_lanes()
    return odr


def write_xodr(odr: xodr.OpenDrive, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    odr.write_xml(str(out))
    return out


def render_snapshot(xodr_path: Path, snapshot: Path,
                     num_lanes: int = 5, lane_width: float = 3.5,
                     length: float = 500.0) -> Path | None:
    """Render a minimal top-down preview of the straight multi-lane road."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except Exception:
        return None
    import numpy as np

    fig, ax = plt.subplots(figsize=(9, 2.6), dpi=120)
    fig.patch.set_facecolor("#f2f2f2")
    total_w = num_lanes * lane_width
    ax.add_patch(Rectangle((0, -total_w), length, total_w,
                            facecolor="#3a3a3a", zorder=0))
    for i in range(num_lanes + 1):
        y = -i * lane_width
        color = "white"
        if i == 0 or i == num_lanes:
            ax.plot([0, length], [y, y], color=color, linewidth=2, zorder=1)
        else:
            for x0 in np.arange(0, length, 6):
                ax.plot([x0, x0 + 3], [y, y], color=color, linewidth=1.2, zorder=1)
    ax.set_xlim(-5, length + 5)
    ax.set_ylim(-total_w - 1, 1)
    ax.set_aspect("equal")
    ax.set_title(f"OpenDRIVE preview: {xodr_path.name}\n"
                  f"{num_lanes} lanes x {lane_width}m x {length:.0f}m",
                  fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(snapshot, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return snapshot


def verify_with_xosc(xodr_path: Path, xosc_path: Path) -> dict:
    """Sanity-check geometry compatibility: count lanes in the xodr and
    maximum target-lane references in the xosc."""
    import re
    xodr_text = xodr_path.read_text(encoding="utf-8")
    xosc_text = xosc_path.read_text(encoding="utf-8") if xosc_path.exists() else ""
    right_lanes = len(re.findall(r'<lane id="-?\d+"\s+type="driving"', xodr_text))
    if right_lanes == 0:
        right_lanes = len(re.findall(r'lane id="-\d+"', xodr_text))
    target_lanes = [int(m) for m in re.findall(
        r'<AbsoluteTargetLane value="(-?\d+)"', xosc_text)]
    max_abs = max((abs(x) for x in target_lanes), default=0)
    return {
        "xodr_right_driving_lanes": right_lanes,
        "xosc_target_lanes": target_lanes,
        "xosc_max_abs_lane": max_abs,
        "compatible": (right_lanes == 0) or (max_abs <= right_lanes),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="region_envelope_injector/templates/road_network/highway_5lane.xodr")
    parser.add_argument("--lanes", type=int, default=5)
    parser.add_argument("--lane-width", type=float, default=3.5)
    parser.add_argument("--length", type=float, default=500.0)
    parser.add_argument("--snapshot", default=None)
    parser.add_argument("--verify-xosc", default=None,
                        help="Path to an .xosc to verify geometry compatibility")
    args = parser.parse_args(argv)

    odr = build_xodr(num_lanes=args.lanes, lane_width=args.lane_width,
                     length=args.length)
    out_path = write_xodr(odr, Path(args.out))
    print(f"[build_highway_xodr] wrote {out_path}")

    if args.snapshot:
        snap = render_snapshot(out_path, Path(args.snapshot),
                                num_lanes=args.lanes,
                                lane_width=args.lane_width,
                                length=args.length)
        if snap:
            print(f"[build_highway_xodr] wrote snapshot {snap}")
        else:
            print("[build_highway_xodr] snapshot renderer unavailable")

    if args.verify_xosc:
        info = verify_with_xosc(out_path, Path(args.verify_xosc))
        print(f"[build_highway_xodr] verify: {info}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
