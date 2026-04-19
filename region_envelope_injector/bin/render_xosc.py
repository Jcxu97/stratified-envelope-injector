"""Parse a .xosc ParameterDeclarations block and render a top-down GIF that
reflects its Story actions. Not a full OpenSCENARIO interpreter -- just enough
to visualize what the template encodes for demo / review purposes.

Supported templates (matched by file name):
  - *_consecutive_lane_change / *LC.xosc  -> Ego executes N sinusoidal LCs
  - *_cut_in_conflict.xosc                -> Target cuts from right-adjacent lane into Ego lane
  - *_close_following.xosc                -> Lead brakes linearly; Ego car-follows

Usage
-----
python -m region_envelope_injector.bin.render_xosc \
    --xosc region_envelope_injector/demo/output_follow/scenario_CN_J_close_following.xosc \
    --out  region_envelope_injector/demo/output_follow/scenario_CN_J_close_following.gif
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

LANE_WIDTH = 3.5
NUM_LANES = 5
FPS = 15
VIEW_AHEAD = 70.0
VIEW_BEHIND = 40.0


def _parse_params(xosc_path: Path) -> dict[str, float | str]:
    root = ET.parse(xosc_path).getroot()
    out: dict[str, float | str] = {}
    for pd in root.iter("ParameterDeclaration"):
        name = pd.attrib["name"]
        val = pd.attrib["value"]
        try:
            out[name] = float(val)
        except ValueError:
            out[name] = val
    sim_cond = None
    for cond in root.iter("SimulationTimeCondition"):
        v = cond.attrib.get("value")
        try:
            f = float(v)
            if f > (sim_cond or 0):
                sim_cond = f
        except ValueError:
            pass
    if sim_cond is not None:
        out["__sim_time"] = sim_cond
    return out


def _detect_template(xosc_path: Path, root: ET.Element) -> str:
    name = xosc_path.name
    if "cut_in_conflict" in name:
        return "cut_in_conflict"
    if "close_following" in name:
        return "close_following"
    if re.search(r"\d+LC", name):
        return "consecutive_lane_change"
    if root.find(".//LaneChangeAction") is not None:
        return "consecutive_lane_change"
    return "consecutive_lane_change"


def _parse_lc_triggers(root: ET.Element) -> list[float]:
    triggers = []
    for act in root.iter("Act"):
        if not act.attrib.get("name", "").startswith("LC_"):
            continue
        for cond in act.iter("SimulationTimeCondition"):
            try:
                v = float(cond.attrib["value"])
                if v > 0.0:
                    triggers.append(v)
            except ValueError:
                continue
    return sorted(triggers)


def _lane_at_consecutive(t: float, triggers: list[float], T_LC: float,
                          num_lc: int) -> float:
    """Sinusoidal LC profile matched to the xosc LaneChangeAction trigger list."""
    y = 0.0
    for i, t_start in enumerate(triggers[:num_lc]):
        t_end = t_start + T_LC
        y_from = i * LANE_WIDTH
        y_to = (i + 1) * LANE_WIDTH
        if t < t_start:
            return y_from if i == 0 else y_to if y == y_to else y
        if t >= t_end:
            y = y_to
            continue
        u = (t - t_start) / T_LC
        return y_from + (y_to - y_from) * 0.5 * (1.0 - math.cos(math.pi * u))
    return num_lc * LANE_WIDTH


def _style_ax(ax, title: str):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _draw_road(ax, x_center: float):
    x_min = x_center - VIEW_BEHIND
    x_max = x_center + VIEW_AHEAD
    artists = []
    road = Rectangle((x_min, -LANE_WIDTH / 2 - 0.6),
                     x_max - x_min, NUM_LANES * LANE_WIDTH + 1.2,
                     facecolor="#3a3a3a", zorder=0)
    ax.add_patch(road)
    artists.append(road)
    for k in range(NUM_LANES + 1):
        y = -LANE_WIDTH / 2 + k * LANE_WIDTH
        if k == 0 or k == NUM_LANES:
            ln, = ax.plot([x_min, x_max], [y, y], color="white",
                          linewidth=2, zorder=1)
            artists.append(ln)
        else:
            for xs in np.arange(math.floor(x_min / 6) * 6, x_max, 6):
                ln, = ax.plot([xs, xs + 3], [y, y], color="white",
                              linewidth=1.2, zorder=1)
                artists.append(ln)
    ax.set_xlim(x_min, x_max)
    return artists


def _draw_vehicle(ax, x, y, color, label=None):
    rect = Rectangle((x - 2.25, y - 1.0), 4.5, 2.0,
                     facecolor=color, edgecolor="black", linewidth=0.8, zorder=5)
    ax.add_patch(rect)
    artists = [rect]
    if label:
        t = ax.annotate(label, xy=(x, y + 1.2), ha="center", va="bottom",
                        fontsize=7, zorder=6)
        artists.append(t)
    return artists


def _overlay(ax, lines: list[str]):
    text = "\n".join(lines)
    t = ax.text(0.02, 0.95, text, transform=ax.transAxes,
                fontsize=8.5, color="white", va="top",
                bbox=dict(facecolor="black", alpha=0.55,
                          edgecolor="none", boxstyle="round,pad=0.3"))
    return t


def render_consecutive_lc(params: dict, triggers: list[float], out_path: Path,
                           tag: str) -> Path:
    ego_v = float(params["ego_init_speed"])
    T_LC = float(params["lc_duration"])
    num_lc = len(triggers)
    sim_time = float(params.get("__sim_time", 40.0))
    times = np.arange(0, sim_time + 1 / FPS, 1 / FPS)

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    fig.patch.set_facecolor("#f6f6f6")
    _style_ax(ax, f"{tag} -- {num_lc} consecutive LC   "
                   f"(T_LC={T_LC:.2f}s, v0={ego_v:.1f} m/s, sim={sim_time:.1f}s)")
    ax.set_ylim(-LANE_WIDTH, NUM_LANES * LANE_WIDTH)

    state = {"artists": []}

    def draw(i):
        for a in state["artists"]:
            a.remove()
        state["artists"] = []
        t = min(times[i], sim_time)
        ego_x = ego_v * t
        ego_y = _lane_at_consecutive(t, triggers, T_LC, num_lc)
        state["artists"].extend(_draw_road(ax, ego_x))
        state["artists"].extend(_draw_vehicle(ax, ego_x, ego_y, "#2ecc71", "EGO"))
        state["artists"].append(_overlay(ax, [
            f"t = {t:5.2f} s",
            f"lane = {int(round(ego_y / LANE_WIDTH))} / {num_lc}",
            f"triggers = {', '.join(f'{x:.1f}' for x in triggers)}",
        ]))
        return state["artists"]

    anim = FuncAnimation(fig, draw, frames=len(times),
                         interval=1000 / FPS, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out_path


def render_cut_in(params: dict, out_path: Path, tag: str) -> Path:
    ego_v = float(params["ego_init_speed"])
    cutin_dhw = float(params["target_cutin_dhw"])
    pet = float(params["pet_threshold"])
    delta_v = float(params["delta_v"])
    T_LC = float(params["lc_duration"])
    t_cut = float(params["cutin_trigger_time"])
    sim_time = float(params.get("__sim_time", 40.0))
    times = np.arange(0, sim_time + 1 / FPS, 1 / FPS)

    target_v = ego_v - delta_v
    ego_lane_idx = 1
    tgt_start_lane_idx = 0

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    fig.patch.set_facecolor("#f6f6f6")
    _style_ax(ax, f"{tag} -- cut-in conflict   "
                   f"(DHW={cutin_dhw:.1f} m, PET={pet:.2f} s, "
                   f"\u0394v={delta_v:.2f}, T_LC={T_LC:.2f}s)")
    ax.set_ylim(-LANE_WIDTH, NUM_LANES * LANE_WIDTH)

    state = {"artists": []}

    def target_pos(t):
        x = cutin_dhw + target_v * t
        if t < t_cut:
            y = tgt_start_lane_idx * LANE_WIDTH
            active = False
        elif t < t_cut + T_LC:
            u = (t - t_cut) / T_LC
            y0 = tgt_start_lane_idx * LANE_WIDTH
            y1 = ego_lane_idx * LANE_WIDTH
            y = y0 + (y1 - y0) * 0.5 * (1.0 - math.cos(math.pi * u))
            active = True
        else:
            y = ego_lane_idx * LANE_WIDTH
            active = True
        return x, y, active

    def draw(i):
        for a in state["artists"]:
            a.remove()
        state["artists"] = []
        t = min(times[i], sim_time)
        ego_x = ego_v * t
        ego_y = ego_lane_idx * LANE_WIDTH
        state["artists"].extend(_draw_road(ax, ego_x))
        tx, ty, active = target_pos(t)
        color = "#e74c3c" if active else "#e67e22"
        state["artists"].extend(_draw_vehicle(ax, tx, ty, color, "Target"))
        state["artists"].extend(_draw_vehicle(ax, ego_x, ego_y, "#2ecc71", "EGO"))
        state["artists"].append(_overlay(ax, [
            f"t = {t:5.2f} s   trigger @ {t_cut:.1f}s",
            f"\u0394x = {tx - ego_x:+.1f} m",
            f"DHW={cutin_dhw:.1f}m  PET={pet:.2f}s  \u0394v={delta_v:.2f}",
        ]))
        return state["artists"]

    anim = FuncAnimation(fig, draw, frames=len(times),
                         interval=1000 / FPS, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out_path


def render_close_following(params: dict, out_path: Path, tag: str) -> Path:
    ego_v = float(params["ego_init_speed"])
    lead_gap = float(params["lead_gap"])
    ttc = float(params.get("ttc_target", 2.5))
    delta_v = float(params["delta_v"])
    t_brake = float(params["brake_trigger_time"])
    decel = float(params["brake_deceleration"])
    sim_time = float(params.get("__sim_time", 40.0))
    times = np.arange(0, sim_time + 1 / FPS, 1 / FPS)

    lead_v0 = ego_v - delta_v
    lane_idx = 1

    fig, ax = plt.subplots(figsize=(10, 3.2), dpi=100)
    fig.patch.set_facecolor("#f6f6f6")
    _style_ax(ax, f"{tag} -- close-following   "
                   f"(headway={lead_gap:.1f}m, v_ego={ego_v:.1f}, "
                   f"v_lead0={lead_v0:.1f}, brake@{t_brake:.0f}s, "
                   f"-{decel:.1f} m/s\u00b2)")
    ax.set_ylim(-LANE_WIDTH, NUM_LANES * LANE_WIDTH)

    def lead_kinematics(t):
        if t < t_brake:
            x = lead_gap + lead_v0 * t
            v = lead_v0
        else:
            dt = t - t_brake
            v_stop_t = lead_v0 / decel
            if dt >= v_stop_t:
                x_at_stop = lead_gap + lead_v0 * t_brake + 0.5 * lead_v0 * v_stop_t
                x = x_at_stop
                v = 0.0
            else:
                v = lead_v0 - decel * dt
                x = lead_gap + lead_v0 * t_brake + lead_v0 * dt - 0.5 * decel * dt * dt
        return x, v

    state = {"artists": []}

    def draw(i):
        for a in state["artists"]:
            a.remove()
        state["artists"] = []
        t = min(times[i], sim_time)
        ego_x = ego_v * t
        ego_y = lane_idx * LANE_WIDTH
        lead_x, lead_v = lead_kinematics(t)
        lead_y = lane_idx * LANE_WIDTH
        gap = lead_x - ego_x
        rel_v = ego_v - lead_v
        cur_ttc = gap / rel_v if rel_v > 0.01 else float("inf")

        state["artists"].extend(_draw_road(ax, ego_x))
        state["artists"].extend(_draw_vehicle(ax, lead_x, lead_y,
                                               "#4a90e2", "Lead"))
        state["artists"].extend(_draw_vehicle(ax, ego_x, ego_y,
                                               "#2ecc71", "EGO"))
        ttc_str = f"{cur_ttc:.2f}" if math.isfinite(cur_ttc) else "inf"
        state["artists"].append(_overlay(ax, [
            f"t = {t:5.2f} s",
            f"gap = {gap:5.2f} m   v_lead = {lead_v:5.2f}",
            f"TTC = {ttc_str} s  (target={ttc:.2f})",
        ]))
        return state["artists"]

    anim = FuncAnimation(fig, draw, frames=len(times),
                         interval=1000 / FPS, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out_path


def render_xosc(xosc_path: Path, out_path: Path | None = None,
                tag: str | None = None) -> Path:
    params = _parse_params(xosc_path)
    root = ET.parse(xosc_path).getroot()
    template = _detect_template(xosc_path, root)

    if out_path is None:
        out_path = xosc_path.with_suffix(".gif")
    tag = tag or xosc_path.stem

    if template == "consecutive_lane_change":
        triggers = _parse_lc_triggers(root)
        return render_consecutive_lc(params, triggers, out_path, tag)
    if template == "cut_in_conflict":
        return render_cut_in(params, out_path, tag)
    if template == "close_following":
        return render_close_following(params, out_path, tag)
    raise ValueError(f"Unknown template for {xosc_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xosc", required=True,
                        help="Path to input .xosc (or directory of .xosc)")
    parser.add_argument("--out", default=None,
                        help="Output .gif path (default: same stem as xosc)")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args(argv)

    p = Path(args.xosc)
    if p.is_dir():
        for x in sorted(p.rglob("*.xosc")):
            out = x.with_suffix(".gif")
            print(f"[render_xosc] {x.name} -> {out.name}")
            render_xosc(x, out, tag=x.stem)
    else:
        out = Path(args.out) if args.out else None
        r = render_xosc(p, out, tag=args.tag)
        print(f"[render_xosc] wrote {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
