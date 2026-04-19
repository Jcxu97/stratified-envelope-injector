"""Render top-down animation of the sampled CN vs DE consecutive-LC scenarios
to a GIF file, mimicking the visual style of Chat2Scenario's demo.gif but
driven entirely by this module's own sampled envelopes (no esmini / CarMaker
dependency).

The animation shows ego performing ``num_lc`` consecutive lane changes using
the sampled T_LC / lat_v_max profile, with a lead vehicle, a lag vehicle, and
a cut-in vehicle placed according to ``lead_gap`` / ``lag_gap`` / ``cutin_dhw``.
Because CN and DE sample from different envelopes, the two panels show the
same NL scenario executed with region-specific geometry (tighter gaps and
faster lateral motion in CN, wider gaps and gentler lateral motion in DE).

Run
---
python -m region_envelope_injector.gif_generator \
    --summary region_envelope_injector/demo/output/demo_summary.json \
    --out region_envelope_injector/demo/output/scenario_CN_vs_DE.gif
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

LANE_WIDTH = 3.5
NUM_LANES = 5
FPS = 15
DWELL_BETWEEN_LC = 1.5
VIEW_AHEAD = 60.0
VIEW_BEHIND = 40.0


@dataclass
class ScenarioParams:
    region: str
    stratum: str
    ego_init_speed: float
    lat_v_max: float
    lat_a_max: float
    T_LC: float
    cutin_dhw: float
    lag_gap: float
    lead_gap: float
    pet: float
    delta_v: float
    num_lc: int = 4

    @classmethod
    def from_summary(cls, s: dict, region: str, num_lc: int = 4) -> "ScenarioParams":
        block = s[region]
        p = block["sampled_params"]
        return cls(
            region=region,
            stratum=block["stratum"],
            ego_init_speed=p["ego_init_speed"],
            lat_v_max=p["lat_v_max"],
            lat_a_max=p["lat_a_max"],
            T_LC=p["T_LC"],
            cutin_dhw=p["cutin_dhw"],
            lag_gap=p["lag_gap"],
            lead_gap=p["lead_gap"],
            pet=p["pet"],
            delta_v=p["delta_v"],
            num_lc=num_lc,
        )

    @property
    def total_time(self) -> float:
        return 2.0 + self.num_lc * self.T_LC + (self.num_lc - 1) * DWELL_BETWEEN_LC + 3.0

    def lane_at(self, t: float) -> float:
        """Lateral offset (m) as a function of time, using sinusoidal profile."""
        start = 2.0
        y = 0.0
        for i in range(self.num_lc):
            lc_start = start + i * (self.T_LC + DWELL_BETWEEN_LC)
            lc_end = lc_start + self.T_LC
            y_from = i * LANE_WIDTH
            y_to = (i + 1) * LANE_WIDTH
            if t < lc_start:
                return y_from
            if t >= lc_end:
                y = y_to
                continue
            u = (t - lc_start) / self.T_LC
            return y_from + (y_to - y_from) * 0.5 * (1.0 - math.cos(math.pi * u))
        return self.num_lc * LANE_WIDTH


def _ego_x(t: float, v: float) -> float:
    return v * t


def _lead_traj(p: ScenarioParams, t: float) -> tuple[float, float]:
    """Lead vehicle stays in ego's starting lane ahead of ego."""
    x = _ego_x(t, p.ego_init_speed) + p.lead_gap
    return x, 0.0


def _lag_traj(p: ScenarioParams, t: float) -> tuple[float, float]:
    """Lag vehicle stays one lane up (first target lane) behind ego."""
    x = _ego_x(t, p.ego_init_speed) - p.lag_gap
    return x, LANE_WIDTH


def _cutin_traj(p: ScenarioParams, t: float) -> tuple[float, float, bool]:
    """Cut-in vehicle starts ahead in an adjacent lane, then cuts across at a
    defined time, arriving in ego's *current* lane at cut-in time.
    """
    cutin_time = 2.0 + p.T_LC * 2.0
    cutin_duration = min(p.T_LC, 3.0)
    approach_v = p.ego_init_speed - p.delta_v
    target_lane_idx = 2
    start_lane_idx = 3
    if t < cutin_time:
        x = _ego_x(t, p.ego_init_speed) + p.cutin_dhw + approach_v * (t - cutin_time)
        y = start_lane_idx * LANE_WIDTH
        active = False
    elif t < cutin_time + cutin_duration:
        u = (t - cutin_time) / cutin_duration
        y0 = start_lane_idx * LANE_WIDTH
        y1 = target_lane_idx * LANE_WIDTH
        y = y0 + (y1 - y0) * 0.5 * (1.0 - math.cos(math.pi * u))
        x = _ego_x(t, p.ego_init_speed) + p.cutin_dhw + approach_v * (t - cutin_time)
        active = True
    else:
        y = target_lane_idx * LANE_WIDTH
        x = _ego_x(t, p.ego_init_speed) + p.cutin_dhw + approach_v * (t - cutin_time)
        active = True
    return x, y, active


def _ambient_vehicles(p: ScenarioParams) -> list[tuple[float, int]]:
    """Scatter a few ambient cars at fixed longitudinal positions relative to
    the global frame (they do not move relative to the ground; ego overtakes
    them). Returns list of (x_global, lane_idx).
    """
    rng = np.random.default_rng(hash(p.region) & 0xFFFF)
    cars = []
    for _ in range(8):
        x = rng.uniform(-30, 200)
        lane = int(rng.integers(0, NUM_LANES))
        cars.append((x, lane))
    return cars


def _draw_vehicle(ax, x, y, color, label=None, width=4.5, height=2.0):
    rect = Rectangle((x - width / 2, y - height / 2), width, height,
                     facecolor=color, edgecolor="black", linewidth=0.8, zorder=5)
    ax.add_patch(rect)
    if label:
        ax.annotate(label, xy=(x, y + height / 2 + 0.4), ha="center", va="bottom",
                    fontsize=7, zorder=6)
    return rect


def _draw_lanes(ax, x_center: float):
    x_min = x_center - VIEW_BEHIND
    x_max = x_center + VIEW_AHEAD
    shoulder_w = 0.6
    ax.add_patch(Rectangle((x_min, -LANE_WIDTH / 2 - shoulder_w), x_max - x_min,
                           NUM_LANES * LANE_WIDTH + 2 * shoulder_w,
                           facecolor="#3a3a3a", zorder=0))
    for k in range(NUM_LANES + 1):
        y = -LANE_WIDTH / 2 + k * LANE_WIDTH
        if k == 0 or k == NUM_LANES:
            ax.plot([x_min, x_max], [y, y], color="white", linewidth=2, zorder=1)
        else:
            for xs in np.arange(x_min, x_max, 6):
                ax.plot([xs, xs + 3], [y, y], color="white", linewidth=1, zorder=1)


def _build_axis(ax, title: str):
    ax.set_xlim(-VIEW_BEHIND, VIEW_AHEAD)
    ax.set_ylim(-LANE_WIDTH, NUM_LANES * LANE_WIDTH)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=11, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _frame_times(total_time: float, fps: int = FPS) -> np.ndarray:
    return np.arange(0.0, total_time + 1 / fps, 1 / fps)


def render_gif(summary: dict, out_path: Path, num_lc: int = 4) -> Path:
    cn = ScenarioParams.from_summary(summary, "CN", num_lc=num_lc)
    de = ScenarioParams.from_summary(summary, "DE", num_lc=num_lc)

    total_time = max(cn.total_time, de.total_time)
    times = _frame_times(total_time)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), dpi=100,
                             gridspec_kw={"hspace": 0.22})
    fig.patch.set_facecolor("#f6f6f6")

    def setup(ax, p, tag):
        _build_axis(ax, f"{tag} (stratum {p.stratum}): "
                        f"v0={p.ego_init_speed:.1f} m/s, T_LC={p.T_LC:.2f} s, "
                        f"lat_v={p.lat_v_max:.2f}, DHW={p.cutin_dhw:.1f}, "
                        f"PET={p.pet:.2f}, Δv={p.delta_v:.2f}")

    setup(axes[0], cn, "CN (AD4CHE)")
    setup(axes[1], de, "DE (highD)")

    ambient_cn = _ambient_vehicles(cn)
    ambient_de = _ambient_vehicles(de)

    state = {"artists": []}

    def draw_frame(i):
        for art in state["artists"]:
            art.remove()
        state["artists"] = []

        t = times[i]
        for ax, p, ambient in [(axes[0], cn, ambient_cn), (axes[1], de, ambient_de)]:
            if t > p.total_time:
                t_eff = p.total_time
            else:
                t_eff = t

            ego_x = _ego_x(t_eff, p.ego_init_speed)
            ego_y = p.lane_at(t_eff)
            x_min = ego_x - VIEW_BEHIND
            x_max = ego_x + VIEW_AHEAD

            ax.set_xlim(x_min, x_max)

            road = Rectangle((x_min, -LANE_WIDTH / 2 - 0.6),
                             x_max - x_min,
                             NUM_LANES * LANE_WIDTH + 1.2,
                             facecolor="#3a3a3a", zorder=0)
            ax.add_patch(road)
            state["artists"].append(road)
            for k in range(NUM_LANES + 1):
                y = -LANE_WIDTH / 2 + k * LANE_WIDTH
                if k == 0 or k == NUM_LANES:
                    ln, = ax.plot([x_min, x_max], [y, y],
                                  color="white", linewidth=2, zorder=1)
                    state["artists"].append(ln)
                else:
                    for xs in np.arange(math.floor(x_min / 6) * 6, x_max, 6):
                        ln, = ax.plot([xs, xs + 3], [y, y],
                                      color="white", linewidth=1.2, zorder=1)
                        state["artists"].append(ln)

            for (ax_global, lane) in ambient:
                y_a = lane * LANE_WIDTH
                if abs(ax_global - ego_x) < VIEW_AHEAD + 5:
                    r = _draw_vehicle(ax, ax_global, y_a, color="#8c8c8c")
                    state["artists"].append(r)

            lead_x, lead_y = _lead_traj(p, t_eff)
            state["artists"].append(_draw_vehicle(ax, lead_x, lead_y,
                                                  color="#4a90e2", label="lead"))

            lag_x, lag_y = _lag_traj(p, t_eff)
            state["artists"].append(_draw_vehicle(ax, lag_x, lag_y,
                                                  color="#9b59b6", label="lag"))

            cx, cy, c_active = _cutin_traj(p, t_eff)
            if abs(cx - ego_x) < VIEW_AHEAD + 10:
                color = "#e74c3c" if c_active else "#e67e22"
                state["artists"].append(_draw_vehicle(ax, cx, cy,
                                                      color=color, label="cut-in"))

            state["artists"].append(_draw_vehicle(ax, ego_x, ego_y,
                                                  color="#2ecc71", label="EGO"))

            tlabel = ax.text(0.02, 0.92, f"t = {t_eff:5.2f} s",
                             transform=ax.transAxes,
                             fontsize=9, color="white",
                             bbox=dict(facecolor="black", alpha=0.55,
                                       edgecolor="none", boxstyle="round,pad=0.25"))
            state["artists"].append(tlabel)

        return state["artists"]

    fig.suptitle("region_envelope_injector: CN vs DE consecutive LC "
                 "(same NL, region-specific envelopes)",
                 fontsize=12, y=0.98)
    anim = FuncAnimation(fig, draw_frame, frames=len(times),
                         interval=1000 / FPS, blit=False)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = PillowWriter(fps=FPS)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    return out_path


def render_single_region(summary: dict, region: str,
                          out_path: Path, num_lc: int = 4) -> Path:
    p = ScenarioParams.from_summary(summary, region, num_lc=num_lc)
    times = _frame_times(p.total_time)

    fig, ax = plt.subplots(figsize=(10.5, 3.4), dpi=100)
    fig.patch.set_facecolor("#f6f6f6")
    _build_axis(ax, f"{region} stratum {p.stratum} | v0={p.ego_init_speed:.1f}  "
                    f"T_LC={p.T_LC:.2f}  DHW={p.cutin_dhw:.1f}  PET={p.pet:.2f}")

    ambient = _ambient_vehicles(p)
    state = {"artists": []}

    def draw_frame(i):
        for art in state["artists"]:
            art.remove()
        state["artists"] = []
        t_eff = min(times[i], p.total_time)
        ego_x = _ego_x(t_eff, p.ego_init_speed)
        ego_y = p.lane_at(t_eff)
        ax.set_xlim(ego_x - VIEW_BEHIND, ego_x + VIEW_AHEAD)

        x_min = ego_x - VIEW_BEHIND
        x_max = ego_x + VIEW_AHEAD
        road = Rectangle((x_min, -LANE_WIDTH / 2 - 0.6), x_max - x_min,
                         NUM_LANES * LANE_WIDTH + 1.2,
                         facecolor="#3a3a3a", zorder=0)
        ax.add_patch(road)
        state["artists"].append(road)
        for k in range(NUM_LANES + 1):
            y = -LANE_WIDTH / 2 + k * LANE_WIDTH
            if k == 0 or k == NUM_LANES:
                ln, = ax.plot([x_min, x_max], [y, y],
                              color="white", linewidth=2, zorder=1)
                state["artists"].append(ln)
            else:
                for xs in np.arange(math.floor(x_min / 6) * 6, x_max, 6):
                    ln, = ax.plot([xs, xs + 3], [y, y],
                                  color="white", linewidth=1.2, zorder=1)
                    state["artists"].append(ln)

        for (ax_global, lane) in ambient:
            y_a = lane * LANE_WIDTH
            if abs(ax_global - ego_x) < VIEW_AHEAD + 5:
                state["artists"].append(_draw_vehicle(ax, ax_global, y_a,
                                                      color="#8c8c8c"))

        lead_x, lead_y = _lead_traj(p, t_eff)
        state["artists"].append(_draw_vehicle(ax, lead_x, lead_y,
                                              color="#4a90e2", label="lead"))
        lag_x, lag_y = _lag_traj(p, t_eff)
        state["artists"].append(_draw_vehicle(ax, lag_x, lag_y,
                                              color="#9b59b6", label="lag"))
        cx, cy, c_active = _cutin_traj(p, t_eff)
        color = "#e74c3c" if c_active else "#e67e22"
        state["artists"].append(_draw_vehicle(ax, cx, cy, color=color,
                                              label="cut-in"))
        state["artists"].append(_draw_vehicle(ax, ego_x, ego_y,
                                              color="#2ecc71", label="EGO"))
        tlabel = ax.text(0.02, 0.88, f"t = {t_eff:5.2f} s",
                         transform=ax.transAxes, fontsize=9, color="white",
                         bbox=dict(facecolor="black", alpha=0.55,
                                   edgecolor="none", boxstyle="round,pad=0.25"))
        state["artists"].append(tlabel)
        return state["artists"]

    anim = FuncAnimation(fig, draw_frame, frames=len(times),
                         interval=1000 / FPS, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", required=True,
                        help="Path to demo_summary.json (output of run_demo.py)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (defaults to summary's parent)")
    parser.add_argument("--num-lc", type=int, default=4)
    parser.add_argument("--single", action="store_true",
                        help="Also emit per-region GIFs")
    args = parser.parse_args(argv)

    summary_path = Path(args.summary)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir) if args.out_dir else summary_path.parent

    combined = out_dir / "scenario_CN_vs_DE.gif"
    print(f"[gif_generator] Rendering combined CN vs DE -> {combined}")
    render_gif(summary, combined, num_lc=args.num_lc)

    if args.single:
        for region in ("CN", "DE"):
            out = out_dir / f"scenario_{region}_4LC.gif"
            print(f"[gif_generator] Rendering single {region} -> {out}")
            render_single_region(summary, region, out, num_lc=args.num_lc)
    print("[gif_generator] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
