"""Render a scenario as a top-down GIF with REAL ambient traffic drawn from
AD4CHE / highD, plus the scripted ego maneuver from the generated .xosc.

This supersedes ``bin/render_xosc.py`` for the UI path: the xosc alone does
not describe ambient cars, so we overlay the scripted actors (Ego / Target /
Lead) on top of genuine dataset frames to make the demo look like the paper's
sample clips.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

from region_envelope_injector.dataset_loader import DatasetClip

FPS = 12
# Long scenarios (4-LC with 7.8s T_LC = 40+s sim_time) render slowly frame-by-frame.
# Cap the visible window at RENDER_CAP_S so the UI stays responsive.
RENDER_CAP_S = 20.0
VIEW_AHEAD = 80.0
VIEW_BEHIND = 50.0
LANE_WIDTH = 3.5
EGO_CLEAR_X = 5.5          # longitudinal buffer (m) cleared around scripted ego
EGO_CLEAR_Y = LANE_WIDTH * 0.55
ACTOR_CLEAR_X = 5.5        # buffer around Lead / Target
ACTOR_CLEAR_Y = LANE_WIDTH * 0.55
MIN_SCRIPTED_GAP = 5.5     # min bumper-to-bumper centre distance for Lead/Target vs ego
AMBIENT_MIN_SPACING = 6.0  # min per-lane longitudinal spacing between ambient cars


@dataclass
class XoscScenario:
    template: str
    params: dict
    lc_triggers: list[float]
    sim_time: float


def parse_xosc(xosc_path: Path) -> XoscScenario:
    root = ET.parse(xosc_path).getroot()
    params: dict = {}
    for pd in root.iter("ParameterDeclaration"):
        try:
            params[pd.attrib["name"]] = float(pd.attrib["value"])
        except ValueError:
            params[pd.attrib["name"]] = pd.attrib["value"]

    sim_time = 0.0
    for cond in root.iter("SimulationTimeCondition"):
        try:
            v = float(cond.attrib["value"])
            sim_time = max(sim_time, v)
        except ValueError:
            pass

    name = xosc_path.name
    if "cut_in_conflict" in name:
        template = "cut_in_conflict"
    elif "close_following" in name:
        template = "close_following"
    else:
        template = "consecutive_lane_change"

    triggers: list[float] = []
    if template == "consecutive_lane_change":
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
        triggers.sort()

    return XoscScenario(template=template, params=params,
                         lc_triggers=triggers, sim_time=sim_time)


def _dominant_lanes(clip: DatasetClip, k: int = 4) -> list[float]:
    """Pick the k most populated lane centrelines (AD4CHE lane count is
    inflated by merge lanes; we want the through-traffic lanes)."""
    if not clip.lane_y_centers:
        return []
    pop = clip.tracks.groupby("lane").size().sort_values(ascending=False)
    top_lanes = pop.head(k).index.tolist()
    y_by_lane = clip.tracks.groupby("lane")["y"].mean().to_dict()
    ys = sorted(float(y_by_lane[l]) for l in top_lanes if l in y_by_lane)
    return ys


def _draw_road(ax, x_min, x_max, lane_ys: list[float]):
    top = min(lane_ys) - LANE_WIDTH / 2 - 0.6
    bot = max(lane_ys) + LANE_WIDTH / 2 + 0.6
    artists = []
    road = Rectangle((x_min, top), x_max - x_min, bot - top,
                     facecolor="#3a3a3a", zorder=0)
    ax.add_patch(road)
    artists.append(road)
    edge_top = min(lane_ys) - LANE_WIDTH / 2
    edge_bot = max(lane_ys) + LANE_WIDTH / 2
    ln, = ax.plot([x_min, x_max], [edge_top, edge_top], color="white",
                   linewidth=2, zorder=1)
    artists.append(ln)
    ln, = ax.plot([x_min, x_max], [edge_bot, edge_bot], color="white",
                   linewidth=2, zorder=1)
    artists.append(ln)
    for k in range(len(lane_ys) - 1):
        y = 0.5 * (lane_ys[k] + lane_ys[k + 1])
        for xs in np.arange(math.floor(x_min / 6) * 6, x_max, 6):
            ln, = ax.plot([xs, xs + 3], [y, y], color="white",
                          linewidth=1.1, zorder=1)
            artists.append(ln)
    return artists


def _draw_vehicle(ax, x, y, color, label=None, zorder=5,
                   width=4.5, height=2.0, alpha=1.0):
    rect = Rectangle((x - width / 2, y - height / 2), width, height,
                     facecolor=color, edgecolor="black", linewidth=0.6,
                     zorder=zorder, alpha=alpha)
    ax.add_patch(rect)
    artists = [rect]
    if label:
        t = ax.annotate(label, xy=(x, y + height / 2 + 0.35),
                        ha="center", va="bottom", fontsize=7,
                        color="white", zorder=zorder + 1,
                        bbox=dict(facecolor="black", alpha=0.55,
                                   edgecolor="none", boxstyle="round,pad=0.15"))
        artists.append(t)
    return artists


def _lane_at_consecutive(t: float, triggers: list[float], T_LC: float,
                          ego_lane_idx: int, lane_ys: list[float]) -> float:
    """Map time to ego y-coord: ego starts at lane ``ego_lane_idx`` and climbs
    toward lane 0 (leftmost) one lane per trigger."""
    y = lane_ys[ego_lane_idx]
    for i, t_start in enumerate(triggers):
        t_end = t_start + T_LC
        from_idx = max(ego_lane_idx - i, 0)
        to_idx = max(ego_lane_idx - (i + 1), 0)
        if t < t_start:
            return y
        if t >= t_end:
            y = lane_ys[to_idx]
            continue
        u = (t - t_start) / T_LC
        return lane_ys[from_idx] + (lane_ys[to_idx] - lane_ys[from_idx]) * \
               0.5 * (1.0 - math.cos(math.pi * u))
    return y


def _snap_to_lane(y: float, lane_ys: list[float]) -> float:
    """Snap a raw dataset y to the closest rendered lane centre. Prevents
    drifting/off-lane ambient cars that read as "car not on the road"."""
    if not lane_ys:
        return y
    return min(lane_ys, key=lambda ly: abs(ly - y))


def _init_ambient_sim(clip: DatasetClip, lane_ys: list[float],
                       ego_lane_y: float) -> list[dict]:
    """Take ONE snapshot from the dataset (t=0), snap each car to the nearest
    rendered lane and record (x0, y, vx). From that moment on, the scene
    renderer propagates every car forward in world coordinates at its own
    recorded vx — we never re-query the dataset per frame.

    Why: the previous per-frame resampling (``clip.at(t_s)`` with
    ``clip_t = t % clip_duration``) silently swapped in a different set of
    dataset vehicles whenever the clip wrapped, which the user sees as
    "cars teleporting / disappearing". A stable, physics-propagated set
    guarantees that every car that was visible stays visible until it
    leaves the view through the edge of the window.

    Exclusions at spawn time:
      * drop any car within ``EGO_CLEAR_X`` longitudinally of ego's
        spawn (x=0) regardless of lane — ego sits in the middle of the
        screen at t=0 and overlapping sprites read as a collision;
      * drop any car in ego's spawn lane within ``EGO_LANE_CLEAR`` so
        ego gets clear longitudinal space ahead/behind at t=0 (the
        dataset's lead/follower will still populate the other lanes).
    """
    snap = clip.at(0.0)
    if snap.empty:
        return []
    y_top = min(lane_ys) - LANE_WIDTH * 0.6 if lane_ys else -1e9
    y_bot = max(lane_ys) + LANE_WIDTH * 0.6 if lane_ys else 1e9
    clip_span = max(clip.x_range[1] - clip.x_range[0], 1.0)
    EGO_LANE_CLEAR = 18.0    # 4 car lengths — enough ego is never shoulder-to-shoulder at t=0.
    cars: list[dict] = []
    for _, row in snap.iterrows():
        y_raw = float(row["y"])
        if y_raw < y_top or y_raw > y_bot:
            continue
        y = _snap_to_lane(y_raw, lane_ys)
        x_rel = float(row["x"]) - clip.x_range[0]
        x0 = x_rel - clip_span / 2  # spread around ego (ego at x=0 at t=0)
        if abs(x0) < EGO_CLEAR_X and abs(y - ego_lane_y) < LANE_WIDTH * 0.9:
            continue  # overlaps ego's bounding box or its immediate neighbours
        if abs(y - ego_lane_y) < LANE_WIDTH * 0.4 and abs(x0) < EGO_LANE_CLEAR:
            continue  # same lane as ego, too close to be safe at t=0
        vx = abs(float(row.get("vx", 20.0)))
        if vx < 5.0:
            vx = 20.0
        cars.append({
            "id": int(row["id"]),
            "x": x0,
            "y": y,
            "vx_base": vx,
            "vx_cur": vx,
            "yielding": False,
        })
    # De-overlap at t=0. Two dataset rows occasionally land on top of each
    # other after lane snap; 10 m same-lane spacing ensures 4.5 m-long car
    # sprites don't visually touch.
    cars.sort(key=lambda c: (round(c["y"], 1), c["x"]))
    dedup: list[dict] = []
    for c in cars:
        if any(abs(d["y"] - c["y"]) < LANE_WIDTH * 0.4
               and abs(d["x"] - c["x"]) < 10.0 for d in dedup):
            continue
        dedup.append(c)
    return dedup


def _advance_ambient(cars: list[dict], dt: float, ego_x: float, ego_v: float,
                      in_lc: bool, lc_source_y: float, lc_target_y: float
                      ) -> None:
    """Advance each ambient car by dt. During a lane change ego physically
    sweeps across BOTH the source lane (the one it's leaving) and the target
    lane (the one it's entering), so both lanes' nearby cars must yield:

    * target-lane car roughly alongside/ahead-of ego → decelerate so ego can
      merge in front;
    * source-lane car behind ego → decelerate too, otherwise it rear-ends
      ego before ego has cleared the source lane (previously this car would
      catch up and visually overlap ego mid-merge).

    Once the lane change completes every yielding car returns to its cruise
    speed. This replaces the old "cars vanish when ego gets close" behaviour
    with real-traffic yielding behaviour."""
    if dt <= 0:
        return
    for c in cars:
        rel = c["x"] - ego_x
        in_target_lane = in_lc and abs(c["y"] - lc_target_y) < LANE_WIDTH * 0.4
        in_source_lane = (in_lc and lc_source_y is not None
                          and abs(c["y"] - lc_source_y) < LANE_WIDTH * 0.4)
        # Target lane: yield window is from 6m behind ego up to 15m ahead
        # — ego is merging into this space.
        # Source lane: yield window is from 20m behind ego up to 2m ahead
        # — ego is still partially in this lane, the follower must stay back.
        if in_target_lane and -6.0 < rel < 15.0:
            c["vx_cur"] = max(min(c["vx_base"], ego_v - 3.0), 4.0)
            c["yielding"] = True
        elif in_source_lane and -20.0 < rel < 2.0:
            c["vx_cur"] = max(min(c["vx_base"], ego_v - 3.0), 4.0)
            c["yielding"] = True
        elif not in_lc and c["yielding"]:
            c["vx_cur"] = c["vx_base"]
            c["yielding"] = False
        c["x"] += c["vx_cur"] * dt
        # Hard geometric safety net during LC: speed-based yielding needs
        # time to open a gap, but if a car was already close to ego at LC
        # start (e.g. rel=-3m) the sprites overlap visually for a second
        # or two before the speed differential takes effect. Both the
        # source-lane follower and target-lane car therefore get a hard
        # minimum separation: anything in either lane is clamped to at
        # least ``MIN_LC_GAP`` behind ego (follower) or ahead of ego
        # (leader). This also catches cross-lane adjacency at LC start
        # that ``_init_ambient_sim`` could not fully eliminate.
        MIN_LC_GAP = 8.0
        if in_target_lane or in_source_lane:
            rel_now = c["x"] - ego_x
            if -MIN_LC_GAP < rel_now <= 0:
                c["x"] = ego_x - MIN_LC_GAP
                c["yielding"] = True
            elif 0 < rel_now < MIN_LC_GAP:
                c["x"] = ego_x + MIN_LC_GAP
                c["yielding"] = True


def render_scenario(xosc_path: Path, clip: DatasetClip, out_path: Path,
                     tag: str | None = None) -> Path:
    scenario = parse_xosc(xosc_path)
    # Render the actual number of through-lanes the dataset has, but force
    # them to equal width (raw dataset y-centres are irregular — merge
    # lanes, ramps, sensor drift — which made each lane look a different
    # width). Uniform LANE_WIDTH spacing, anchored on the dataset's mean
    # lane band so snapped cars broadly line up with their raw positions.
    if clip.lane_y_centers:
        n_lanes = max(3, min(len(clip.lane_y_centers), 6))
        y_band_center = 0.5 * (min(clip.lane_y_centers)
                                + max(clip.lane_y_centers))
    else:
        n_lanes = 4
        y_band_center = 0.0
    lane_ys = [y_band_center + (i - (n_lanes - 1) / 2) * LANE_WIDTH
                for i in range(n_lanes)]

    ego_v = float(scenario.params.get("ego_init_speed", 20.0))
    sim_time = min(scenario.sim_time or 40.0, RENDER_CAP_S)

    ego_lane_idx = len(lane_ys) - 1
    times = np.arange(0, sim_time + 1 / FPS, 1 / FPS)

    tag = tag or xosc_path.stem
    template_label = {
        "consecutive_lane_change": "Consecutive LC",
        "cut_in_conflict": "Cut-in Conflict",
        "close_following": "Close Following",
    }.get(scenario.template, scenario.template)

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=100)
    fig.patch.set_facecolor("#f6f6f6")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.90, bottom=0.22)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{tag} -- {template_label} | ambient: {clip.source} "
        f"({clip.num_vehicles} veh, {clip.frame_rate:.0f}Hz)",
        fontsize=10, pad=4,
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    y_min = min(lane_ys) - LANE_WIDTH
    y_max = max(lane_ys) + LANE_WIDTH
    ax.set_ylim(y_min, y_max)

    state = {
        "artists": [],
        "ambient_cars": _init_ambient_sim(clip, lane_ys,
                                          ego_lane_y=lane_ys[ego_lane_idx]),
        "last_t": 0.0,
    }

    T_LC_global = float(scenario.params.get("lc_duration", 4.0))

    def _ego_lc_info(t: float) -> tuple[bool, float, float]:
        """Return (in_lane_change, source_y, target_y) at time ``t``.

        ``source_y`` is the lane ego is currently LEAVING — needed so the
        renderer can make the follower in that lane yield. Without it, the
        origin-lane follower (which was perfectly spaced at t=0) catches up
        to ego during the LC sweep and visually rear-ends the ego sprite."""
        if scenario.template != "consecutive_lane_change":
            y = lane_ys[ego_lane_idx]
            return False, y, y
        for idx, t_start in enumerate(scenario.lc_triggers):
            t_end = t_start + T_LC_global
            if t_start <= t < t_end:
                from_idx = max(ego_lane_idx - idx, 0)
                to_idx = max(ego_lane_idx - (idx + 1), 0)
                return True, lane_ys[from_idx], lane_ys[to_idx]
        y = lane_ys[ego_lane_idx]
        return False, y, y

    def draw(i):
        for a in state["artists"]:
            a.remove()
        state["artists"] = []
        t = min(times[i], sim_time)

        ego_x = ego_v * t
        if scenario.template == "consecutive_lane_change":
            ego_y = _lane_at_consecutive(
                t, scenario.lc_triggers, T_LC_global,
                ego_lane_idx, lane_ys,
            )
        else:
            ego_y = lane_ys[ego_lane_idx]

        # Propagate ambient simulation by dt. Cars only leave the view by
        # reaching the window edge — never mid-frame.
        dt = max(t - state["last_t"], 0.0)
        state["last_t"] = t
        in_lc, lc_source_y, lc_target_y = _ego_lc_info(t)
        _advance_ambient(state["ambient_cars"], dt, ego_x, ego_v,
                          in_lc, lc_source_y, lc_target_y)

        x_min = ego_x - VIEW_BEHIND
        x_max = ego_x + VIEW_AHEAD
        ax.set_xlim(x_min, x_max)
        state["artists"].extend(_draw_road(ax, x_min, x_max, lane_ys))

        scripted: list[tuple[float, float, str, str | None]] = []
        overlay_lines: list[str] = [f"t = {t:5.2f} s"]

        if scenario.template == "cut_in_conflict":
            t_cut = float(scenario.params.get("cutin_trigger_time", 6.0))
            T_LC = float(scenario.params.get("lc_duration", 4.0))
            dhw = float(scenario.params.get("target_cutin_dhw", 8.0))
            delta_v = float(scenario.params.get("delta_v", 2.0))
            tgt_v = ego_v - delta_v
            tgt_x = ego_x + dhw + tgt_v * (t - t_cut)
            start_idx = max(ego_lane_idx - 1, 0)
            if t < t_cut:
                tgt_y = lane_ys[start_idx]
                active = False
            elif t < t_cut + T_LC:
                u = (t - t_cut) / T_LC
                y0 = lane_ys[start_idx]
                y1 = lane_ys[ego_lane_idx]
                tgt_y = y0 + (y1 - y0) * 0.5 * (1.0 - math.cos(math.pi * u))
                active = True
            else:
                tgt_y = lane_ys[ego_lane_idx]
                active = True
            tgt_color = "#e74c3c" if active else "#e67e22"
            tgt_x = max(tgt_x, ego_x + MIN_SCRIPTED_GAP)
            scripted.append((tgt_x, tgt_y, tgt_color, "Target"))
            overlay_lines.append(
                f"DHW={dhw:.1f}m  PET={scenario.params.get('pet_threshold', 0):.2f}  "
                f"\u0394v={delta_v:.2f}  cut@{t_cut:.1f}s")

        elif scenario.template == "close_following":
            lead_gap = float(scenario.params.get("lead_gap", 18.0))
            delta_v = float(scenario.params.get("delta_v", 2.0))
            t_brake = float(scenario.params.get("brake_trigger_time", 5.0))
            decel = float(scenario.params.get("brake_deceleration", 4.0))
            lead_v0 = ego_v - delta_v
            if t < t_brake:
                lead_x = lead_gap + lead_v0 * t
                lead_v = lead_v0
            else:
                dt = t - t_brake
                stop_t = lead_v0 / decel if decel > 0 else 1e6
                if dt >= stop_t:
                    lead_x = lead_gap + lead_v0 * t_brake + 0.5 * lead_v0 * stop_t
                    lead_v = 0.0
                else:
                    lead_v = lead_v0 - decel * dt
                    lead_x = (lead_gap + lead_v0 * t_brake + lead_v0 * dt
                              - 0.5 * decel * dt * dt)
            lead_x = max(lead_x, ego_x + MIN_SCRIPTED_GAP)
            scripted.append((lead_x, lane_ys[ego_lane_idx], "#4a90e2", "Lead"))
            gap = lead_x - ego_x
            ttc = gap / (ego_v - lead_v) if ego_v - lead_v > 0.01 else float("inf")
            ttc_str = f"{ttc:.2f}" if math.isfinite(ttc) else "inf"
            overlay_lines.append(
                f"gap={gap:5.2f}m  v_lead={lead_v:5.2f}  TTC={ttc_str}s  "
                f"brake@{t_brake:.0f}s -{decel:.1f}m/s\u00b2")

        elif scenario.template == "consecutive_lane_change":
            T_LC = float(scenario.params.get("lc_duration", 4.0))
            overlay_lines.append(
                f"T_LC={T_LC:.2f}s  triggers="
                f"{', '.join(f'{x:.1f}' for x in scenario.lc_triggers)}")

        for c in state["ambient_cars"]:
            if x_min - 8 < c["x"] < x_max + 8:
                color = "#d9a441" if c["yielding"] else "#8c8c8c"
                state["artists"].extend(
                    _draw_vehicle(ax, c["x"], c["y"], color, zorder=3))

        for (sx, sy, scolor, slabel) in scripted:
            state["artists"].extend(
                _draw_vehicle(ax, sx, sy, scolor, label=slabel, zorder=6))

        state["artists"].extend(
            _draw_vehicle(ax, ego_x, ego_y, "#2ecc71", label="EGO", zorder=7))

        txt = fig.text(0.02, 0.08, "  |  ".join(overlay_lines),
                       fontsize=9, color="#222", va="bottom", ha="left",
                       bbox=dict(facecolor="white", alpha=0.95,
                                 edgecolor="#888", boxstyle="round,pad=0.35"))
        state["artists"].append(txt)
        return state["artists"]

    anim = FuncAnimation(fig, draw, frames=len(times),
                         interval=1000 / FPS, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out_path


def render_original_clip(clip: DatasetClip, out_path: Path,
                          duration: float = 12.0,
                          view_window: float = 180.0) -> Path:
    """Render the raw dataset clip (no scripted ego overlay) as a reference.

    Uses *all* lane centres and draws each vehicle at its raw (x, y) so the
    real lateral positions and inter-vehicle spacing are preserved. The x
    axis pans with a fixed window so the car boxes aren't crushed into
    invisibility on long recordings."""
    lane_ys = sorted(clip.lane_y_centers) if clip.lane_y_centers else [
        i * LANE_WIDTH for i in range(5)]

    times = np.arange(0, duration + 1 / FPS, 1 / FPS)

    tracks_by_t = {t: g for t, g in clip.tracks.groupby("t")}
    all_ts = sorted(tracks_by_t.keys())

    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=100)
    fig.patch.set_facecolor("#f6f6f6")
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.22)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"Original clip: {clip.source} -- "
                  f"{clip.num_vehicles} vehicles, "
                  f"{clip.frame_rate:.0f}Hz, duration={clip.duration_s:.0f}s",
                  fontsize=10, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)
    y_min = min(lane_ys) - LANE_WIDTH
    y_max = max(lane_ys) + LANE_WIDTH
    ax.set_ylim(y_min, y_max)

    x_lo_global = clip.x_range[0]
    x_hi_global = clip.x_range[1]
    x_center_fixed = 0.5 * (x_lo_global + x_hi_global)
    x_min = x_center_fixed - view_window / 2
    x_max = x_center_fixed + view_window / 2
    ax.set_xlim(x_min, x_max)

    state = {"artists": []}

    def draw(i):
        for a in state["artists"]:
            a.remove()
        state["artists"] = []
        t = min(times[i], duration)
        idx = int(t / duration * (len(all_ts) - 1))
        snapshot = tracks_by_t[all_ts[idx]]
        state["artists"].extend(_draw_road(ax, x_min, x_max, lane_ys))
        # Stable per-frame ordering by vehicle id avoids dedup flicker when
        # two cars cross-over in x. Y-band filter culls merge/shoulder cars
        # without dropping them from the upstream data.
        y_band_top = min(lane_ys) - LANE_WIDTH * 0.5
        y_band_bot = max(lane_ys) + LANE_WIDTH * 0.5
        rows = sorted(
            ((int(r["id"]), float(r["x"]), float(r["y"]))
             for _, r in snapshot.iterrows()),
            key=lambda v: v[0],
        )
        accepted: list[tuple[float, float]] = []
        for (_vid, x_w, y) in rows:
            if y < y_band_top or y > y_band_bot:
                continue
            if x_w < x_min - 5 or x_w > x_max + 5:
                continue
            if any(abs(ay - y) < LANE_WIDTH * 0.4 and abs(ax_ - x_w) < AMBIENT_MIN_SPACING
                   for (ax_, ay) in accepted):
                continue
            accepted.append((x_w, y))
            state["artists"].extend(
                _draw_vehicle(ax, x_w, y, "#8c8c8c", zorder=3,
                              width=3.6, height=1.6))
        txt = fig.text(0.02, 0.08,
                       f"{clip.source}  t = {t:5.2f} s  "
                       f"({snapshot['id'].nunique()} visible)",
                       fontsize=9, color="#222", va="bottom", ha="left",
                       bbox=dict(facecolor="white", alpha=0.95,
                                 edgecolor="#888", boxstyle="round,pad=0.35"))
        state["artists"].append(txt)
        return state["artists"]

    anim = FuncAnimation(fig, draw, frames=len(times),
                         interval=1000 / FPS, blit=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return out_path
