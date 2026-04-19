"""Inject constant-speed ambient ScenarioObjects into a just-emitted .xosc
from a sampled :class:`DatasetClip` snapshot.

Note on 3D models: esmini v3.0.2 ignores the ``model_id`` Property and
looks up visuals by the Vehicle ``name`` attribute, which in practice
maps ``car_white`` → ``car_white.osgb`` and everything else → ``car_red``
(its fallback). To actually get colour variety we must supply an explicit
``model3d`` absolute path pointing at the files shipped in
``<ESMINI_ROOT>/resources/models/``.

Motivation
----------
The base templates only declare an ``Ego`` actor (and optionally ``Target`` /
``Lead`` for conflict templates). esmini thus renders a single car on the
highway in the 3D preview. Chat2Scenario-style 3D requires dataset traffic to
be embedded directly in the xosc. This module post-processes the xosc XML and
appends:

* ``<ScenarioObject name="Ambient_N">`` entries before ``</Entities>``
* ``<Private entityRef="Ambient_N">`` blocks (teleport + AbsoluteTargetSpeed)
  before ``</Actions>`` inside ``<Init>``

Each ambient is given a lane-based initial position (``LanePosition`` using
roadId=1 and laneId in ``[-1 .. -num_lanes]``) and a constant absolute speed.
This is enough for esmini's default controller to simulate constant-velocity
highway flow behind/ahead of the scripted Ego.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import numpy as np

from region_envelope_injector.dataset_loader import DatasetClip


def _resolve_esmini_models_dir() -> Path | None:
    """Best-effort lookup of ``<ESMINI_ROOT>/resources/models``. Mirrors
    :func:`esmini_renderer.locate_esmini_root` but does not raise — the
    caller falls back to name-based dispatch if the folder is missing."""
    candidates = []
    env = os.environ.get("ESMINI_ROOT")
    if env:
        candidates.append(Path(env))
    candidates += [
        Path("C:/Users/82077/Desktop/STLA/tools/esmini"),
        Path("/opt/esmini"),
    ]
    for root in candidates:
        models = root / "resources" / "models"
        if models.is_dir():
            return models
    return None

AMBIENT_VEHICLE_XML = """        <ScenarioObject name="{name}">
            <Vehicle name="{model_name}" vehicleCategory="car"{model3d_attr}>
                <BoundingBox>
                    <Center x="1.4" y="0.0" z="0.75"/>
                    <Dimensions width="1.8" length="4.5" height="1.5"/>
                </BoundingBox>
                <Performance maxSpeed="60" maxDeceleration="9.5" maxAcceleration="4.0"/>
                <Axles>
                    <FrontAxle maxSteering="0.5" wheelDiameter="0.65" trackWidth="1.6" positionX="2.7" positionZ="0.325"/>
                    <RearAxle  maxSteering="0.0" wheelDiameter="0.65" trackWidth="1.6" positionX="0.0" positionZ="0.325"/>
                </Axles>
                <Properties>
                    <Property name="model_id" value="{model_id}"/>
                </Properties>
            </Vehicle>
        </ScenarioObject>
"""

# Cycle through esmini's built-in car models so the 3D preview shows a
# real car mesh per ambient (otherwise esmini falls back to a grey
# bounding-box cuboid, which the user sees as "a brick, not a car").
_AMBIENT_MODELS = [
    (1, "car_blue"),
    (3, "car_yellow"),
    (2, "car_red"),
    (0, "car_white"),
]

AMBIENT_INIT_XML = """                <Private entityRef="{name}">
                    <PrivateAction>
                        <TeleportAction>
                            <Position>
                                <LanePosition roadId="1" laneId="{lane_id}" offset="0.0" s="{s:.2f}"/>
                            </Position>
                        </TeleportAction>
                    </PrivateAction>
                    <PrivateAction>
                        <LongitudinalAction>
                            <SpeedAction>
                                <SpeedActionDynamics dynamicsShape="step" value="0" dynamicsDimension="time"/>
                                <SpeedActionTarget>
                                    <AbsoluteTargetSpeed value="{speed:.2f}"/>
                                </SpeedActionTarget>
                            </SpeedAction>
                        </LongitudinalAction>
                    </PrivateAction>
                </Private>
"""


def _assign_lane(y: float, lane_ys: list[float], num_lanes: int,
                  ego_lane_id: int = -1) -> int:
    """Return esmini laneId in ``[-1..-num_lanes]`` EXCLUDING ``ego_lane_id``
    so ambient actors never share ego's lane (which would otherwise cause
    rear-end collisions in the 3D preview once ego catches up)."""
    if not lane_ys:
        return -2
    sorted_ys = sorted(lane_ys)
    idx = int(np.argmin([abs(y - ly) for ly in sorted_ys]))
    lane_id = -(idx + 1) if idx < num_lanes else -num_lanes
    if lane_id == ego_lane_id:
        # Bump to the next available lane (prefer inner lane -2 if ego=-1).
        lane_id = -2 if ego_lane_id == -1 else -1
    return lane_id


def _sample_ambient(clip: DatasetClip,
                     num_lanes: int,
                     road_length: float,
                     n: int,
                     ego_speed: float = 20.0,
                     ego_lane_id: int = -1,
                     ego_s: float = 0.0,
                     window_behind: float = 25.0,
                     window_ahead: float = 70.0,
                     ) -> list[dict]:
    """Pick up to ``n`` ambient vehicles and place them UNIFORMLY spread
    in a tight window around ego (``[ego_s - window_behind, ego_s + window_ahead]``)
    so the 3D camera actually sees cars next to ego (not only at the far horizon).

    We keep each dataset car's recorded lane (snapped) and speed, but we
    replace its longitudinal position with an evenly-spaced s along the
    ego-centred window. Reason: the dataset clip's x-span is several hundred
    metres and all cars have similar x_norm, so using x_norm for s collapsed
    the ambients into a narrow horizon cluster regardless of ``window``.
    Uniform distribution along ``[s_lo, s_hi]`` guarantees visible cars
    roughly every ``window/n`` metres — including within 25 m of ego.
    """
    snap = clip.at(0.0)
    if snap.empty:
        return []
    min_amb_v = ego_speed + 5.0
    # Per-lane staggered s schedule. Build a list of (lane_id, s_slot) so
    # cars naturally spread across every non-ego lane AND along the window.
    non_ego_lanes = [lid for lid in range(-1, -num_lanes - 1, -1)
                     if lid != ego_lane_id]
    if not non_ego_lanes:
        non_ego_lanes = [-2]
    s_lo = max(5.0, ego_s - window_behind)
    s_hi = min(road_length - 10.0, ego_s + window_ahead)
    # Even-count grid of slots: `cells_per_lane = ceil(n / len(non_ego_lanes))`,
    # then stagger s by half-step between adjacent lanes so no two cars share s.
    per_lane = max(1, (n + len(non_ego_lanes) - 1) // len(non_ego_lanes))
    s_step = (s_hi - s_lo) / max(per_lane, 1)
    slots: list[tuple[int, float]] = []
    for li, lid in enumerate(non_ego_lanes):
        offset = s_step * 0.5 * (li / max(len(non_ego_lanes) - 1, 1))
        for k in range(per_lane):
            s = s_lo + offset + k * s_step
            if s_lo <= s <= s_hi:
                slots.append((lid, s))
    # Pair each slot with a dataset car's (vx) — cycle the snapshot.
    out: list[dict] = []
    rows = list(snap.iterrows())
    if not rows:
        return []
    for i, (lid, s) in enumerate(slots[:n]):
        _, row = rows[i % len(rows)]
        vx = abs(float(row.get("vx", 20.0)))
        if vx < min_amb_v:
            vx = min_amb_v
        out.append({"lane_id": lid, "s": float(s), "speed": vx})
    # Defensive dedup (same-lane cars within 12 m are likely to overlap
    # visually and trip esmini's collision detector).
    out.sort(key=lambda d: (d["lane_id"], d["s"]))
    filtered: list[dict] = []
    for a in out:
        if any(f["lane_id"] == a["lane_id"] and abs(f["s"] - a["s"]) < 12.0
               for f in filtered):
            continue
        filtered.append(a)
    return filtered


def _parse_ego_spawn(xosc_text: str) -> tuple[int, float]:
    """Return ``(ego_lane_id, ego_s)`` read directly from the xosc's Init
    block so the ambient sampler knows where ego actually starts. Falls
    back to ``(-1, 0.0)`` if the scan fails (matches template defaults)."""
    m = re.search(
        r'<Private\s+entityRef="Ego">.*?<LanePosition[^/]*'
        r'laneId="(-?\d+)"[^/]*s="([0-9.+eE-]+)"',
        xosc_text, flags=re.DOTALL)
    if not m:
        return -1, 0.0
    return int(m.group(1)), float(m.group(2))


def _scripted_lane_positions(xosc_text: str) -> list[tuple[int, float]]:
    """Return list of ``(laneId, s)`` for every non-ego ScenarioObject that
    spawns via a ``LanePosition`` — Lead, Target, etc. Used to avoid placing
    ambient actors on top of scripted conflict vehicles."""
    out: list[tuple[int, float]] = []
    for m in re.finditer(
            r'<Private\s+entityRef="([^"]+)">.*?'
            r'<LanePosition[^/]*laneId="(-?\d+)"[^/]*s="([0-9.+eE-]+)"',
            xosc_text, flags=re.DOTALL):
        name, lane_id, s = m.group(1), int(m.group(2)), float(m.group(3))
        if name == "Ego":
            continue
        out.append((lane_id, s))
    return out


def inject_ambient(xosc_path: Path,
                    clip: DatasetClip,
                    *,
                    n_ambient: int = 10,
                    num_lanes: int = 4,
                    road_length: float = 500.0,
                    ego_speed: float = 20.0) -> Path:
    """Mutate ``xosc_path`` in-place, embedding up to ``n_ambient`` ambient
    actors sampled from ``clip`` at t=0. Returns the same path."""
    xosc_path = Path(xosc_path)
    text = xosc_path.read_text(encoding="utf-8")
    if "</Entities>" not in text or "</Actions>" not in text:
        return xosc_path

    ego_lane_id, ego_s = _parse_ego_spawn(text)
    ambients = _sample_ambient(clip, num_lanes, road_length, n_ambient,
                                 ego_speed=ego_speed,
                                 ego_lane_id=ego_lane_id,
                                 ego_s=ego_s)
    # Drop any ambient within 15m longitudinally on the same lane as a
    # scripted Lead / Target — otherwise esmini spawns them overlapping.
    scripted = _scripted_lane_positions(text)
    ambients = [a for a in ambients
                 if not any(a["lane_id"] == sl and abs(a["s"] - ss) < 15.0
                             for (sl, ss) in scripted)]
    # Also drop any ambient within EGO_S_CLEAR m of ego's own spawn cell
    # — ego is laneId=ego_lane_id at s=ego_s, we can't have ambients
    # literally on top of ego at t=0.
    EGO_S_CLEAR = 15.0
    ambients = [a for a in ambients
                 if not (a["lane_id"] == ego_lane_id
                          and abs(a["s"] - ego_s) < EGO_S_CLEAR)]
    if not ambients:
        return xosc_path

    models_dir = _resolve_esmini_models_dir()
    def _model3d_attr(model_name: str) -> str:
        if models_dir is None:
            return ""
        osgb = models_dir / f"{model_name}.osgb"
        if not osgb.is_file():
            return ""
        # Forward-slash absolute path works on both Win + Linux esmini.
        p = str(osgb).replace("\\", "/")
        return f' model3d="{p}"'

    entity_xml = "".join(
        AMBIENT_VEHICLE_XML.format(
            name=f"Ambient_{i}",
            model_id=_AMBIENT_MODELS[i % len(_AMBIENT_MODELS)][0],
            model_name=_AMBIENT_MODELS[i % len(_AMBIENT_MODELS)][1],
            model3d_attr=_model3d_attr(
                _AMBIENT_MODELS[i % len(_AMBIENT_MODELS)][1]),
        )
        for i in range(len(ambients))
    )
    init_xml = "".join(
        AMBIENT_INIT_XML.format(
            name=f"Ambient_{i}",
            lane_id=a["lane_id"],
            s=a["s"],
            speed=a["speed"],
        )
        for i, a in enumerate(ambients)
    )

    text = text.replace("</Entities>", entity_xml + "    </Entities>", 1)
    # Only the first </Actions> inside <Init> — that's the first occurrence.
    text = text.replace("</Actions>", init_xml + "            </Actions>", 1)
    xosc_path.write_text(text, encoding="utf-8")
    return xosc_path
