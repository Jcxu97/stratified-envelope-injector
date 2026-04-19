"""Emit a parametrised OpenSCENARIO 1.2 .xosc file from the template.

This is a minimal text-substitution emitter. For production use with full
road-network handling, pipe the sampled parameters into scenariogeneration
(https://github.com/pyoscx/scenariogeneration) which Chat2Scenario already
depends on; this module is sufficient to demonstrate end-to-end injection.
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

DWELL_BETWEEN_LC = 1.5
LC_START_OFFSET = 2.0
LC_TAIL_PAD = 3.0


def _render_lc_events(num_lc: int, T_LC: float,
                      dwell: float = DWELL_BETWEEN_LC) -> str:
    blocks = []
    for i in range(num_lc):
        t0 = LC_START_OFFSET + i * (T_LC + dwell)
        target_lane = -1 - (i + 1)
        blocks.append(f"""
            <Act name="LC_{i+1}">
                <ManeuverGroup maximumExecutionCount="1" name="mg_lc_{i+1}">
                    <Actors selectTriggeringEntities="false">
                        <EntityRef entityRef="Ego"/>
                    </Actors>
                    <Maneuver name="maneuver_lc_{i+1}">
                        <Event name="ev_lc_{i+1}" priority="overwrite">
                            <Action name="act_lc_{i+1}">
                                <PrivateAction>
                                    <LateralAction>
                                        <LaneChangeAction>
                                            <LaneChangeActionDynamics
                                                dynamicsShape="sinusoidal"
                                                value="$lc_duration"
                                                dynamicsDimension="time"/>
                                            <LaneChangeTarget>
                                                <AbsoluteTargetLane value="{target_lane}"/>
                                            </LaneChangeTarget>
                                        </LaneChangeAction>
                                    </LateralAction>
                                </PrivateAction>
                            </Action>
                            <StartTrigger>
                                <ConditionGroup>
                                    <Condition name="StartLC_{i+1}" delay="0" conditionEdge="rising">
                                        <ByValueCondition>
                                            <SimulationTimeCondition value="{t0:.2f}" rule="greaterThan"/>
                                        </ByValueCondition>
                                    </Condition>
                                </ConditionGroup>
                            </StartTrigger>
                        </Event>
                    </Maneuver>
                </ManeuverGroup>
                <StartTrigger>
                    <ConditionGroup>
                        <Condition name="ActStart_{i+1}" delay="0" conditionEdge="rising">
                            <ByValueCondition>
                                <SimulationTimeCondition value="0" rule="greaterThan"/>
                            </ByValueCondition>
                        </Condition>
                    </ConditionGroup>
                </StartTrigger>
            </Act>""".strip("\n"))
    return "\n".join(blocks)


def _base_substitutions(params: dict[str, Any], scenario_description: str,
                         road_network: str, sim_time: float,
                         T_LC: float) -> dict[str, str]:
    return {
        "{{TIMESTAMP}}": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "{{SCENARIO_DESCRIPTION}}": scenario_description.replace('"', "'"),
        "{{EGO_INIT_SPEED}}": f"{params.get('ego_init_speed', 18.0):.3f}",
        "{{LC_LAT_V}}": f"{params.get('lat_v_max', 0.8):.3f}",
        "{{LC_LAT_A}}": f"{params.get('lat_a_max', 1.2):.3f}",
        "{{LC_DURATION}}": f"{T_LC:.3f}",
        "{{CUTIN_DHW}}": f"{params.get('cutin_dhw', 8.0):.3f}",
        "{{LAG_GAP}}": f"{params.get('lag_gap', 5.0):.3f}",
        "{{LEAD_GAP}}": f"{params.get('lead_gap', 18.0):.3f}",
        "{{PET}}": f"{params.get('pet', 1.5):.3f}",
        "{{DELTA_V}}": f"{params.get('delta_v', 2.0):.3f}",
        "{{TTC_TARGET}}": f"{params.get('ttc_target', 2.5):.3f}",
        "{{CUTIN_TRIGGER_TIME}}": f"{LC_START_OFFSET + max(T_LC, 2.0):.2f}",
        "{{BRAKE_TRIGGER_TIME}}": f"{LC_START_OFFSET + 3.0:.2f}",
        "{{BRAKE_DECEL}}": f"{params.get('brake_decel', 4.0):.3f}",
        "{{ROAD_NETWORK}}": road_network,
        "{{SIM_TIME}}": f"{sim_time:.1f}",
    }


def emit_xosc(template_path: str | Path,
              out_path: str | Path,
              params: dict[str, Any],
              num_lc: int,
              scenario_description: str,
              *,
              road_network: str = "./road_network/highway_4lane.xodr",
              sim_time: float = 40.0,
              template_id: str = "consecutive_lane_change") -> str:
    """Substitute {{PLACEHOLDERS}} in template with sampled parameter values.

    ``template_id`` drives per-template sim_time heuristics and which
    placeholder block (if any) replaces ``<!-- ACT_LC_EVENTS_PLACEHOLDER -->``.
    """
    template = Path(template_path).read_text(encoding="utf-8")
    T_LC = float(params.get("T_LC", 4.0))

    if template_id == "consecutive_lane_change":
        computed_sim_time = (LC_START_OFFSET + num_lc * T_LC
                             + max(num_lc - 1, 0) * DWELL_BETWEEN_LC
                             + LC_TAIL_PAD)
    elif template_id == "cut_in_conflict":
        computed_sim_time = LC_START_OFFSET + max(T_LC, 2.0) + T_LC + LC_TAIL_PAD + 5.0
    elif template_id == "close_following":
        computed_sim_time = LC_START_OFFSET + 3.0 + 8.0 + LC_TAIL_PAD
    else:
        computed_sim_time = sim_time
    sim_time = max(sim_time, computed_sim_time)

    out = template
    for k, v in _base_substitutions(params, scenario_description,
                                    road_network, sim_time, T_LC).items():
        out = out.replace(k, v)
    if template_id == "consecutive_lane_change":
        out = out.replace("<!-- ACT_LC_EVENTS_PLACEHOLDER -->",
                          _render_lc_events(num_lc, T_LC))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(out, encoding="utf-8")
    return str(out_path)


_TARGET_VEHICLE_CYCLE = ("car_red", "car_blue", "car_yellow")
_AMBIENT_VEHICLE_CYCLE = ("car_white", "car_grey")


def _headings_from_xy(xs: list[float], ys: list[float]) -> list[float]:
    """Per-frame heading via atan2 over consecutive rows; pad first with second."""
    if len(xs) < 2:
        return [0.0] * len(xs)
    headings: list[float] = []
    for i in range(1, len(xs)):
        headings.append(math.atan2(ys[i] - ys[i - 1], xs[i] - xs[i - 1]))
    return [headings[0]] + headings


def _frame_times(df: "pd.DataFrame", frame_rate: float) -> list[float]:
    """raw_tracks only carries `frame`; derive time via frame/frame_rate."""
    if "t" in df.columns:
        return [float(v) for v in df["t"].tolist()]
    return [float(f) / frame_rate for f in df["frame"].tolist()]


def _build_follow_traj_action(df: "pd.DataFrame", name: str, t0: float,
                              frame_rate: float):
    from scenariogeneration import xosc

    xs = [float(v) for v in df["x"].tolist()]
    ys_flipped = [-float(v) for v in df["y"].tolist()]  # flip y to match image coords
    ts = [v - t0 for v in _frame_times(df, frame_rate)]
    headings = _headings_from_xy(xs, ys_flipped)

    position_list = [
        xosc.WorldPosition(xs[i], ys_flipped[i], 0, headings[i], 0, 0)
        for i in range(len(xs))
    ]
    polyline = xosc.Polyline(ts, position_list)
    traj = xosc.Trajectory(f"traj_{name}", False)
    traj.add_shape(polyline)
    return xosc.FollowTrajectoryAction(traj, xosc.FollowingMode.position)


def _init_speed_and_teleport(df: "pd.DataFrame", entityname: str,
                              default_speed: float = 20.0):
    from scenariogeneration import xosc

    x0 = float(df["x"].iloc[0])
    y0 = -float(df["y"].iloc[0])  # flip y
    if len(df) >= 2:
        h0 = math.atan2(
            -float(df["y"].iloc[1]) - y0,
            float(df["x"].iloc[1]) - x0,
        )
    else:
        h0 = 0.0
    if "xVelocity" in df.columns and len(df) > 0:
        v0 = float(df["xVelocity"].iloc[0])
    else:
        v0 = default_speed

    step_time = xosc.TransitionDynamics(
        xosc.DynamicsShapes.step, xosc.DynamicsDimension.time, 1
    )
    speed_action = xosc.AbsoluteSpeedAction(v0, step_time)
    tele_action = xosc.TeleportAction(xosc.WorldPosition(x0, y0, 0, h0, 0, 0))
    return speed_action, tele_action


def _attach_follow_traj(story, entityname: str, traj_action) -> None:
    from scenariogeneration import xosc

    man_group = xosc.ManeuverGroup(f"mg_{entityname}")
    man_group.add_actor(entityname)
    maneuver = xosc.Maneuver(f"man_{entityname}")
    event = xosc.Event(f"ev_{entityname}", xosc.Priority.overwrite)
    start_trigger = xosc.ValueTrigger(
        f"start_{entityname}",
        0,
        xosc.ConditionEdge.none,
        xosc.SimulationTimeCondition(0, xosc.Rule.greaterOrEqual),
    )
    event.add_trigger(start_trigger)
    event.add_action(f"act_{entityname}", traj_action)
    maneuver.add_event(event)
    man_group.add_maneuver(maneuver)

    act_start_trigger = xosc.ValueTrigger(
        f"act_start_{entityname}",
        0,
        xosc.ConditionEdge.none,
        xosc.SimulationTimeCondition(0, xosc.Rule.greaterOrEqual),
    )
    act = xosc.Act(f"act_{entityname}", act_start_trigger)
    act.add_maneuver_group(man_group)
    story.add_act(act)


def emit_followtrajectory_xosc(
    ego_track,
    target_tracks,
    ambient_tracks,
    *,
    out_path: str | Path,
    region: str,
    road_network: str = "./road_network/highway_4lane.xodr",
    frame_rate: float = 25.0,
    scenario_name: str | None = None,
) -> str:
    """Emit an OpenSCENARIO 1.2 file that replays recorded trajectories.

    Ego + scripted targets follow recorded dataset (x, y) via
    FollowTrajectoryAction; ambient vehicles get only TeleportAction +
    AbsoluteSpeedAction. y is flipped (``-y``) to match the dataset's image
    coordinate system, matching Chat2Scenario's ``create_action`` convention.
    """
    try:
        from scenariogeneration import xosc
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "install scenariogeneration: pip install scenariogeneration==0.14.4"
        ) from exc

    ego_ts = _frame_times(ego_track, frame_rate)
    t0, t_end = ego_ts[0], ego_ts[-1]
    sim_duration = (t_end - t0) + 0.5

    # ----- parameter declarations -----
    paramdec = xosc.ParameterDeclarations()
    paramdec.add_parameter(
        xosc.Parameter("$HostVehicle", xosc.ParameterType.string, "car_white")
    )

    # ----- road network -----
    road = xosc.RoadNetwork(roadfile=road_network)

    # ----- entities + init -----
    entities = xosc.Entities()
    init = xosc.Init()

    def _add_entity(name: str, vehicle_name: str, df, *, with_traj: bool):
        cataref = xosc.CatalogReference("VehicleCatalog", vehicle_name)
        entities.add_scenario_object(name, cataref)
        speed_action, tele_action = _init_speed_and_teleport(df, name)
        init.add_init_action(name, tele_action)
        init.add_init_action(name, speed_action)
        if with_traj and len(df) >= 2:
            return _build_follow_traj_action(df, name, t0, frame_rate)
        return None

    ego_traj_action = _add_entity("Ego", "car_white", ego_track, with_traj=True)

    target_traj_actions: list[tuple[str, Any]] = []
    for i, tdf in enumerate(target_tracks, start=1):
        name = f"Target_{i}"
        veh = _TARGET_VEHICLE_CYCLE[(i - 1) % len(_TARGET_VEHICLE_CYCLE)]
        action = _add_entity(name, veh, tdf, with_traj=True)
        target_traj_actions.append((name, action))

    for i, adf in enumerate(ambient_tracks, start=1):
        name = f"Ambient_{i}"
        veh = _AMBIENT_VEHICLE_CYCLE[(i - 1) % len(_AMBIENT_VEHICLE_CYCLE)]
        _add_entity(name, veh, adf, with_traj=False)

    # ----- story / storyboard -----
    story = xosc.Story(scenario_name or f"replay_{region}")
    _attach_follow_traj(story, "Ego", ego_traj_action)
    for name, action in target_traj_actions:
        _attach_follow_traj(story, name, action)

    stop_trigger = xosc.ValueTrigger(
        "stop_simulation",
        0,
        xosc.ConditionEdge.rising,
        xosc.SimulationTimeCondition(sim_duration, xosc.Rule.greaterThan),
        "stop",
    )
    storyboard = xosc.StoryBoard(init, stoptrigger=stop_trigger)
    storyboard.add_story(story)

    scenario = xosc.Scenario(
        name=scenario_name or f"replay_{region}_{datetime.utcnow():%Y%m%dT%H%M%SZ}",
        author="region_envelope_injector",
        parameters=paramdec,
        entities=entities,
        storyboard=storyboard,
        roadnetwork=road,
        catalog=xosc.Catalog(),
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    scenario.write_xml(str(out_path))
    return str(out_path)
