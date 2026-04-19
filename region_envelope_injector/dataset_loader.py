"""Load real trajectory clips from AD4CHE (CN) and highD (DE).

Returns a :class:`DatasetClip` holding per-frame vehicle positions so the
scene renderer can overlay a scripted ego maneuver on genuine ambient traffic
instead of hand-synthesised scatter.

Dataset schemas
---------------
Both datasets share a core track schema (frame, id, x, y, xVelocity, yVelocity,
laneId, thw, ttc, ...). Differences:

- AD4CHE (CN): files under ``DJI_00XX/NN_tracks.csv``, frameRate=30 Hz, coords
  in metres (``scale=0.0375 m/px`` applied upstream). Up to 5 driving lanes
  including an expressway merge lane; laneId enumerated in ``*_lanePicture.png``.
- highD (DE): files under ``NN_tracks.csv``, frameRate=25 Hz, coords in metres,
  upper/lowerLaneMarkings recorded in ``*_recordingMeta.csv``. laneId 2-4 for
  upper bound, 5-7 for lower (direction-split).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_AD4CHE_ROOT = Path("C:/Users/82077/Desktop/STLA/CHINA specific scenarios/"
                            "AD4CHE_V1.0-001/AD4CHE_V1.0/AD4CHE_Data_V1.0")
DEFAULT_HIGHD_ROOT = Path("C:/Users/82077/Desktop/STLA/CHINA specific scenarios/"
                           "highD/data")


@dataclass
class DatasetClip:
    region: str                         # "CN" or "DE"
    source: str                         # file stem / DJI id
    frame_rate: float                   # Hz
    duration_s: float
    tracks: pd.DataFrame                # columns: t, id, x, y, vx, vy, lane
    lane_y_centers: list[float]         # absolute y-coord of each lane centreline
    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = (0.0, 0.0)
    background_image: Path | None = None
    meta: dict = field(default_factory=dict)
    # Un-renamed, full-column dataframe preserved for Chat2Scenario's
    # main_fcn_veh_activity (which expects `xAcceleration`, `precedingId`,
    # etc. by their original AD4CHE/highD names). Populated by the loader;
    # downstream consumers that just need positions should keep using `tracks`.
    raw_tracks: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def num_vehicles(self) -> int:
        return int(self.tracks["id"].nunique())

    @property
    def num_lanes(self) -> int:
        return len(self.lane_y_centers)

    def at(self, t_s: float, tolerance: float = 0.05) -> pd.DataFrame:
        """Return vehicles visible at time ``t_s`` seconds from clip start."""
        t_target = self.tracks["t"].min() + t_s
        nearest = self.tracks.loc[
            (self.tracks["t"] - t_target).abs() < tolerance]
        if nearest.empty:
            frame = self.tracks["t"].sub(t_target).abs().idxmin()
            target_t = self.tracks.loc[frame, "t"]
            nearest = self.tracks[self.tracks["t"] == target_t]
        return nearest


def _parse_lane_markings(raw: str) -> list[float]:
    if pd.isna(raw) or not isinstance(raw, str):
        return []
    return [float(x.strip()) for x in raw.split(";") if x.strip()]


def _select_main_lanes(df: pd.DataFrame, min_spacing: float = 3.0,
                       max_lanes: int = 6, min_vehicles: int = 4
                       ) -> tuple[pd.DataFrame, list[float]]:
    """Pick the subset of ``laneId``s that look like real highway lanes:
    populated, >= ``min_spacing`` m apart. Returns the filtered df and the
    sorted lane centre y-coords."""
    if df.empty:
        return df, []
    lane_y = df.groupby("lane")["y"].mean().to_dict()
    lane_count = df.groupby("lane")["id"].nunique().sort_values(ascending=False)
    kept: list = []
    kept_y: list[float] = []
    for lane_id, count in lane_count.items():
        if count < min_vehicles:
            continue
        y = float(lane_y[lane_id])
        if any(abs(y - ky) < min_spacing for ky in kept_y):
            continue
        kept.append(lane_id)
        kept_y.append(y)
        if len(kept) >= max_lanes:
            break
    if not kept:
        return df, []
    sub = df[df["lane"].isin(kept)].copy().reset_index(drop=True)
    lane_centers = sorted(float(lane_y[k]) for k in kept)
    return sub, lane_centers


def _filter_dominant_direction(df: pd.DataFrame) -> pd.DataFrame:
    """AD4CHE / highD tracks cover BOTH driving directions. Keep only the
    direction with more vehicles so visualisation shows a single highway
    direction (otherwise opposing traffic appears to 'collide' with ours at
    every pass). If the dominant direction drives in -x, flip x and vx so
    downstream code can always assume ego moves in +x."""
    if df.empty:
        return df
    pos_ids = df.loc[df["vx"] > 0, "id"].nunique()
    neg_ids = df.loc[df["vx"] < 0, "id"].nunique()
    if pos_ids >= neg_ids:
        kept = df[df["vx"] >= 0].copy()
    else:
        kept = df[df["vx"] <= 0].copy()
        kept["x"] = -kept["x"]
        kept["vx"] = -kept["vx"]
    return kept.reset_index(drop=True)


def _dominant_direction_sign(df: pd.DataFrame) -> int:
    """Return +1 if positive x-direction dominates, -1 otherwise. Mirrors the
    logic inside :func:`_filter_dominant_direction` so we can apply the same
    flip to `raw_tracks` (which keeps native column names and therefore can't
    go through `_filter_dominant_direction` directly)."""
    if df.empty:
        return 1
    pos_ids = df.loc[df["vx"] > 0, "id"].nunique()
    neg_ids = df.loc[df["vx"] < 0, "id"].nunique()
    return 1 if pos_ids >= neg_ids else -1


def _align_raw_to_df(raw: pd.DataFrame, df: pd.DataFrame,
                      sign: int) -> pd.DataFrame:
    """Keep only (frame, id) rows of `raw` that survived dominant-direction
    filtering in `df`, and mirror x/xVelocity/xAcceleration if sign=-1 so
    coordinate conventions match."""
    if raw.empty or df.empty:
        return raw.iloc[0:0].copy()
    kept_pairs = df[["frame", "id"]].drop_duplicates()
    merged = raw.merge(kept_pairs, on=["frame", "id"], how="inner")
    if sign == -1:
        merged = merged.copy()
        for col in ("x", "xVelocity", "xAcceleration"):
            if col in merged.columns:
                merged[col] = -merged[col]
    return merged.reset_index(drop=True)


def _load_highd_recording(recording_id: int,
                           root: Path = DEFAULT_HIGHD_ROOT,
                           duration: float = 30.0,
                           min_vehicles: int = 8) -> DatasetClip:
    stem = f"{recording_id:02d}"
    tracks_path = root / f"{stem}_tracks.csv"
    meta_path = root / f"{stem}_recordingMeta.csv"
    bg_path = root / f"{stem}_highway.png"
    if not tracks_path.exists():
        raise FileNotFoundError(tracks_path)

    meta = pd.read_csv(meta_path).iloc[0].to_dict()
    fps = float(meta.get("frameRate", 25))
    upper = _parse_lane_markings(meta.get("upperLaneMarkings", ""))
    lower = _parse_lane_markings(meta.get("lowerLaneMarkings", ""))
    all_markings = sorted(set(upper + lower))

    raw = pd.read_csv(tracks_path,
                     usecols=["frame", "id", "x", "y", "xVelocity", "yVelocity",
                              "xAcceleration", "yAcceleration", "laneId",
                              "precedingId", "followingId",
                              "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
                              "rightPrecedingId", "rightAlongsideId", "rightFollowingId"])
    raw["t"] = raw["frame"] / fps
    t0 = raw["t"].min()
    raw = raw[raw["t"] <= t0 + duration].copy()
    df = raw.rename(columns={"xVelocity": "vx", "yVelocity": "vy",
                       "laneId": "lane"})
    sign = _dominant_direction_sign(df)
    df = _filter_dominant_direction(df)
    _sub, lane_centers = _select_main_lanes(df)
    raw_aligned = _align_raw_to_df(raw, df, sign)
    # Keep the full direction-filtered df but use the main-lane y-band for
    # lane centres. Vehicles drifting into merge lanes are culled by the
    # renderer's y-band filter, not by dropping their rows — so lane-changing
    # cars don't pop in/out when their laneId switches.

    if df["id"].nunique() < min_vehicles:
        raise RuntimeError(
            f"highD recording {recording_id}: only "
            f"{df['id'].nunique()} vehicles; need {min_vehicles}")
    return DatasetClip(
        region="DE",
        source=f"highD-{stem}",
        frame_rate=fps,
        duration_s=duration,
        tracks=df[["t", "id", "x", "y", "vx", "vy", "lane"]],
        lane_y_centers=lane_centers,
        x_range=(float(df["x"].min()), float(df["x"].max())),
        y_range=(float(df["y"].min()), float(df["y"].max())),
        background_image=bg_path if bg_path.exists() else None,
        meta=meta,
        raw_tracks=raw_aligned,
    )


def _load_ad4che_recording(dji_id: int,
                            root: Path = DEFAULT_AD4CHE_ROOT,
                            duration: float = 30.0,
                            min_vehicles: int = 8) -> DatasetClip:
    folder = root / f"DJI_{dji_id:04d}"
    if not folder.exists():
        raise FileNotFoundError(folder)
    stems = sorted(folder.glob("*_tracks.csv"))
    if not stems:
        raise FileNotFoundError(f"no tracks in {folder}")
    tracks_path = stems[0]
    stem = tracks_path.name.replace("_tracks.csv", "")
    meta_path = folder / f"{stem}_recordingMeta.csv"
    bg_path = folder / f"{stem}_highway.png"

    meta = pd.read_csv(meta_path).iloc[0].to_dict()
    fps = float(meta.get("frameRate", 30))

    # AD4CHE schema is a superset of highD's — only read columns that exist,
    # and fill missing ones (if any) with zeros so main_fcn_veh_activity can
    # still iterate without KeyErrors.
    header = pd.read_csv(tracks_path, nrows=0).columns.tolist()
    wanted = ["frame", "id", "x", "y", "xVelocity", "yVelocity",
              "xAcceleration", "yAcceleration", "laneId",
              "precedingId", "followingId",
              "leftPrecedingId", "leftAlongsideId", "leftFollowingId",
              "rightPrecedingId", "rightAlongsideId", "rightFollowingId"]
    present = [c for c in wanted if c in header]
    raw = pd.read_csv(tracks_path, usecols=present)
    for col in wanted:
        if col not in raw.columns:
            raw[col] = 0
    raw["t"] = raw["frame"] / fps
    t0 = raw["t"].min()
    raw = raw[raw["t"] <= t0 + duration].copy()
    df = raw.rename(columns={"xVelocity": "vx", "yVelocity": "vy",
                       "laneId": "lane"})
    sign = _dominant_direction_sign(df)
    df = _filter_dominant_direction(df)
    _sub, lane_centers = _select_main_lanes(df)
    raw_aligned = _align_raw_to_df(raw, df, sign)

    if df["id"].nunique() < min_vehicles:
        raise RuntimeError(
            f"AD4CHE recording {dji_id}: only "
            f"{df['id'].nunique()} vehicles; need {min_vehicles}")
    return DatasetClip(
        region="CN",
        source=f"AD4CHE-DJI_{dji_id:04d}",
        frame_rate=fps,
        duration_s=duration,
        tracks=df[["t", "id", "x", "y", "vx", "vy", "lane"]],
        lane_y_centers=lane_centers,
        x_range=(float(df["x"].min()), float(df["x"].max())),
        y_range=(float(df["y"].min()), float(df["y"].max())),
        background_image=bg_path if bg_path.exists() else None,
        meta=meta,
        raw_tracks=raw_aligned,
    )


@lru_cache(maxsize=16)
def list_available(region: str,
                    ad4che_root: str | None = None,
                    highd_root: str | None = None) -> tuple[int, ...]:
    region = region.upper()
    if region == "CN":
        root = Path(ad4che_root) if ad4che_root else DEFAULT_AD4CHE_ROOT
        ids = []
        for p in sorted(root.glob("DJI_*")):
            try:
                ids.append(int(p.name.split("_")[1]))
            except ValueError:
                continue
        return tuple(ids)
    root = Path(highd_root) if highd_root else DEFAULT_HIGHD_ROOT
    ids = []
    for p in sorted(root.glob("*_tracks.csv")):
        try:
            ids.append(int(p.name.split("_")[0]))
        except ValueError:
            continue
    return tuple(ids)


def load_clip(region: str, recording_id: int | None = None,
              duration: float = 30.0,
              seed: int = 42,
              ad4che_root: str | None = None,
              highd_root: str | None = None) -> DatasetClip:
    region = region.upper()
    if region == "CN":
        available = list_available("CN", ad4che_root)
        if recording_id is None:
            recording_id = random.Random(seed).choice(available)
        root = Path(ad4che_root) if ad4che_root else DEFAULT_AD4CHE_ROOT
        return _load_ad4che_recording(recording_id, root, duration=duration)
    available = list_available("DE", highd_root=highd_root)
    if recording_id is None:
        recording_id = random.Random(seed).choice(available)
    root = Path(highd_root) if highd_root else DEFAULT_HIGHD_ROOT
    return _load_highd_recording(recording_id, root, duration=duration)


def normalise_clip(clip: DatasetClip,
                    target_lane_width: float = 3.5,
                    target_num_lanes: int = 5) -> DatasetClip:
    """Rescale clip coords so y is measured from 0 at the top lane centreline
    and lane spacing is ``target_lane_width``. Also mirrors y so that CN
    (driving on the right in data) and DE appear visually consistent."""
    df = clip.tracks.copy()
    if not clip.lane_y_centers:
        return clip

    centers = np.array(clip.lane_y_centers)
    spacing = np.mean(np.abs(np.diff(centers))) if len(centers) > 1 else target_lane_width
    scale = target_lane_width / spacing if spacing > 1e-3 else 1.0

    top = centers.min()
    x_shift = float(df["x"].min())
    df["y"] = (df["y"] - top) * scale
    df["x"] = df["x"] - x_shift
    clip.tracks = df
    if clip.raw_tracks is not None and not clip.raw_tracks.empty:
        raw = clip.raw_tracks.copy()
        raw["y"] = (raw["y"] - top) * scale
        raw["x"] = raw["x"] - x_shift
        clip.raw_tracks = raw
    clip.lane_y_centers = [(c - top) * scale for c in clip.lane_y_centers]
    clip.y_range = (float(df["y"].min()), float(df["y"].max()))
    clip.x_range = (float(df["x"].min()), float(df["x"].max()))
    return clip
