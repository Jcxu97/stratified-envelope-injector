"""3D preview of a generated .xosc via esmini.

Pipeline
--------
1. Locate the esmini binary (bundled locally on Windows, or in the Linux
   site-packages layout on Databricks). The ``ESMINI_ROOT`` env var overrides
   auto-detection.
2. Run ``esmini --record scene.dat`` headlessly. This requires no GL context
   so works on any machine (Windows / Databricks Linux, etc).
3. Replay with ``replayer --capture_screen``. On Windows this spawns a native
   GL window; on Databricks it needs ``xvfb-run`` in front (see
   ``scripts/setup_databricks.sh``).
4. Stitch the emitted ``screen_shot_*.tga`` sequence into an MP4 (preferred)
   or GIF using ``imageio-ffmpeg``.

Public entry point: :func:`render_xosc_3d`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_LOCAL_ROOT = Path("C:/Users/82077/Desktop/STLA/tools/esmini")
DEFAULT_LINUX_ROOT = Path("/opt/esmini")


def locate_esmini_root() -> Path:
    """Find esmini. Env var ``ESMINI_ROOT`` wins; falls back to the bundled
    Windows location on dev machines or ``/opt/esmini`` on Linux."""
    env = os.environ.get("ESMINI_ROOT")
    if env:
        p = Path(env)
        if (p / "bin").is_dir():
            return p
    candidates: list[Path] = [DEFAULT_LOCAL_ROOT, DEFAULT_LINUX_ROOT]
    for c in candidates:
        if (c / "bin").is_dir():
            return c
    raise FileNotFoundError(
        "esmini binary not found. Set ESMINI_ROOT or place it at "
        f"{DEFAULT_LOCAL_ROOT} (Windows) / {DEFAULT_LINUX_ROOT} (Linux).")


def _bin(name: str, root: Path) -> Path:
    exe = ".exe" if sys.platform.startswith("win") else ""
    p = root / "bin" / f"{name}{exe}"
    if not p.exists():
        raise FileNotFoundError(p)
    return p


def _needs_xvfb() -> bool:
    return (not sys.platform.startswith("win")
            and not os.environ.get("DISPLAY"))


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    if _needs_xvfb() and shutil.which("xvfb-run"):
        cmd = ["xvfb-run", "-a", "-s", "-screen 0 1280x480x24"] + cmd
    subprocess.run(cmd, cwd=cwd, check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def _tga_frames_to_mp4(frames_dir: Path, out_path: Path, fps: int = 25
                        ) -> Path:
    import imageio.v2 as imageio
    tgas = sorted(frames_dir.glob("screen_shot_*.tga"))
    if not tgas:
        raise RuntimeError(f"no TGA frames found in {frames_dir}")
    # Verify frames aren't all-black (happens when GL context failed).
    mid = imageio.imread(tgas[len(tgas) // 2])
    if mid.std() < 1.0:
        raise RuntimeError(
            f"esmini frames are blank (GL context likely failed). "
            f"On Linux ensure xvfb is installed; on Windows run from a "
            f"desktop session so a GL window can be created.")
    writer_kwargs = {"fps": fps, "codec": "libx264", "quality": 7} \
        if out_path.suffix.lower() == ".mp4" else {"fps": fps}
    with imageio.get_writer(str(out_path), **writer_kwargs) as w:
        for tga in tgas:
            w.append_data(imageio.imread(tga))
    return out_path


def render_xosc_3d(xosc_path: Path, out_path: Path,
                   *,
                   window: tuple[int, int, int, int] = (0, 0, 1280, 480),
                   fixed_timestep: float = 0.04,
                   fps: int = 25,
                   keep_tga: bool = False,
                   esmini_root: Path | None = None) -> Path:
    """Render ``xosc_path`` to a 3D MP4 (or GIF) via esmini.

    Parameters
    ----------
    xosc_path : Path
        Scenario file. Its referenced ``.xodr`` (OpenDRIVE) must exist on
        disk; the ``LogicFile filepath=`` in the xosc is resolved relative
        to ``xosc_path``'s directory.
    out_path : Path
        Output .mp4 / .gif file.
    window : (x, y, w, h)
        Viewer window position/size. On Linux (Databricks) the Xvfb
        screen size is taken from ``w, h``.
    """
    xosc_path = Path(xosc_path).resolve()
    out_path = Path(out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    root = Path(esmini_root) if esmini_root else locate_esmini_root()
    esmini = _bin("esmini", root)
    frames_dir = out_path.parent / f".{out_path.stem}_frames"
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Stage the xosc and its xodr into the frames dir so esmini can resolve
    # the relative LogicFile path and write TGAs locally.
    staged_xosc = frames_dir / xosc_path.name
    shutil.copy(xosc_path, staged_xosc)
    import re
    text = xosc_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'LogicFile\s+filepath="([^"]+)"', text)
    if m:
        xodr_ref = Path(m.group(1))
        src_xodr = (xosc_path.parent / xodr_ref).resolve()
        if not src_xodr.exists():
            raise FileNotFoundError(f"xodr {src_xodr} referenced by xosc "
                                     f"not found")
        dst_xodr = frames_dir / xodr_ref
        dst_xodr.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_xodr, dst_xodr)

    x, y, w, h = window
    cmd = [str(esmini),
           "--osc", staged_xosc.name,
           "--fixed_timestep", f"{fixed_timestep:.3f}",
           "--window", str(x), str(y), str(w), str(h),
           "--capture_screen",
           "--disable_stdout",
           "--aa_mode", "4"]
    _run(cmd, cwd=frames_dir)

    result = _tga_frames_to_mp4(frames_dir, out_path, fps=fps)
    if not keep_tga:
        for t in frames_dir.glob("screen_shot_*.tga"):
            t.unlink()
    return result


def launch_xosc_viewer(xosc_path: Path,
                        *,
                        window: tuple[int, int, int, int] = (100, 100, 1280, 720),
                        esmini_root: Path | None = None) -> int:
    """Spawn the esmini viewer non-blocking — opens a draggable 3D window the
    user can interact with. Side-steps the ``--capture_screen`` subprocess-GL
    bug on Windows (where TGAs come back blank). Returns the PID."""
    xosc_path = Path(xosc_path).resolve()
    root = Path(esmini_root) if esmini_root else locate_esmini_root()
    esmini = _bin("esmini", root)

    frames_dir = xosc_path.parent / f".{xosc_path.stem}_viewer"
    frames_dir.mkdir(parents=True, exist_ok=True)
    staged_xosc = frames_dir / xosc_path.name
    shutil.copy(xosc_path, staged_xosc)
    import re
    text = xosc_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r'LogicFile\s+filepath="([^"]+)"', text)
    if m:
        xodr_ref = Path(m.group(1))
        src_xodr = (xosc_path.parent / xodr_ref).resolve()
        if src_xodr.exists():
            dst_xodr = frames_dir / xodr_ref
            dst_xodr.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src_xodr, dst_xodr)

    x, y, w, h = window
    cmd = [str(esmini),
           "--osc", staged_xosc.name,
           "--window", str(x), str(y), str(w), str(h),
           "--aa_mode", "4"]
    # Non-blocking: give the user a live window they can drag / orbit.
    proc = subprocess.Popen(cmd, cwd=frames_dir,
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
    return proc.pid


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--xosc", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--window", default="0,0,1280,480")
    args = parser.parse_args()
    win = tuple(int(v) for v in args.window.split(","))
    p = render_xosc_3d(Path(args.xosc), Path(args.out), window=win)
    print(f"wrote {p}")
