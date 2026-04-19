"""Thin compatibility bridge to the vendored Chat2Scenario modules.

Chat2Scenario was written against pandas < 2.0 (uses the removed
``DataFrame.append`` method) and streamlit (imports ``streamlit as st`` at
module scope). We can't modify the vendored source — so this bridge:

1. Monkey-patches ``pandas.DataFrame.append`` back in as a no-op-compatible
   wrapper around ``pandas.concat`` before importing Chat2Scenario.
2. Adds the Chat2Scenario repo root to ``sys.path`` so its internal relative
   imports (``from utils.helper_data_functions import *``) still resolve.
3. Re-exports the two functions the injector pipeline actually needs.

Callers should always import from here, never directly from Chat2Scenario.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent / "Chat2Scenario"


def _install_pandas_append_shim() -> None:
    """Re-introduce DataFrame.append as a concat-based wrapper.

    Chat2Scenario's activity_identification.py does
    ``df = df.append(row_dict, ignore_index=True)`` — a removed API in pandas
    2.x. Restore just enough shape to keep it working."""
    if hasattr(pd.DataFrame, "append"):
        return

    def _append(self, other, ignore_index=False, verify_integrity=False, sort=False):
        if isinstance(other, dict):
            other_df = pd.DataFrame([other])
        elif isinstance(other, pd.Series):
            other_df = other.to_frame().T
        elif isinstance(other, (list, tuple)):
            other_df = pd.DataFrame(list(other))
        else:
            other_df = other
        return pd.concat([self, other_df], ignore_index=ignore_index,
                         verify_integrity=verify_integrity, sort=sort)

    pd.DataFrame.append = _append  # type: ignore[attr-defined]


_install_pandas_append_shim()


def _install_streamlit_shim() -> None:
    """Chat2Scenario's scenario_identification.py does ``import streamlit as st``
    at module scope. If streamlit isn't installed, register a stub that no-ops
    the handful of attributes Chat2Scenario reads (``st.progress``,
    ``st.warning``, etc.)."""
    if "streamlit" in sys.modules:
        return
    try:
        import streamlit  # noqa: F401
        return
    except ImportError:
        pass
    import types
    stub = types.ModuleType("streamlit")

    class _Noop:
        def __call__(self, *a, **kw): return self
        def __getattr__(self, _): return _Noop()
        def __enter__(self): return self
        def __exit__(self, *a): return False

    _noop = _Noop()
    for attr in ("progress", "warning", "info", "error", "write", "markdown",
                 "text", "success", "spinner", "empty"):
        setattr(stub, attr, _noop)
    sys.modules["streamlit"] = stub


_install_streamlit_shim()

if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))
# Chat2Scenario modules live under subpackages that themselves import siblings
# by bare name (``from utils.helper_data_functions import *``), so the repo
# root needs to be importable as a package root.
for sub in ("NLP", "scenario_mining", "utils", "Metric"):
    sub_path = _REPO / sub
    if sub_path.is_dir() and str(sub_path) not in sys.path:
        sys.path.insert(0, str(sub_path))

# Heavy imports deferred to avoid paying streamlit startup cost at module
# import time; the injector calls these lazily.
_activity_mod = None
_scenario_mod = None


def main_fcn_veh_activity(tracks: pd.DataFrame, progress_bar=None):
    """Chat2Scenario's vehicle activity classifier.

    Expects the un-renamed dataset columns (``xAcceleration``, ``precedingId``,
    ``laneId``, etc.) — i.e., pass ``clip.raw_tracks``, not ``clip.tracks``.
    Returns ``(longActDict, latActDict, interactIdDict)``.
    """
    global _activity_mod
    if _activity_mod is None:
        from scenario_mining import activity_identification as _am  # type: ignore
        _activity_mod = _am
    return _activity_mod.main_fcn_veh_activity(tracks, progress_bar=progress_bar)


def mainFunctionScenarioIdentification(raw_tracks: pd.DataFrame,
                                        key_label: dict,
                                        latActDict: dict,
                                        longActDict: dict,
                                        interactIdDict: dict,
                                        progress_bar=None):
    """Chat2Scenario's single-target scenario identifier.

    Used by :mod:`search_matcher` as one of multiple scoring branches. For
    multi-target requests the injector wraps this call per-target and merges
    the resulting scenario lists.
    """
    global _scenario_mod
    if _scenario_mod is None:
        from scenario_mining import scenario_identification as _sm  # type: ignore
        _scenario_mod = _sm
    return _scenario_mod.mainFunctionScenarioIdentification(
        raw_tracks, key_label, latActDict, longActDict, interactIdDict,
        progress_bar=progress_bar)
