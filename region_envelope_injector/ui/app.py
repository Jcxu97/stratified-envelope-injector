"""Streamlit UI for region_envelope_injector.

Run:
    streamlit run region_envelope_injector/ui/app.py

Features:
  - Free-text NL input (English / Chinese / mixed)
  - LLM config (endpoint URL + model name + token) persisted per-session
  - CN / DE region checkboxes (generate either or both side-by-side)
  - Dataset recording selector per region (AD4CHE DJI_* / highD NN)
  - Renders the generated scenario over real ambient traffic sampled from the
    chosen dataset clip
  - Shows the ORIGINAL dataset clip (reference) and the GENERATED scenario
    side by side
  - Downloadable .xosc and trace.json per region
"""
from __future__ import annotations

import json
import os
import random
import sys
import tempfile
import traceback
from pathlib import Path

import streamlit as st

_IS_WINDOWS = sys.platform.startswith("win")

_THIS = Path(__file__).resolve()
_ROOT = _THIS.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from region_envelope_injector.dataset_loader import (  # noqa: E402
    DEFAULT_AD4CHE_ROOT, DEFAULT_HIGHD_ROOT, list_available, load_clip,
    normalise_clip,
)
from region_envelope_injector.injector import generate_region_scenario  # noqa: E402
from region_envelope_injector.scene_renderer import (  # noqa: E402
    render_original_clip, render_scenario,
)
from region_envelope_injector.esmini_renderer import (  # noqa: E402
    locate_esmini_root, render_xosc_3d, launch_xosc_viewer,
)
from region_envelope_injector.ambient_xosc import inject_ambient  # noqa: E402

st.set_page_config(page_title="Region Envelope Injector",
                    layout="wide",
                    initial_sidebar_state="expanded")

DEFAULT_ENVELOPES = str(_ROOT / "region_envelope_injector" /
                         "scenario_envelopes_paperlocked.json")
SESSION_OUT = Path(tempfile.gettempdir()) / "region_envelope_injector_ui"
SESSION_OUT.mkdir(parents=True, exist_ok=True)


def _cached_clip(region: str, rec_id: int, duration: float,
                  ad4che_root: str, highd_root: str):
    key = f"clip::{region}::{rec_id}::{duration}"
    if key in st.session_state:
        return st.session_state[key]
    clip = normalise_clip(load_clip(region, recording_id=rec_id,
                                     duration=duration,
                                     ad4che_root=ad4che_root,
                                     highd_root=highd_root))
    st.session_state[key] = clip
    return clip


def sidebar() -> dict:
    st.sidebar.title("⚙️ Configuration")

    mode = st.sidebar.radio(
        "Mode",
        options=["General (single scenario)", "Comparison (CN ‖ DE)"],
        index=0,
        help="General = Chat2Scenario-style: pick any region + recording, "
             "one scenario out. Comparison = paper-mode: CN and DE "
             "side-by-side from the same description.",
    )

    st.sidebar.subheader("LLM endpoint (optional)")
    use_llm = st.sidebar.checkbox("Use LLM parser",
                                    value=bool(os.environ.get("ANTHROPIC_BASE_URL")
                                                or os.environ.get("DATABRICKS_HOST")),
                                    help="If unchecked the heuristic keyword "
                                         "parser is used (no network calls).")
    llm_endpoint = st.sidebar.text_input(
        "Endpoint (ANTHROPIC_BASE_URL or DATABRICKS_HOST)",
        value=os.environ.get("ANTHROPIC_BASE_URL",
                              os.environ.get("DATABRICKS_HOST", "")),
        placeholder="https://<workspace>.cloud.databricks.com",
    )
    llm_model = st.sidebar.text_input(
        "Model / endpoint name",
        value=os.environ.get("DATABRICKS_LLM_ENDPOINT",
                              "databricks-claude-opus-4-7"),
    )
    llm_token = st.sidebar.text_input(
        "API token (DATABRICKS_TOKEN)",
        value=os.environ.get("DATABRICKS_TOKEN", ""),
        type="password",
    )

    st.sidebar.subheader("Dataset roots")
    ad4che_root = st.sidebar.text_input(
        "AD4CHE root",
        value=str(DEFAULT_AD4CHE_ROOT),
    )
    highd_root = st.sidebar.text_input(
        "highD root",
        value=str(DEFAULT_HIGHD_ROOT),
    )

    st.sidebar.subheader("Envelope")
    envelopes_path = st.sidebar.text_input(
        "Paper-locked envelope JSON",
        value=DEFAULT_ENVELOPES,
    )

    return {
        "mode": "compare" if mode.startswith("Comparison") else "general",
        "use_llm": use_llm,
        "llm_endpoint": llm_endpoint.strip(),
        "llm_model": llm_model.strip(),
        "llm_token": llm_token.strip(),
        "ad4che_root": ad4che_root.strip(),
        "highd_root": highd_root.strip(),
        "envelopes_path": envelopes_path.strip(),
    }


def _apply_llm_env(cfg: dict):
    if cfg["use_llm"] and cfg["llm_endpoint"]:
        ep = cfg["llm_endpoint"]
        if "/anthropic" in ep or ep.endswith("ai-gateway.azuredatabricks.net"):
            os.environ["ANTHROPIC_BASE_URL"] = ep
            os.environ.pop("DATABRICKS_HOST", None)
        else:
            os.environ["DATABRICKS_HOST"] = ep
            os.environ.pop("ANTHROPIC_BASE_URL", None)
    if cfg["llm_model"]:
        os.environ["DATABRICKS_LLM_ENDPOINT"] = cfg["llm_model"]
    if cfg["llm_token"]:
        os.environ["DATABRICKS_TOKEN"] = cfg["llm_token"]


def generate_for_region(region: str, nl_text: str, seed: int,
                         rec_id: int, duration: float, cfg: dict) -> dict:
    out_dir = SESSION_OUT / region
    out_dir.mkdir(parents=True, exist_ok=True)
    result = generate_region_scenario(
        scenario_description=nl_text,
        region=region,
        envelopes_path=cfg["envelopes_path"],
        out_dir=str(out_dir),
        seed=seed,
        use_llm_parser=cfg["use_llm"],
    )
    clip = _cached_clip(region, rec_id, duration,
                        cfg["ad4che_root"], cfg["highd_root"])
    # Embed dataset vehicles into the xosc so esmini's 3D preview shows
    # ambient traffic. Ambient speed is boosted above ego_init_speed inside
    # inject_ambient so constant-speed ambients pull ahead of ego rather
    # than being overtaken — prevents rear-end collisions during LC.
    if result.xosc_path:
        ego_v = float(result.sampled_params.get("ego_init_speed", 20.0))
        try:
            inject_ambient(Path(result.xosc_path), clip,
                            n_ambient=12, ego_speed=ego_v)
        except Exception as e:
            st.warning(f"Ambient injection skipped ({region}): {e}")
    gif_out = out_dir / f"scenario_{region}_{result.request.stratum}.gif"
    tag = (f"{region} {result.request.stratum}/"
           f"{result.request.template_id.replace('_', ' ')}")
    if result.xosc_path:
        render_scenario(Path(result.xosc_path), clip, gif_out, tag=tag)
    else:
        gif_out = None
    # Lazy: the original-clip GIF is only needed when the user opens the
    # Original tab. Record the intended output path but defer rendering.
    orig_gif = out_dir / f"original_{region}_{rec_id}.gif"
    return {
        "result": result,
        "generated_gif": gif_out,
        "original_gif": orig_gif,
        "original_duration": min(duration, 12.0),
        "clip": clip,
    }


def _render_result_tabs(results: dict):
    """Shared tab renderer — accepts {region_key: blob} and produces the
    Generated / Original / 3D / Trace tabs in a column-per-region layout."""
    tab_gen, tab_orig, tab_3d, tab_trace = st.tabs(
        ["🎬 Generated scenario", "📼 Original dataset clip",
         "🧊 3D preview (esmini)", "🧾 Trace"]
    )

    with tab_gen:
        cols = st.columns(len(results) or 1)
        for col, (region, blob) in zip(cols, results.items()):
            with col:
                r = blob["result"]
                st.markdown(
                    f"### {region} | stratum **{r.request.stratum}** | "
                    f"{r.request.template_id}"
                )
                if blob["generated_gif"] and Path(blob["generated_gif"]).exists():
                    st.image(str(blob["generated_gif"]),
                              caption=f"Generated on ambient traffic: "
                                      f"{blob['clip'].source}")
                else:
                    st.warning("No .xosc emitted for this template.")
                st.markdown("**Sampled params**")
                st.dataframe(
                    [{"param": k, "value": f"{v:.3f}" if isinstance(v, float) else str(v)}
                     for k, v in r.sampled_params.items()],
                    use_container_width=True, hide_index=True)
                st.markdown("**Tier routing**")
                st.dataframe(
                    [{"dim": d["dim"], "tier": d["tier"], "source": d["source"]}
                     for d in r.tier_decisions],
                    use_container_width=True, hide_index=True)
                if r.xosc_path and Path(r.xosc_path).exists():
                    with open(r.xosc_path, "rb") as f:
                        st.download_button(
                            f"⬇️ Download {region} .xosc",
                            f.read(),
                            file_name=Path(r.xosc_path).name,
                            mime="application/xml",
                        )

    with tab_orig:
        cols = st.columns(len(results) or 1)
        for col, (region, blob) in zip(cols, results.items()):
            with col:
                st.markdown(f"### {region} — {blob['clip'].source}")
                st.caption(f"{blob['clip'].num_vehicles} vehicles, "
                            f"{blob['clip'].num_lanes} lane clusters, "
                            f"{blob['clip'].frame_rate:.0f} Hz, "
                            f"{blob['clip'].duration_s:.0f}s")
                orig_path = Path(blob["original_gif"])
                if not orig_path.exists():
                    if st.button(f"▶ Render original clip ({region})",
                                  key=f"origgif_{region}"):
                        with st.spinner("rendering..."):
                            render_original_clip(
                                blob["clip"], orig_path,
                                duration=blob.get("original_duration", 12.0))
                if orig_path.exists():
                    st.image(str(orig_path),
                              caption="Original dataset trajectories "
                                      "(no scripted ego overlay)")

    with tab_3d:
        try:
            esmini_root = locate_esmini_root()
            st.caption(f"esmini at `{esmini_root}`.")
        except FileNotFoundError as e:
            st.warning(str(e))
            st.markdown(
                "**Setup**: Linux/Databricks → run "
                "`bash region_envelope_injector/scripts/setup_databricks.sh`. "
                "Windows → download `esmini-bin_Windows.zip` from "
                "[esmini releases](https://github.com/esmini/esmini/releases/latest) "
                "and set `ESMINI_ROOT`.")
            esmini_root = None

        if esmini_root is not None:
            cols = st.columns(len(results) or 1)
            for col, (region, blob) in zip(cols, results.items()):
                with col:
                    r = blob["result"]
                    st.markdown(f"### {region} | {r.request.template_id}")
                    if not r.xosc_path:
                        st.info("No .xosc emitted for this template.")
                        continue
                    out_mp4 = Path(r.xosc_path).with_suffix(".3d.mp4")
                    # --capture_screen produces blank TGAs when esmini runs
                    # as a subprocess of streamlit on Windows (no attached GL
                    # context). Only show the MP4 button on Linux where xvfb
                    # makes capture reliable.
                    show_mp4_btn = not _IS_WINDOWS
                    cols_btn = st.columns(2 if show_mp4_btn else 1)
                    with cols_btn[0]:
                        if st.button(f"▶ Launch viewer ({region})",
                                      key=f"launch3d_{region}_{r.request.stratum}"):
                            try:
                                pid = launch_xosc_viewer(Path(r.xosc_path))
                                st.success(f"esmini spawned (pid={pid}).")
                            except Exception as e:
                                st.error(f"{type(e).__name__}: {e}")
                    if show_mp4_btn:
                        with cols_btn[1]:
                            if st.button(f"🎥 Render MP4 ({region})",
                                          key=f"render3d_{region}_{r.request.stratum}"):
                                with st.spinner("esmini rendering..."):
                                    try:
                                        render_xosc_3d(Path(r.xosc_path), out_mp4)
                                        st.success(f"wrote {out_mp4.name}")
                                    except Exception as e:
                                        st.error(f"{type(e).__name__}: {e}")
                    if out_mp4.exists():
                        st.video(str(out_mp4))
                        with open(out_mp4, "rb") as f:
                            st.download_button(
                                f"⬇️ {region} 3D MP4",
                                f.read(), file_name=out_mp4.name,
                                mime="video/mp4")

    with tab_trace:
        for region, blob in results.items():
            st.markdown(f"### {region} trace")
            st.code(blob["result"].to_json(), language="json")


def general_mode(cfg: dict):
    """Chat2Scenario-style: pick region + recording, single scenario out."""
    st.title("Region Envelope Injector — General")
    st.caption("Chat2Scenario-style: describe a scenario, pick a dataset "
                "clip, get one OpenSCENARIO back.")

    col_left, col_right = st.columns([3, 2])
    with col_left:
        nl_text = st.text_area(
            "场景描述 (Natural language)",
            value="Ego performs four consecutive lane changes in a dense "
                  "highway to reach the leftmost lane.",
            height=120,
        )
    with col_right:
        region = st.selectbox("Region", ["CN (AD4CHE)", "DE (highD)"], index=0)
        region_key = "CN" if region.startswith("CN") else "DE"
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1,
                                value=42, step=1)
        duration = st.slider("Clip duration (s)", 10.0, 40.0, 20.0, step=1.0)

    with st.expander("Recording", expanded=True):
        try:
            ids = list_available(
                region_key,
                ad4che_root=cfg["ad4che_root"],
                highd_root=cfg["highd_root"])
        except Exception:
            ids = ()
        rec_choice = st.selectbox(
            f"{region_key} recording",
            ["🎲 Random"] + list(ids or [1]),
            index=0,
            help="Default = random pick seeded by the Seed value. Pick a "
                 "specific ID to reproduce a known clip.",
        )
        if rec_choice == "🎲 Random":
            pool = list(ids or [1])
            rnd = random.Random(int(seed))
            rec_id = rnd.choice(pool)
            st.caption(f"Random pick → recording **{rec_id}** (seed={int(seed)})")
        else:
            rec_id = rec_choice

    if st.button("🚀 Generate", type="primary", use_container_width=True):
        results: dict[str, dict] = {}
        with st.status("Running injector...", expanded=True) as status:
            status.update(label=f"{region_key} — parsing + sampling + rendering...")
            try:
                results[region_key] = generate_for_region(
                    region_key, nl_text, int(seed), int(rec_id),
                    duration, cfg)
                r = results[region_key]["result"]
                st.write(f"✅ done: {r.request.template_id} / stratum {r.request.stratum}")
            except Exception as e:
                st.error(f"{region_key} failed: {e}\n{traceback.format_exc()}")
            status.update(label="Done", state="complete")
        st.session_state["_last_results"] = results

    results = st.session_state.get("_last_results", {})
    if not results:
        st.info("👈 Fill description, pick recording, click Generate.")
        return
    _render_result_tabs(results)


def compare_mode(cfg: dict):
    """Paper-mode: CN and DE side-by-side for cross-national comparison."""
    st.title("Region Envelope Injector — Comparison")
    st.caption("Paper mode: generate paper-locked CN / DE scenarios from a "
                "single description and view them side-by-side.")

    col_left, col_right = st.columns([3, 2])

    with col_left:
        nl_text = st.text_area(
            "场景描述 (Natural language)",
            value="Ego performs four consecutive lane changes in a dense "
                  "highway to reach the leftmost lane.",
            height=120,
        )

    with col_right:
        st.markdown("**Generate region(s)**")
        cn_on = st.checkbox("🇨🇳 CN (AD4CHE)", value=True)
        de_on = st.checkbox("🇩🇪 DE (highD)", value=True)
        seed = st.number_input("Seed", min_value=0, max_value=2**31 - 1,
                                value=42, step=1)
        duration = st.slider("Clip duration (s)", 10.0, 40.0, 20.0, step=1.0)

    with st.expander("Recording selector", expanded=True):
        colA, colB = st.columns(2)
        with colA:
            try:
                cn_ids = list_available("CN", ad4che_root=cfg["ad4che_root"])
            except Exception:
                cn_ids = ()
            cn_choice = st.selectbox(
                "AD4CHE recording",
                ["🎲 Random"] + list(cn_ids or [1]), index=0)
            if cn_choice == "🎲 Random":
                cn_rec = random.Random(int(seed)).choice(list(cn_ids or [1]))
                st.caption(f"Random → **{cn_rec}**")
            else:
                cn_rec = cn_choice
        with colB:
            try:
                de_ids = list_available("DE", highd_root=cfg["highd_root"])
            except Exception:
                de_ids = ()
            de_choice = st.selectbox(
                "highD recording",
                ["🎲 Random"] + list(de_ids or [1]), index=0)
            if de_choice == "🎲 Random":
                de_rec = random.Random(int(seed) + 1).choice(list(de_ids or [1]))
                st.caption(f"Random → **{de_rec}**")
            else:
                de_rec = de_choice

    go = st.button("🚀 Generate", type="primary", use_container_width=True)

    if go:
        if not (cn_on or de_on):
            st.error("Select at least one region.")
            st.stop()
        results: dict[str, dict] = {}
        with st.status("Running injector...", expanded=True) as status:
            if cn_on:
                status.update(label="CN — parsing + sampling + rendering...")
                try:
                    results["CN"] = generate_for_region(
                        "CN", nl_text, int(seed), int(cn_rec),
                        duration, cfg)
                    st.write(f"✅ CN done: {results['CN']['result'].request.template_id} "
                              f"/ stratum {results['CN']['result'].request.stratum}")
                except Exception as e:
                    st.error(f"CN failed: {e}\n{traceback.format_exc()}")
            if de_on:
                status.update(label="DE — parsing + sampling + rendering...")
                try:
                    results["DE"] = generate_for_region(
                        "DE", nl_text, int(seed), int(de_rec),
                        duration, cfg)
                    st.write(f"✅ DE done: {results['DE']['result'].request.template_id} "
                              f"/ stratum {results['DE']['result'].request.stratum}")
                except Exception as e:
                    st.error(f"DE failed: {e}\n{traceback.format_exc()}")
            status.update(label="Done", state="complete")
        st.session_state["_last_results"] = results

    results = st.session_state.get("_last_results", {})
    if not results:
        st.info("👈 Fill in the description, pick regions, and click Generate.")
        return
    _render_result_tabs(results)


def main():
    cfg = sidebar()
    _apply_llm_env(cfg)
    if cfg["mode"] == "compare":
        compare_mode(cfg)
    else:
        general_mode(cfg)


if __name__ == "__main__":
    main()
