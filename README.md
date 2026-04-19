# stratified-envelope-injector · V1

OpenSCENARIO 1.2 generator for the cross-national highway-behavior study
(AD4CHE vs. highD), built on three ideas:

1. **Per-region envelopes** — D1–D14 behavior dimensions (TTC, THW, PET,
   gap acceptance, cut-in DHW, lane-change dynamics, …) locked to percentile
   bounds derived from the AD4CHE (CN) and highD (DE) recordings.
2. **Stratified sampling** — every envelope is keyed on the traffic phase
   `{F: free-flow, S: synchronised, J: wide-jam}`; the sampler only draws from
   the `(region, stratum)` cell matching the requested scenario.
3. **Template injection** — sampled scalars are substituted into parameterised
   `.xosc` scenario templates (consecutive lane change, cut-in conflict,
   close following) to emit a ready-to-replay OpenSCENARIO file.

## Pipeline

```
NL text ──► nl_llm_parser / nl_region_parser
             │   (LLM or keyword heuristic)
             ▼
         ScenarioRequest(template_id, stratum, dimensions)
             │
             ▼
         tier_router ──► sampler        (paper-locked → dataset fallback)
             │
             ▼
         emit_xosc(template, params)    (substitution into .xosc templates)
             │
             ▼
         ambient_xosc.inject_ambient    (overlay dataset-derived ambient)
             │
             ▼
         scene_renderer / esmini_renderer
```

## Quickstart (web UI)

```bash
git clone https://github.com/Jcxu97/stratified-envelope-injector
cd stratified-envelope-injector

python -m venv .venv
.venv\Scripts\activate             # Windows
# source .venv/bin/activate        # macOS / Linux

pip install -r region_envelope_injector/requirements.txt

# Point the loader at your AD4CHE / highD roots (edit DEFAULT_AD4CHE_ROOT /
# DEFAULT_HIGHD_ROOT in region_envelope_injector/dataset_loader.py, OR set
# them from the sidebar when the UI opens).

streamlit run region_envelope_injector/ui/app.py
```

The UI lets you:
- type a scenario description (CN / EN / mixed),
- pick a region (CN = AD4CHE, DE = highD) and a recording ID,
- optionally plug in a Databricks Claude endpoint for LLM-based classification,
- generate + preview the `.xosc` with ambient traffic overlaid as a GIF,
- launch esmini (3D viewer) on the emitted `.xosc`.

## CLI

```bash
python -m region_envelope_injector.cli \
    --region CN \
    --text "ego performs four consecutive lane changes to reach the leftmost lane" \
    --out ./demo/out.xosc
```

## LLM parser (optional)

Heuristic keyword parsing works out of the box. To use Databricks-Claude for
richer classification, set the env vars before launching:

```bash
export DATABRICKS_HOST="https://<workspace>.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi-..."
export DATABRICKS_LLM_ENDPOINT="databricks-claude-opus-4-7"
```

## Envelopes

`region_envelope_injector/scenario_envelopes_paperlocked.json` carries the
paper-locked percentile bounds. `bin/rebuild_envelopes_from_paper.py`
regenerates them from the raw dataset when new recordings are added.

## Version

**V1.0.1** — template-driven synthesis. Tagged snapshot of the approach used
in the IEEE T-ITS 2026 paper.
