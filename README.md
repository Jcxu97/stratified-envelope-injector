# stratified-envelope-injector · V1

OpenSCENARIO 1.2 generator for the cross-national highway-behavior study
(AD4CHE vs. highD), built on three ideas:

1. **Per-region envelopes** — behavior dimensions (D1–D14: TTC, THW, PET,
   gap acceptance, cut-in DHW, lane-change dynamics, …) are locked to
   percentile bounds derived from AD4CHE (CN) and highD (DE).
2. **Stratified sampling** — every envelope is keyed on the traffic phase
   `{F: free-flow, S: synchronised, J: wide-jam}`; the sampler only ever
   draws from the `(region, stratum)` cell that matches the requested
   scenario.
3. **Template injection** — sampled scalars are substituted into
   parameterised `.xosc.j2` scenario templates (consecutive lane change,
   cut-in conflict, close following) to emit a ready-to-replay
   OpenSCENARIO file.

## Pipeline

```
NL text  ──►  nl_llm_parser              (LLM or keyword heuristic)
              │
              ▼
          ScenarioRequest(template_id, stratum, dimensions)
              │
              ▼
          tier_router ──► sampler         (paper-locked envelope → dataset → default)
              │
              ▼
          emit_xosc(template, params)     (Jinja-style substitution)
              │
              ▼
          ambient_xosc.inject_ambient     (sprinkles dataset-derived ambient traffic)
              │
              ▼
          scene_renderer / esmini_renderer
```

## Entry points

```bash
# CLI
python -m region_envelope_injector.cli \
    --region CN \
    --text "ego performs four consecutive lane changes to reach the leftmost lane" \
    --out ./demo/out.xosc

# Web UI
streamlit run region_envelope_injector/ui/app.py
```

## Envelopes

`region_envelope_injector/scenario_envelopes_paperlocked.json` carries
the paper-locked percentile bounds. `bin/rebuild_envelopes_from_paper.py`
regenerates them from the raw dataset when new recordings are added.

## Version

**V1** — template-driven synthesis.
Development continues on a V2 line (data-driven real-trajectory replay).
