# envelope-template-synthesizer

Frozen snapshot (2026-04-19) of the **template-based** OpenSCENARIO generator for
the cross-national highway-behavior study (AD4CHE vs. highD).

This approach is preserved here as a standalone reference; the main line of
development has moved to a **data-driven Chat2Scenario-style replay** pipeline
(LLM activity labels + real trajectory match + `FollowTrajectoryAction`).

## What this pipeline does

1. `nl_llm_parser.parse_scenario_request_llm` classifies a natural-language
   scenario request into `(template_id, stratum, num_lane_changes, relevant_dimensions)`.
2. `tier_router` + `sampler` draw parameter values from paper-locked envelopes
   per `(region, stratum, dimension)` (scenario_envelopes\*.json).
3. `xosc_emitter.emit_xosc` fills `templates/*.xosc.j2` with the sampled values.
4. `ambient_xosc.inject_ambient` post-processes the emitted `.xosc` to sprinkle
   ambient `<ScenarioObject>` entries around ego.
5. `scene_renderer` / `esmini_renderer` produce a GIF / replay for QA.

## Entry points

```
python -m region_envelope_injector.cli \
    --region CN \
    --text "ego cuts in from left after lead brakes" \
    --out ./demo/out.xosc
```

```
streamlit run region_envelope_injector/ui/app.py
```

## Status

**Frozen.** New work in the companion data-driven repo. Use this snapshot when
you need to reproduce the template-synthesis numbers in the IEEE T-ITS paper.
