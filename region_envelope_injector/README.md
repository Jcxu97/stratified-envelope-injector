# region_envelope_injector

Region-aware envelope injector that bridges **Workflow A** (stratified
cross-national quantification) and **Workflow B** (executable scenario
generation via Chat2Scenario) of the paper
*"Stratified Cross-National Comparison of Highway Driving Behavior"* (2026).

## What it does

Given

- a free-text scenario description (e.g. *"Ego performs four consecutive lane
  changes on a dense, congested highway"*), and
- a region selector (`CN` = AD4CHE / `DE` = highD),

it produces

1. a **region-specific parameter sample** drawn from the corresponding
   envelope in `scenario_envelopes.json`,
2. a **Chat2Scenario-compatible config** (`metric_options` block with
   region-specific thresholds), and
3. a **parametrised OpenSCENARIO 1.2 `.xosc`** file for direct simulation.

The tier routing is mechanical: every `(dimension, stratum, region)` triple is
classified as Tier I / Tier II / Tier III / LLM-dominant per the paper's
`scenario_envelopes.json` contract (see §III-G).

## Install

No third-party dependencies beyond the standard library for the core path.
Chat2Scenario itself (used for trajectory extraction) has its own
requirements; see `../Chat2Scenario/requirements.txt`.

## Quickstart

```bash
# From the project root
python -m region_envelope_injector.cli \
    --region CN \
    --text "Ego performs four consecutive lane changes in dense congested traffic" \
    --out-dir output
```

Or programmatically:

```python
from region_envelope_injector import generate_region_scenario

r = generate_region_scenario(
    "Ego performs four consecutive lane changes to reach the leftmost lane "
    "on a congested Chinese expressway.",
    region="CN",
)
print(r.sampled_params)
print(r.xosc_path)
```

## Demo

```bash
python region_envelope_injector/demo/run_demo.py
```

Produces `scenario_CN_J_4LC.xosc` and `scenario_DE_J_4LC.xosc` side-by-side
plus a `demo_summary.json` comparing the sampled CN vs DE parameters. Typical
output on the demo envelopes shows CN cut-in DHW at ~50% of DE, CN PET at
~55% of DE, and CN peak lateral velocity at ~150% of DE — matching the
cross-national direction reported in §V-B.

## Architecture

```
  NL request + region
           |
           v
+----------------------+     scenario_envelopes.json
|  nl_region_parser    | <-- (Workflow A product)
+----------+-----------+
           |
           v
+----------------------+
|  tier_router          |   I / II / III / LLM-dominant
+----------+-----------+
           |
           v
+----------------------+
|  sampler              |   P10-P90 piecewise-linear draw
+----------+-----------+
           |
     +-----+------+
     v            v
Chat2Scenario   xosc_emitter
  config         (template substitution)
```

## File map

| File | Role |
|---|---|
| `scenario_envelopes.json` | Contract between Workflow A and Workflow B (14 dims × 3 strata × 2 regions, with Cliff's δ and FDR p). **Illustrative values**; rebuild from `statistical_results.csv` via `envelope_loader.rebuild_from_stats_csv()` once the paper's full statistical run is finalised. |
| `envelope_loader.py` | Loads the JSON into a typed `EnvelopeStore`; contains the rebuild hook. |
| `tier_router.py` | Mechanical three-tier decision tree (maps to §III-G of the paper). |
| `sampler.py` | Piecewise-linear inverse-CDF sampler over P10/P50/P90 envelopes. |
| `nl_region_parser.py` | Keyword-based NL → (template, stratum, relevant dims). Swap with a Databricks Claude call for production. |
| `metric_mapper.py` | Translates dimension envelopes into Chat2Scenario `metric_options`. |
| `xosc_emitter.py` | Template substitution → OpenSCENARIO 1.2 .xosc. |
| `injector.py` | End-to-end orchestrator. |
| `cli.py` | `python -m region_envelope_injector.cli --region CN --text ...` |
| `templates/consecutive_lane_change.xosc` | Parametrised .xosc template. |
| `demo/run_demo.py` | CN vs DE side-by-side demo. |

## Relationship to the paper

- **§III-G (From Quantified Differences to Scenario Parameters)** defines the
  three-tier decision tree. `tier_router.py` is the mechanical
  implementation.
- **§V-A / §V-B** produce per-`(dimension, stratum)` effect sizes that
  populate `scenario_envelopes.json`.
- **§III-E** dual-track LLM audit flags D7 as `llm_safety_net=true` in the S
  stratum; `tier_router.route()` returns `tier="LLM_DOMINANT"` for that case.

## Limitations (honest)

- **Envelope values are illustrative placeholders** until the paper's full
  `statistical_results.csv` is wired in. The *shape* of the output (direction
  and magnitude of CN/DE differences) tracks the paper's findings; the *exact
  numbers* will shift after the real CSV is loaded.
- **NL parser is heuristic** (keyword matching). Adequate for demo; production
  use should replace it with a Databricks Claude prompt that emits a
  structured `{template_id, stratum, num_lc}` JSON response.
- **Road-network file** is referenced but not generated. Any OpenSCENARIO
  tool (esmini, scenariogeneration, CarMaker) needs a matching `.xodr` to
  actually simulate. A 4-lane highway `.xodr` suitable for the demo is
  standard and is left out of scope here.
- **Closed-loop validation** (running the generated scenarios through a
  controller under each envelope and comparing failure modes) is out of
  scope — tracked as Stage-2 future work in §V-D of the paper.
