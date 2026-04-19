"""Region-aware envelope injector for Chat2Scenario.

Consumes scenario_envelopes.json (Workflow A output) plus a free-text scenario
description plus a region selector (CN/DE), and emits either:
  (a) a Chat2Scenario config dict with region-specific metric thresholds, or
  (b) a parametrised OpenSCENARIO .xosc file sampled from the region envelope.
"""
from region_envelope_injector.envelope_loader import EnvelopeStore, load_envelopes
from region_envelope_injector.tier_router import TierDecision, route
from region_envelope_injector.sampler import sample_envelope, sample_scenario_params
from region_envelope_injector.nl_region_parser import parse_scenario_request
from region_envelope_injector.injector import generate_region_scenario

__all__ = [
    "EnvelopeStore",
    "load_envelopes",
    "TierDecision",
    "route",
    "sample_envelope",
    "sample_scenario_params",
    "parse_scenario_request",
    "generate_region_scenario",
]
