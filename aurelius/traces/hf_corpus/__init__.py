"""Federated Hugging Face benchmark corpus for Aurelius.

This package implements the **federated benchmark corpus** described in
``docs/HF_DATASET_REGISTRY.md``: each public Hugging Face dataset stays
separate, is classified into a canonical trace type, normalises into an
explicit canonical record schema with provenance + field-quality labels, and
is only promoted into Aurelius' backtest / training-priors / dynamic-
calibration corpora after passing schema + bounded-size + license gates.

Public surface:

- ``schemas`` — canonical record types (one per canonical trace type).
- ``discovery`` — HF metadata client + scoring + classification.
- ``ingestion`` — bounded ingestion + summary writer.
- ``promotion`` — promotion gates + registry writer.
- ``evaluation`` — compatibility-routed evaluation harness.

Nothing here modifies the robust energy engine, the static or dynamic safe
utilisation frontier, the production scheduler, or any controller. Nothing
here is a production telemetry source — see
``docs/PUBLIC_TRACE_BACKTESTS.md`` for the trust hierarchy.
"""

from __future__ import annotations

from . import discovery, evaluation, ingestion, promotion, schemas

__all__ = [
    "schemas",
    "discovery",
    "ingestion",
    "promotion",
    "evaluation",
]
