"""Public-trace ingestion + replay framework for Aurelius.

BurstGPT (CANONICAL_TRACE_BACKTEST_BURSTGPT_V1) and Azure LLM
(CANONICAL_TRACE_BACKTEST_AZURE_LLM_V1) inference traces are implemented. The
shared ``schema.NormalizedLLMRequest`` contract is designed so the remaining
datasets in ``docs/PUBLIC_TRACE_BACKTESTS.md`` can normalize into the same
record without changing the replay / backtest layers.

Nothing here is a production-savings claim: a public trace is replayed serving
traffic, not customer telemetry.
"""

from .schema import (
    LOG_TYPE_API,
    LOG_TYPE_CONVERSATION,
    NormalizedLLMRequest,
    TraceSchemaError,
    TraceSummary,
    summarize_trace,
    validate_columns,
)

__all__ = [
    "NormalizedLLMRequest",
    "TraceSchemaError",
    "TraceSummary",
    "summarize_trace",
    "validate_columns",
    "LOG_TYPE_API",
    "LOG_TYPE_CONVERSATION",
]
