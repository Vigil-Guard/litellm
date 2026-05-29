"""Trace-context + Baggage helpers."""

from typing import Mapping

from opentelemetry import baggage
from opentelemetry.context import Context, get_current
from opentelemetry.trace import Span, get_current_span, set_span_in_context
from opentelemetry.trace.propagation.tracecontext import (
    TraceContextTextMapPropagator,
)

_PROPAGATOR = TraceContextTextMapPropagator()


def set_request_baggage(
    values: Mapping[str, str], context: Context | None = None
) -> Context:
    """Return a context with ``values`` written into Baggage."""
    ctx = context
    for key, value in values.items():
        ctx = baggage.set_baggage(key, value, context=ctx)
    return ctx if ctx is not None else (context or get_current())


def get_baggage_attributes(context: Context | None = None) -> dict[str, str]:
    """All Baggage entries on ``context`` as strings."""
    return {key: str(value) for key, value in baggage.get_all(context).items()}


def context_from_span(span: Span, context: Context | None = None) -> Context:
    """A context with ``span`` as the active span (for explicit parenting)."""
    return set_span_in_context(span, context=context)


def resolve_parent_context(
    threaded: Span | None = None, *, prefer_threaded: bool = False
) -> Context:
    """The context a child span should parent under.

    By default this is ambient-first: parent to the active OTel context (the
    server span, restored by the logging worker or active in the request task),
    falling back to the span the proxy threaded explicitly
    (``litellm_parent_otel_span``) only when the ambient context has no recordable
    span — e.g. logging dispatched from a detached ``asyncio.create_task``, or a
    background service call. When neither is recordable the ambient context is
    returned unchanged, so the span starts a new root trace.

    ``prefer_threaded`` flips the precedence: when the threaded span is recordable
    it wins over ambient. Use this for **request-level** spans (the LLM call, the
    guardrail) that must hang off the server span regardless of what phase span
    happens to be active — otherwise a span emitted while the ``auth`` phase span
    is active (e.g. failure logging) would nest under ``auth`` instead of the
    request root. Service/DB spans deliberately do *not* set this, so they still
    nest under whatever phase is active.
    """
    ctx = get_current()
    ambient_ok = is_recordable_span(get_current_span(ctx))
    if is_recordable_span(threaded) and (prefer_threaded or not ambient_ok):
        ctx = context_from_span(threaded, context=ctx)  # type: ignore[arg-type]
    return ctx


def is_recordable_span(obj: object) -> bool:
    """True if ``obj`` is a live span with a valid context (safe to parent under)."""
    if not isinstance(obj, Span):
        return False
    try:
        ctx = obj.get_span_context()
    except Exception:
        return False
    return ctx is not None and ctx.is_valid


def extract_traceparent(headers: Mapping[str, str]) -> Context | None:
    """Extract a remote parent context from incoming HTTP headers, if present."""
    if not any(key.lower() == "traceparent" for key in headers):
        return None
    carrier = {str(key).lower(): value for key, value in headers.items()}
    return _PROPAGATOR.extract(carrier)
