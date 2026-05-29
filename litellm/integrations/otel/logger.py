"""``CustomLogger`` adapter on the OpenTelemetry span engine.

Thin adapter: it translates litellm's logging callbacks into typed ``*SpanData``
and hands them to the engine (:mod:`emitter`), with multi-tenant tracer routing
in :mod:`routing`. It emits the gen-ai spans (LLM call, guardrail, service).

The proxy server span is NOT owned here. It is created by the FastAPI
instrumentation mounted in ``proxy_server``'s startup event, which stamps the
``http.*`` attributes and handles inbound context propagation. The proxy-span
methods below are therefore no-ops: routes never modify spans.

Gen-ai spans parent to that server span via the ambient OTel context rather
than an explicitly threaded span. litellm's async logging worker copies the
request's context at enqueue time, so ``async_log_success_event`` runs with the
server span active. Emission is therefore async-only — the sync callback runs
in an out-of-context thread, where there is no parent span, so it is a no-op.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any, Iterator, Mapping, cast

from opentelemetry.context import attach, get_current
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Span, Tracer, get_current_span, use_span

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.integrations.otel.baggage import promoted_baggage
from litellm.integrations.otel.config import OpenTelemetryV2Config
from litellm.integrations.otel.context import (
    is_recordable_span,
    resolve_parent_context,
    set_request_baggage,
)
from litellm.integrations.otel.emitter import SpanEmitter
from litellm.integrations.otel.mappers import resolve_mappers
from litellm.integrations.otel.payloads import (
    GuardrailSpanData,
    LLMCallSpanData,
    RequestIdentity,
    ServiceSpanData,
    SpanError,
)
from litellm.integrations.otel.providers import build_tracer_provider, get_tracer
from litellm.integrations.otel.routing import TenantTracerCache
from litellm.integrations.otel.spans import SpanRole, span_role_for_service
from litellm.integrations.otel.utils import to_ns

if TYPE_CHECKING:
    from litellm.types.utils import StandardLoggingGuardrailInformation

LITELLM_TRACER_NAME = "litellm"
LITELLM_PROXY_REQUEST_SPAN_NAME = "Received Proxy Server Request"

# Any callback whose class belongs to one of these modules is "the OTel
# callback" for proxy-global-registration purposes.
_OTEL_MODULES = (
    "litellm.integrations.otel",
    "litellm.integrations.opentelemetry",
)


def _threaded_parent_span(kwargs: Mapping[str, Any]) -> Span | None:
    """The proxy SERVER span threaded through request metadata, if any.

    Normally the LLM-call span parents to the ambient OTel context (the active
    server span). But pass-through logging runs in a detached
    ``asyncio.create_task`` whose copied context may no longer carry that span,
    so the proxy also threads it explicitly as ``litellm_parent_otel_span`` (see
    ``litellm_pre_call_utils`` for proxy routes and the pass-through endpoint for
    catch-all routes). This reads it back so the call span can fall back to it.
    """
    litellm_params = kwargs.get("litellm_params")
    candidates: list[Any] = []
    if isinstance(litellm_params, Mapping):
        candidates.append(litellm_params.get("metadata"))
        candidates.append(litellm_params.get("litellm_metadata"))
    candidates.append(kwargs.get("metadata"))
    for meta in candidates:
        if isinstance(meta, Mapping):
            span = meta.get("litellm_parent_otel_span")
            if span is not None:
                return cast("Span", span)
    return None


def _pre_call_guardrail_blocked(payload: Mapping[str, Any]) -> bool:
    """True when a pre-call guardrail blocked the request (no LLM call happened).

    A blocked pre-call guardrail raises before the upstream call, yet litellm
    still emits a failure log — which would otherwise produce a phantom CLIENT
    span for a call that never occurred. We detect the case (request failed AND a
    ``pre_call`` guardrail intervened) so the caller can skip that span. A
    pre-call guardrail that merely *masks* lets the call proceed, so the request
    succeeds and this returns False — only genuine blocks fail the request.
    """
    if payload.get("status") != "failure":
        return False
    info = payload.get("guardrail_information")
    if not isinstance(info, list):
        return False
    for entry in info:
        if not isinstance(entry, dict):
            continue
        mode = entry.get("guardrail_mode")
        is_pre_call = mode == "pre_call" or (
            isinstance(mode, (list, tuple)) and "pre_call" in mode
        )
        if is_pre_call and entry.get("guardrail_status") == "guardrail_intervened":
            return True
    return False


def _rejected_before_llm_call(payload: Mapping[str, Any]) -> bool:
    """True when the proxy rejected the request before any upstream LLM call.

    Auth / budget / rate-limit / blocked-route rejections happen at the proxy
    gate and raise ``ProxyException`` — yet litellm still emits a failure log,
    which would otherwise produce a phantom CLIENT ``chat …`` span for a call
    that never reached the model (e.g. a 401). LLM provider errors use litellm's
    own exception classes (``RateLimitError``, ``APIError``, …) and record the
    ``api_base`` they hit, so the guards below keep a genuinely-attempted call —
    even one the proxy later wrapped in a ``ProxyException`` — from being skipped.
    """
    if payload.get("status") != "failure":
        return False
    info = payload.get("error_information")
    error_class = info.get("error_class") if isinstance(info, Mapping) else None
    if error_class != "ProxyException":
        return False
    response = payload.get("response")
    has_response = isinstance(response, Mapping) and bool(response.get("id"))
    hidden = payload.get("hidden_params")
    api_base = payload.get("api_base") or (
        hidden.get("api_base") if isinstance(hidden, Mapping) else None
    )
    return not has_response and not api_base


class OpenTelemetryV2(CustomLogger):
    """The ``CustomLogger`` for OpenTelemetry.

    The constructor accepts an optional config, callback name, and pre-built
    OTel providers; when a provider is omitted it is built from the config.
    ``logger_provider`` and ``meter_provider`` are accepted but reserved for
    future OTel logs and metrics support.
    """

    def __init__(
        self,
        config: OpenTelemetryV2Config | None = None,
        callback_name: str | None = None,
        tracer_provider: TracerProvider | None = None,
        logger_provider: Any | None = None,  # reserved for OTel logs
        meter_provider: Any | None = None,  # reserved for metrics
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        # Build the config from any settings passed through ``kwargs`` so
        # ``callback_settings.otel.*`` in config.yaml (e.g. ``baggage_promoted_keys``,
        # ``capture_message_content``) configures the logger. ``OpenTelemetryV2Config``
        # ignores extra keys, so unrelated kwargs are dropped harmlessly.
        self.config: OpenTelemetryV2Config = config or OpenTelemetryV2Config(**kwargs)
        self.callback_name = callback_name
        self._tracer_provider: TracerProvider = (
            tracer_provider
            if tracer_provider is not None
            else build_tracer_provider(self.config)
        )
        self.tracer: Tracer = get_tracer(self._tracer_provider, LITELLM_TRACER_NAME)
        self._emitter = SpanEmitter(
            self.tracer, self.config, mappers=resolve_mappers(self.config.mapper_names)
        )
        self._tenant_tracers = TenantTracerCache(
            self.config, callback_name, LITELLM_TRACER_NAME
        )
        self._init_otel_logger_on_litellm_proxy()

    # ====================================================================== #
    #  Proxy global registration
    # ====================================================================== #

    def _init_otel_logger_on_litellm_proxy(self) -> None:
        """Claim ``proxy_server.open_telemetry_logger`` if no one else has."""
        try:
            from litellm.proxy import proxy_server
        except Exception:
            return
        try:
            # Mutate ``litellm.service_callback`` in place. ``getattr(..) or []``
            # would bind a throwaway local when the list is empty (an empty list
            # is falsy), so the append would never reach the global and service
            # spans (Redis, Postgres, …) would be silently dropped from traces.
            service_callback = litellm.service_callback
            already_otel = any(
                cb.__class__.__module__.startswith(_OTEL_MODULES)
                for cb in service_callback
                if hasattr(cb, "__class__")
            )
            if not already_otel:
                service_callback.append(self)
        except Exception:
            pass
        if getattr(proxy_server, "open_telemetry_logger", None) is None:
            setattr(proxy_server, "open_telemetry_logger", self)

    # ====================================================================== #
    #  LLM-call callbacks
    # ====================================================================== #

    # Async-only: the async path runs inside the request's restored OTel context
    # (the logging worker copies it at enqueue), so the span parents to the
    # instrumentor's server span via ambient context. The sync path runs in an
    # out-of-context thread with no parent span, so it is a no-op.

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        return None

    def log_failure_event(self, kwargs, response_obj, start_time, end_time):
        return None

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        self._emit_llm_call(kwargs, start_time, end_time)

    def _emit_llm_call(
        self,
        kwargs: Mapping[str, Any],
        start_time: datetime | float | None,
        end_time: datetime | float | None,
    ) -> Span | None:
        payload = kwargs.get("standard_logging_object")
        if not payload:
            return None
        mapping_payload = cast("Mapping[str, Any]", payload)
        if _pre_call_guardrail_blocked(mapping_payload):
            # A pre-call guardrail blocked the request, so the upstream LLM was
            # never called — litellm still emits a failure log, but a CLIENT
            # "chat …" span for a call that didn't happen is misleading. Skip it;
            # the guardrail span (ERROR, with the verdict) is the real outcome.
            return None
        if _rejected_before_llm_call(mapping_payload):
            # Rejected at the proxy gate (auth/budget/rate-limit) before any LLM
            # call — the failure log would otherwise produce a phantom CLIENT span.
            # The server span (and auth span) already carry the failure.
            return None
        data = LLMCallSpanData.from_standard_logging_payload(
            cast("Any", payload), capture_content=self.config.capture_span_content
        )
        # The LLM call is a request-level span: parent it to the server span the
        # proxy threads through metadata (``prefer_threaded``), not to whatever
        # phase span is ambient. Otherwise an LLM-call log emitted while the
        # ``auth`` phase span is active — e.g. failure logging for a request
        # rejected at auth — would nest under ``auth`` instead of the request
        # root. Falls back to ambient when nothing is threaded.
        parent_ctx = resolve_parent_context(
            threaded=_threaded_parent_span(kwargs), prefer_threaded=True
        )
        # Write identity into Baggage so child spans (guardrails, services)
        # inherit it.
        bag = promoted_baggage(
            data.identity,
            data.request_model,
            promoted_keys=tuple(self.config.baggage_promoted_keys),
            metadata_keys=tuple(self.config.baggage_metadata_keys),
        )
        if bag:
            parent_ctx = set_request_baggage(bag, context=parent_ctx)
        return self._emitter.emit(
            SpanRole.LLM_CALL,
            data,
            parent_context=parent_ctx,
            start_time_ns=to_ns(start_time),
            end_time_ns=to_ns(end_time),
            tracer=self._tenant_tracers.tracer_for(
                self.tracer, kwargs.get("standard_callback_dynamic_params")
            ),
        )

    # ====================================================================== #
    #  Service hooks
    # ====================================================================== #

    async def async_service_success_hook(
        self,
        payload: Any,
        parent_otel_span: Span | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict | None = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=None,
        )

    async def async_service_failure_hook(
        self,
        payload: Any,
        error: str | None = "",
        parent_otel_span: Span | None = None,
        start_time: datetime | float | None = None,
        end_time: datetime | float | None = None,
        event_metadata: dict | None = None,
    ) -> None:
        self._emit_service(
            payload,
            parent_otel_span=parent_otel_span,
            start_time=start_time,
            end_time=end_time,
            event_metadata=event_metadata,
            error_override=error or "error",
        )

    def _emit_service(
        self,
        payload: Any,
        *,
        parent_otel_span: Span | None,
        start_time: datetime | float | None,
        end_time: datetime | float | None,
        event_metadata: dict | None,
        error_override: str | None,
    ) -> Span | None:
        data = ServiceSpanData.from_payload(payload, event_metadata=event_metadata)
        # Decide whether this service call is a span at all, and of what kind.
        # ``None`` means metrics-only (framework instrumentation that duplicates a
        # gen-AI span — ``self``/``router``/``proxy_pre_call`` — or ``auth``, which
        # gets a live phase span instead). Those still feed Prometheus/Datadog via
        # their own hooks; they just never enter the trace.
        role = span_role_for_service(data.service_name)
        if role is None:
            return None
        # A metrics-only ping with neither timing nor a parent (in-memory queue
        # gauges) is not a traceable operation; a span for it would be a
        # zero-duration root with no context, so skip it. Real background work
        # (budget/reset jobs, spend flush) passes start/end times and still emits
        # as a root; anything with a parent emits regardless.
        if (
            error_override is None
            and start_time is None
            and end_time is None
            and parent_otel_span is None
        ):
            return None
        if error_override is not None and data.error is None:
            data = ServiceSpanData(
                service_name=data.service_name,
                call_type=data.call_type,
                error=SpanError(message=error_override),
                event_metadata=data.event_metadata,
            )
        # Parent like every other span: ambient context first (so identity Baggage
        # rides along and the call nests under whatever request phase is active —
        # e.g. a DB lookup under the live ``auth`` span), falling back to the
        # server span the proxy threaded as ``parent_otel_span``. A background
        # service call has neither, so it starts its own root trace.
        parent_context = resolve_parent_context(threaded=parent_otel_span)
        return self._emitter.emit(
            role,
            data,
            parent_context=parent_context,
            start_time_ns=to_ns(start_time),
            end_time_ns=to_ns(end_time),
        )

    # ====================================================================== #
    #  async_post_call_* hooks — emit guardrail spans. The server span's status
    #  / errors are the FastAPI instrumentor's job, so we don't touch it here.
    # ====================================================================== #

    def seed_request_identity(self, user_api_key_dict: Any, model: Any = None) -> None:
        """Attach request-identity Baggage to the current context + server span.

        Seeding identity into Baggage makes **every** span emitted afterwards for
        this request — LLM call, guardrail, DB call — inherit it via
        ``LiteLLMBaggageSpanProcessor``. Called once at the auth boundary (as soon
        as the key resolves) so post-auth spans are labeled consistently; the
        Baggage rides the request task's contextvar from there on. Auth-internal
        DB lookups that run before the key is known stay unlabeled — identity
        isn't determined yet, which is correct.
        """
        try:
            identity = RequestIdentity.from_user_api_key_auth(user_api_key_dict)
            bag = promoted_baggage(
                identity,
                model,
                promoted_keys=tuple(self.config.baggage_promoted_keys),
                metadata_keys=tuple(self.config.baggage_metadata_keys),
            )
            if bag:
                # Attach (no detach): the contextvar is scoped to this request's
                # asyncio task and is reclaimed when the task ends.
                attach(set_request_baggage(bag, context=get_current()))
                # The server span was started by the instrumentor before this ran,
                # so the Baggage processor (which only fires at span start) won't
                # backfill it — stamp identity on it directly.
                server_span = get_current_span()
                if is_recordable_span(server_span):
                    for key, value in bag.items():
                        server_span.set_attribute(key, value)
        except Exception:
            pass

    @contextmanager
    def start_phase_span(self, name: str) -> "Iterator[Span]":
        """Open a live, **active** INTERNAL span for a request phase (e.g. auth).

        Unlike the post-hoc service spans (emitted from start/end timestamps after
        the fact), this span is the active OTel context for the duration of the
        ``with`` block. Service/DB calls fired inside it — even via
        ``asyncio.create_task``, which copies the active context — therefore nest
        under it instead of flattening onto the server span.
        """
        span = self._emitter.start_span(SpanRole.SERVICE, name)
        with use_span(span, end_on_exit=True):
            yield span

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> dict:
        """Re-seed identity Baggage in the request task.

        Identity is first seeded at the auth boundary (``seed_request_identity``),
        but this hook re-seeds with the request ``model`` now known and covers
        entrypoints that don't pass through that boundary (e.g. the SDK). Idempotent.
        """
        self.seed_request_identity(
            user_api_key_dict,
            model=data.get("model") if isinstance(data, dict) else None,
        )
        return data

    async def async_post_call_success_hook(
        self,
        data: Mapping[str, Any],
        user_api_key_dict: Any,
        response: Any,
    ) -> Any:
        self._emit_guardrail_spans(data)
        return response

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, Any],
        original_exception: BaseException | None,
        user_api_key_dict: Any,
        traceback_str: str | None = None,
    ) -> None:
        self._emit_guardrail_spans(request_data)

    def _emit_guardrail_spans(self, request_data: Mapping[str, Any]) -> None:
        # The guardrail is a request-level span: parent it to the server span the
        # proxy threaded (``prefer_threaded``), not to whatever phase span is
        # ambient. Emit each with the guardrail's actual execution window so a
        # pre_call guardrail is placed before the LLM call instead of at post-call
        # emission time.
        metadata = request_data.get("metadata")
        guardrails: list[Any] = []
        if isinstance(metadata, dict):
            info = metadata.get("standard_logging_guardrail_information")
            if isinstance(info, list):
                guardrails = info
            elif isinstance(info, dict):
                guardrails = [info]
        if not guardrails:
            return
        parent_ctx = resolve_parent_context(
            threaded=_threaded_parent_span(request_data), prefer_threaded=True
        )
        for entry in guardrails:
            if not isinstance(entry, dict):
                continue
            data = GuardrailSpanData.from_logging_entry(
                cast("StandardLoggingGuardrailInformation", entry)
            )
            self._emitter.emit(
                SpanRole.GUARDRAIL,
                data,
                parent_context=parent_ctx,
                start_time_ns=to_ns(data.start_time),
                end_time_ns=to_ns(data.end_time),
            )

    # ====================================================================== #
    #  Management endpoint hooks — no-ops. Management endpoints are ordinary
    #  FastAPI routes, so the mounted instrumentor already spans them.
    # ====================================================================== #

    async def async_management_endpoint_success_hook(
        self,
        logging_payload: Any,
        parent_otel_span: Span | None = None,
    ) -> None:
        return None

    async def async_management_endpoint_failure_hook(
        self,
        logging_payload: Any,
        parent_otel_span: Span | None = None,
    ) -> None:
        return None

    # ====================================================================== #
    #  Proxy SERVER-span API — no-ops. The FastAPI instrumentor owns the server
    #  span (creation, http.* attributes, inbound propagation) and gen-ai spans
    #  parent to it via ambient context. These methods are the surface the
    #  proxy and auth call sites invoke; they intentionally do nothing.
    # ====================================================================== #

    def create_litellm_proxy_request_started_span(
        self, start_time: datetime, headers: Mapping[str, str] | None
    ) -> Span | None:
        """Return the active server span instead of creating one.

        The FastAPI instrumentor owns the server span, so V2 creates nothing
        here. But the proxy threads this return value as ``litellm_parent_otel_span``
        — and service logging (Redis, Postgres, …) only invokes the OTel service
        hook when that parent is non-None. Returning the ambient server span lets
        service spans nest under it. The proxy must NOT ``.end()`` this span (the
        instrumentor does); ``_close_dangling_otel_server_span`` skips it under V2.
        """
        span = get_current_span()
        return span if is_recordable_span(span) else None

    @staticmethod
    def set_proxy_request_route_attributes(
        span: Span | None,
        *,
        url_path: str | None = None,
        http_route: str | None = None,
    ) -> None:
        """No-op: the FastAPI instrumentor stamps ``http.route`` / ``url.path``."""

    @staticmethod
    def set_response_status_code_attribute(
        span: Span | None, status_code: int | None
    ) -> None:
        """No-op: the FastAPI instrumentor stamps ``http.response.status_code``."""

    @staticmethod
    def set_preprocessing_duration_attribute(span: Span | None, container: Any) -> None:
        """No-op: the server span belongs to the FastAPI instrumentor."""


# ====================================================================== #
#  Module-level seam for proxy-core call sites (auth, …). These resolve the
#  registered V2 logger and no-op when V2 is not the active logger, so the
#  proxy can call them unconditionally without importing the OTel SDK or
#  knowing whether V2 is enabled.
# ====================================================================== #


def _registered_v2_logger() -> "OpenTelemetryV2 | None":
    """The proxy's registered logger if it is the V2 ``OpenTelemetryV2``, else None."""
    try:
        from litellm.proxy import proxy_server
    except Exception:
        return None
    logger = getattr(proxy_server, "open_telemetry_logger", None)
    return logger if isinstance(logger, OpenTelemetryV2) else None


def seed_request_identity(user_api_key_dict: Any, model: Any = None) -> None:
    """Seed request-identity Baggage at the auth boundary (no-op without V2)."""
    logger = _registered_v2_logger()
    if logger is not None:
        logger.seed_request_identity(user_api_key_dict, model=model)


@contextmanager
def phase_span(name: str) -> "Iterator[Span | None]":
    """Run a request phase inside a live active span so its DB/service calls nest.

    A no-op (yields ``None``) when V2 is not the active logger, so proxy-core
    call sites can wrap a phase unconditionally.
    """
    logger = _registered_v2_logger()
    if logger is None:
        yield None
        return
    with logger.start_phase_span(name) as span:
        yield span
