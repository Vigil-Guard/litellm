"""
This module declares every span the instrumentation can emit and the hierarchy.

Span-name patterns live here as typed builder functions.

Canonical hierarchy::

    PROXY_REQUEST  (SERVER, root)   # owned by the FastAPI instrumentor
    ├── LLM_CALL   (CLIENT)
    ├── GUARDRAIL  (INTERNAL)   # request-lifecycle hook, sibling of LLM_CALL
    ├── DB_CALL    (CLIENT)     # outbound datastore call (redis/postgres)
    └── SERVICE    (INTERNAL)   # litellm-internal work (router, budget jobs, …)

Guardrails parent to PROXY_REQUEST, not LLM_CALL: pre/during/post-call guardrail
hooks are orchestrated by the request lifecycle (a pre-call guardrail runs
before the LLM call even starts), so a guardrail is a sibling of the LLM call,
not a child of it. The emitter parents every span to the ambient OTel context
(the active server span), which matches this.

Service calls are split into two roles by :func:`service_kind`. An outbound call
to an external datastore (redis, postgres) is a ``DB_CALL`` — a CLIENT span that
carries ``db.*`` semconv attributes. Genuinely internal litellm operations
(``self``, ``router``, budget/reset jobs, the pod-lock manager, in-memory
queues) are ``SERVICE`` — INTERNAL spans. Both are built from the same
``ServiceSpanData``; only the role (hence span kind and attribute vocabulary)
differs. Unlike the LLM-call and guardrail spans, a service call can fire
outside any request (a background job), in which case it parents to no server
span and starts its own root trace rather than being dropped.

Management/admin endpoints are ordinary FastAPI routes — their SERVER spans are
owned by the instrumentor too, so they don't appear as a role here.
"""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.integrations.otel.payloads import (
        GuardrailSpanData,
        LLMCallSpanData,
        ProxyRequestSpanData,
        ServiceSpanData,
    )


class SpanRole(str, Enum):
    PROXY_REQUEST = "proxy_request"
    LLM_CALL = "llm_call"
    GUARDRAIL = "guardrail"
    DB_CALL = "db_call"
    SERVICE = "service"


class LiteLLMSpanKind(str, Enum):
    SERVER = "server"
    CLIENT = "client"
    INTERNAL = "internal"
    PRODUCER = "producer"
    CONSUMER = "consumer"


@dataclass(frozen=True)
class SpanSpec:
    role: SpanRole
    kind: LiteLLMSpanKind
    parent: SpanRole | None


SPAN_REGISTRY: dict[SpanRole, SpanSpec] = {
    SpanRole.PROXY_REQUEST: SpanSpec(
        SpanRole.PROXY_REQUEST, LiteLLMSpanKind.SERVER, parent=None
    ),
    SpanRole.LLM_CALL: SpanSpec(
        SpanRole.LLM_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST
    ),
    SpanRole.GUARDRAIL: SpanSpec(
        SpanRole.GUARDRAIL, LiteLLMSpanKind.INTERNAL, parent=SpanRole.PROXY_REQUEST
    ),
    SpanRole.DB_CALL: SpanSpec(
        SpanRole.DB_CALL, LiteLLMSpanKind.CLIENT, parent=SpanRole.PROXY_REQUEST
    ),
    SpanRole.SERVICE: SpanSpec(
        SpanRole.SERVICE, LiteLLMSpanKind.INTERNAL, parent=SpanRole.PROXY_REQUEST
    ),
}


# ``ServiceTypes`` value -> ``db.system.name``. These are outbound datastore
# calls and become CLIENT ``DB_CALL`` spans; ``redis_``-prefixed names cover the
# redis-backed spend queues. Any service not mapped here is litellm-internal work
# and stays an INTERNAL ``SERVICE`` span. This table is the single source of
# datastore knowledge — both the role classifier and the mapper read it.
_DB_SYSTEM_BY_SERVICE: dict[str, str] = {
    "redis": "redis",
    "postgres": "postgresql",
    "batch_write_to_db": "postgresql",
}


def db_system(service_name: str) -> str | None:
    """The ``db.system.name`` for a datastore service, else ``None``.

    ``None`` means the service is not an outbound datastore call. Redis-backed
    spend queues (``redis_*``) map to ``redis``.
    """
    if service_name in _DB_SYSTEM_BY_SERVICE:
        return _DB_SYSTEM_BY_SERVICE[service_name]
    if service_name.startswith("redis_"):
        return "redis"
    return None


def service_kind(service_name: str) -> SpanRole:
    """Role for a service call: ``DB_CALL`` for datastores, else ``SERVICE``."""
    return SpanRole.DB_CALL if db_system(service_name) is not None else SpanRole.SERVICE


# --- span name builders (the naming convention, per role) ------------------- #


def llm_call_span_name(data: "LLMCallSpanData") -> str:
    """``"{operation} {model}"`` e.g. ``"chat gpt-4o"`` (GenAI semconv)."""
    model = data.request_model or ""
    return f"{data.operation.value} {model}".strip()


def proxy_request_span_name(data: "ProxyRequestSpanData") -> str:
    """``"{method} {route}"`` (HTTP semconv)."""
    return f"{data.http_method} {data.route}".strip()


def guardrail_span_name(data: "GuardrailSpanData") -> str:
    return f"execute_guardrail {data.guardrail_name}".strip()


def service_span_name(data: "ServiceSpanData") -> str:
    """``"{service} {call_type}"`` e.g. ``"redis set"`` — service name alone when
    no call type is known, so identically-named calls stay distinguishable."""
    return f"{data.service_name} {data.call_type or ''}".strip()


def root_roles() -> list[SpanRole]:
    """Roles that start a new trace (no in-process parent)."""
    return [role for role, spec in SPAN_REGISTRY.items() if spec.parent is None]


def child_roles(parent: SpanRole) -> list[SpanRole]:
    return [role for role, spec in SPAN_REGISTRY.items() if spec.parent == parent]


def validate_registry(
    registry: dict[SpanRole, SpanSpec] | None = None,
) -> None:
    reg = registry if registry is not None else SPAN_REGISTRY
    for role, spec in reg.items():
        if spec.role is not role:
            raise ValueError(f"SPAN_REGISTRY[{role}] has mismatched role {spec.role}")
        if spec.parent is not None and spec.parent not in reg:
            raise ValueError(f"span role {role} declares unknown parent {spec.parent}")
    missing = [role for role in SpanRole if role not in reg]
    if missing:
        raise ValueError(f"SPAN_REGISTRY is missing roles: {missing}")
