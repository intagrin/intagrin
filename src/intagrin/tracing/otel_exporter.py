"""OpenTelemetry GenAI semantic-convention span exporter for IntaGrin's own runtime events.

`telemetry: ["otel"]` in ai.yaml already exists and already works today — it's wired (see
`RuntimeEngine.initialize()`) via `litellm.success_callback`/`failure_callback`, which instruments
*litellm's own LLM API calls only*. It has zero visibility into IntaGrin-specific concepts: tool
calls, router decisions, handoffs, agent spawning, MCP background tasks. This module fills that
gap. It is activated by the exact same `"otel" in telemetry` check and runs additively alongside
the existing litellm callback — it does not touch or replace that wiring.

Mechanism: subscribes to `tracing.console.EventStreamer` (the same mechanism
`server/monitor.py`'s SSE endpoint already uses) and translates every event type IntaGrin emits
into an OTel span, using the OpenTelemetry GenAI semantic conventions (`gen_ai.*`) wherever a
standard attribute exists (model, token usage), and vendor-namespaced `intagrin.*` attributes for
everything else (router decisions, handoffs, agent spawns, MCP tasks — none of which have a
GenAI-semconv equivalent). This is the one interception point that sees every event type without
touching any of the many scattered `Tracer`/`EventStreamer` call sites in `runtime/engine.py`.

Deliberately does NOT attempt to merge or parent its span tree with the spans litellm's own otel
callback emits — the two run independently and additively. Both can point at the same OTLP
endpoint; a backend will show them as separate traces correlated only by shared
`intagrin.session_id`/service-name attributes, not a single merged span tree. Building genuine
cross-library span parenting would require depending on litellm's internal instrumentation
details, which is a fragile trade for a marginal UX gain.

Exporter selection: `ConsoleSpanExporter` (bundled in the core `opentelemetry-sdk` dependency —
already an unconditional hard dependency of this package — zero extra install) unless the standard
`OTEL_EXPORTER_OTLP_ENDPOINT` environment variable is set, in which case the OTLP/HTTP exporter is
used instead. That exporter's package (`opentelemetry-exporter-otlp-proto-http`) is lazily
imported, gated behind the optional `intagrin[otel]` extra, so a project that never sets the
endpoint never needs it installed. Setting the endpoint without the package installed raises
`IG-RT-009` rather than silently falling back to console output — silently downgrading telemetry a
user explicitly asked for would hide the misconfiguration rather than surface it.
"""

import asyncio
import os
from typing import Any

from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
from opentelemetry.trace import Span, SpanKind, Status, StatusCode, Tracer as OtelTracer

from ..errors import IntaGrinError

_started = False
_tracer_provider: TracerProvider | None = None


def _build_tracer_provider() -> TracerProvider:
    provider = TracerProvider(resource=Resource.create({"service.name": "intagrin"}))
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    if endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        except ImportError:
            raise IntaGrinError(
                "IG-RT-009",
                "OTEL_EXPORTER_OTLP_ENDPOINT is set but 'opentelemetry-exporter-otlp-proto-http' "
                'isn\'t installed. Run: pip install "intagrin[otel]"',
            )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    else:
        provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
    return provider


def _provider_from_model(model: str | None) -> str:
    """LiteLLM model strings are provider-prefixed (e.g. "anthropic/claude-...",
    "gemini/gemini-..."); gen_ai.system wants just the provider. A model string with no '/'
    prefix (a bare custom/self-hosted model name) is used as-is; a missing model is "unknown"."""
    if not model:
        return "unknown"
    return model.split("/", 1)[0] if "/" in model else model


_OTEL_PRIMITIVE_TYPES = (str, bool, int, float)


def _coerce_attr(value: Any) -> str | bool | int | float:
    """OTel span attributes must be a primitive or a homogeneous sequence of primitives — several
    IntaGrin event payloads carry nested dicts/lists (e.g. agent_spawned's full config snapshot).
    Anything not already a safe primitive is stringified rather than dropped, so no event data is
    silently lost from the exported span even though it loses structure."""
    if isinstance(value, _OTEL_PRIMITIVE_TYPES):
        return value
    return str(value)


def _emit_span(
    tracer: OtelTracer, name: str, attributes: dict[str, Any], error: str | None = None
) -> None:
    safe_attrs = {k: _coerce_attr(v) for k, v in attributes.items() if v is not None}
    span: Span = tracer.start_span(name, kind=SpanKind.INTERNAL, attributes=safe_attrs)
    if error:
        span.set_status(Status(StatusCode.ERROR, str(error)))
        span.set_attribute("error.message", str(error))
    span.end()


def _handle_event(tracer: OtelTracer, event: dict) -> None:
    event_type = event.get("type")
    data = event.get("data") or {}
    context = event.get("context") or {}
    session_id = context.get("session_id")
    agent = context.get("agent")
    base = {"intagrin.session_id": session_id, "intagrin.agent": agent}

    if event_type == "cost":
        # Tracer.log_cost's model/prompt_tokens/completion_tokens fields (see tracing/console.py)
        # are what make this single event self-sufficient for a complete GenAI-semconv span —
        # the sibling `llm_exchange` event carries the model too, but not the token split.
        model = data.get("model")
        _emit_span(
            tracer,
            "intagrin.llm_call",
            {
                **base,
                "gen_ai.system": _provider_from_model(model),
                "gen_ai.request.model": model,
                "gen_ai.usage.input_tokens": data.get("prompt_tokens"),
                "gen_ai.usage.output_tokens": data.get("completion_tokens"),
                "intagrin.cost.usd": data.get("cost"),
            },
        )
    elif event_type == "tool_result":
        # Tracer.log_tool_result's payload is `{"result": ...}` only — no tool name is available
        # at this event today (log_tool_call, which does carry the name, has no call sites in
        # engine.py as of this writing). gen_ai.tool.name is therefore omitted rather than
        # fabricated; this is a real, documented limitation of the current event, not an oversight.
        _emit_span(
            tracer,
            "intagrin.tool_result",
            {**base, "intagrin.tool.result_preview": str(data.get("result"))[:200]},
        )
    elif event_type == "router_decision":
        _emit_span(
            tracer,
            "intagrin.router_decision",
            {
                **base,
                "intagrin.router.kind": data.get("kind"),
                "intagrin.router.description": data.get("description"),
                "intagrin.router.fired": data.get("fired"),
                "intagrin.router.target": data.get("target"),
            },
            error=data.get("error"),
        )
    elif event_type == "handoff":
        _emit_span(
            tracer,
            "intagrin.handoff",
            {
                **base,
                "intagrin.handoff.from": data.get("from"),
                "intagrin.handoff.to": data.get("to"),
                "intagrin.handoff.mechanism": data.get("mechanism"),
                "intagrin.handoff.reason": data.get("reason") or data.get("condition"),
            },
        )
    elif event_type in ("agent_spawned", "agent_retired"):
        _emit_span(
            tracer,
            f"intagrin.{event_type}",
            {**base, **{f"intagrin.agent.{k}": v for k, v in data.items()}},
        )
    elif event_type in ("mcp_task_started", "mcp_task_completed"):
        _emit_span(
            tracer,
            f"intagrin.{event_type}",
            {**base, **{f"intagrin.mcp_task.{k}": v for k, v in data.items()}},
        )
    elif event_type == "error":
        _emit_span(tracer, "intagrin.error", base, error=data.get("message"))
    # "step" and "llm_exchange" carry no attributes not already captured more precisely by other
    # event types above (llm_exchange's model is redundant with cost's, and its full prompt/
    # response content is deliberately not exported as span attributes — that's exactly the kind
    # of high-cardinality, potentially sensitive payload OTel backends warn against putting in
    # attributes rather than a dedicated logging pipeline) — intentionally not spanned.


async def _consume_events(tracer: OtelTracer) -> None:
    from .console import EventStreamer

    queue = EventStreamer.subscribe()
    try:
        while True:
            event = await queue.get()
            try:
                _handle_event(tracer, event)
            except Exception:
                # Best-effort, same fail-open convention as every other tracing sink in this
                # codebase (Tracer.log_error itself must never raise) — a malformed event or a
                # transient exporter hiccup must never break the turn that produced it.
                pass
    finally:
        EventStreamer.unsubscribe(queue)


def ensure_started(telemetry_options: list[str]) -> None:
    """Idempotently starts the background EventStreamer-to-OTel-span consumer the first time a
    project with `telemetry: ["otel"]` initializes an engine in this process. Safe to call from
    every `RuntimeEngine.initialize()` (every session, every request) — a second call is a no-op,
    matching the idempotency style of `runtime/shared_resources.py`'s `SharedResourcesCache`.
    Raises `IntaGrinError("IG-RT-009", ...)` synchronously (before starting the consumer) if an
    OTLP endpoint was requested but its exporter package isn't installed, so misconfiguration
    surfaces immediately at startup rather than silently inside a background task."""
    global _started, _tracer_provider
    if _started or "otel" not in telemetry_options:
        return
    _tracer_provider = _build_tracer_provider()
    tracer = _tracer_provider.get_tracer("intagrin")
    asyncio.create_task(_consume_events(tracer))
    _started = True


def reset_for_tests() -> None:
    """Test-only: lets test_otel_exporter.py exercise ensure_started's idempotency gate and the
    IG-RT-009 path repeatedly within one process, since the module-level _started flag would
    otherwise only ever fire once per test session."""
    global _started, _tracer_provider
    _started = False
    _tracer_provider = None
