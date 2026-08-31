import pytest

from intagrin.errors import IntaGrinError
from intagrin.tracing import otel_exporter
from intagrin.tracing.console import EventStreamer, Tracer, clear_trace_context, set_trace_context


@pytest.fixture(autouse=True)
def reset_otel_state(monkeypatch):
    otel_exporter.reset_for_tests()
    clear_trace_context()
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    yield
    otel_exporter.reset_for_tests()
    clear_trace_context()


def _tracer_with_in_memory_exporter():
    """Builds an OTel tracer wired to an InMemorySpanExporter (the OTel SDK's own test double —
    no real network/OTLP call), bypassing ensure_started's real exporter-selection logic so tests
    can assert on captured spans directly."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider.get_tracer("test"), exporter


def test_cost_event_produces_a_span_with_genai_usage_attributes():
    tracer, exporter = _tracer_with_in_memory_exporter()
    set_trace_context(session_id="s1", agent_name="billing")
    event = {
        "type": "cost",
        "data": {
            "tokens": 150,
            "cost": 0.002,
            "model": "anthropic/claude-3-5-sonnet",
            "prompt_tokens": 100,
            "completion_tokens": 50,
        },
        "context": {"session_id": "s1", "agent": "billing"},
    }
    otel_exporter._handle_event(tracer, event)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["gen_ai.system"] == "anthropic"
    assert attrs["gen_ai.request.model"] == "anthropic/claude-3-5-sonnet"
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 50
    assert attrs["intagrin.cost.usd"] == 0.002
    assert attrs["intagrin.session_id"] == "s1"
    assert attrs["intagrin.agent"] == "billing"


def test_cost_event_with_no_slash_prefixed_model_uses_bare_model_as_system():
    tracer, exporter = _tracer_with_in_memory_exporter()
    event = {
        "type": "cost",
        "data": {"tokens": 10, "cost": 0.0, "model": "my-custom-model"},
        "context": {},
    }
    otel_exporter._handle_event(tracer, event)
    assert exporter.get_finished_spans()[0].attributes["gen_ai.system"] == "my-custom-model"


def test_cost_event_with_no_model_uses_unknown_system():
    tracer, exporter = _tracer_with_in_memory_exporter()
    event = {"type": "cost", "data": {"tokens": 10, "cost": 0.0}, "context": {}}
    otel_exporter._handle_event(tracer, event)
    assert exporter.get_finished_spans()[0].attributes["gen_ai.system"] == "unknown"


def test_router_decision_event_produces_span_with_custom_intagrin_attributes():
    tracer, exporter = _tracer_with_in_memory_exporter()
    event = {
        "type": "router_decision",
        "data": {
            "kind": "root",
            "description": "balance check",
            "fired": True,
            "target": "billing",
            "error": None,
        },
        "context": {"session_id": "s2"},
    }
    otel_exporter._handle_event(tracer, event)
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    attrs = spans[0].attributes
    assert attrs["intagrin.router.kind"] == "root"
    assert attrs["intagrin.router.fired"] is True
    assert attrs["intagrin.router.target"] == "billing"
    assert spans[0].status.status_code.name == "UNSET"


def test_router_decision_with_error_sets_error_span_status():
    tracer, exporter = _tracer_with_in_memory_exporter()
    event = {
        "type": "router_decision",
        "data": {"kind": "conditional", "description": "d", "fired": False, "error": "boom"},
        "context": {},
    }
    otel_exporter._handle_event(tracer, event)
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error.message"] == "boom"


def test_handoff_event_produces_span():
    tracer, exporter = _tracer_with_in_memory_exporter()
    event = {
        "type": "handoff",
        "data": {"from": "triage", "to": "billing", "mechanism": "transfer_agent"},
        "context": {"session_id": "s3"},
    }
    otel_exporter._handle_event(tracer, event)
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["intagrin.handoff.from"] == "triage"
    assert attrs["intagrin.handoff.to"] == "billing"
    assert attrs["intagrin.handoff.mechanism"] == "transfer_agent"


def test_agent_spawned_event_stringifies_nested_values():
    tracer, exporter = _tracer_with_in_memory_exporter()
    event = {
        "type": "agent_spawned",
        "data": {"name": "helper", "config": {"tools": ["a", "b"]}},
        "context": {},
    }
    otel_exporter._handle_event(tracer, event)
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["intagrin.agent.name"] == "helper"
    # Nested dict values aren't valid OTel span attributes on their own — coerced to str rather
    # than dropped, so no event data silently disappears from the exported span.
    assert isinstance(attrs["intagrin.agent.config"], str)


def test_mcp_task_events_produce_spans():
    tracer, exporter = _tracer_with_in_memory_exporter()
    event = {
        "type": "mcp_task_started",
        "data": {"task_id": "t1", "tool": "long_job"},
        "context": {"session_id": "s4"},
    }
    otel_exporter._handle_event(tracer, event)
    attrs = exporter.get_finished_spans()[0].attributes
    assert attrs["intagrin.mcp_task.task_id"] == "t1"
    assert attrs["intagrin.mcp_task.tool"] == "long_job"


def test_error_event_sets_error_status():
    tracer, exporter = _tracer_with_in_memory_exporter()
    event = {"type": "error", "data": {"message": "kaboom"}, "context": {}}
    otel_exporter._handle_event(tracer, event)
    span = exporter.get_finished_spans()[0]
    assert span.status.status_code.name == "ERROR"


def test_step_and_llm_exchange_events_produce_no_span():
    """These carry no attributes not already captured more precisely elsewhere (llm_exchange's
    model is redundant with the cost event's, and its full prompt/response content is
    deliberately not exported as span attributes) — intentionally not spanned."""
    tracer, exporter = _tracer_with_in_memory_exporter()
    otel_exporter._handle_event(tracer, {"type": "step", "data": {}, "context": {}})
    otel_exporter._handle_event(
        tracer, {"type": "llm_exchange", "data": {"model": "x"}, "context": {}}
    )
    assert exporter.get_finished_spans() == ()


def test_ensure_started_is_idempotent():
    # ensure_started schedules a background task via asyncio.create_task, which requires a
    # running event loop — exactly the situation it's actually called from in production
    # (RuntimeEngine.initialize() is itself async), matching the asyncio.run(...)-wrapping
    # convention this test suite already uses elsewhere for engine-loop-dependent code.
    async def _run():
        otel_exporter.ensure_started(["otel"])
        otel_exporter.ensure_started(["otel"])
        otel_exporter.ensure_started(["otel"])
        await asyncio.sleep(0)  # let the scheduled background task actually start once
        return otel_exporter._started

    import asyncio

    assert asyncio.run(_run()) is True


def test_ensure_started_noop_when_telemetry_does_not_include_otel():
    otel_exporter.ensure_started(["langfuse"])
    assert otel_exporter._started is False
    otel_exporter.ensure_started([])
    assert otel_exporter._started is False


def test_ensure_started_raises_ig_rt_009_when_otlp_endpoint_set_but_package_missing(monkeypatch):
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    with pytest.raises(IntaGrinError) as exc_info:
        otel_exporter.ensure_started(["otel"])
    assert exc_info.value.code == "IG-RT-009"
    # A failed activation must not leave the module thinking it already started — otherwise
    # fixing the misconfiguration (installing the package, or unsetting the endpoint) and
    # retrying would silently do nothing on the next ensure_started call.
    assert otel_exporter._started is False


def test_cost_log_call_carries_model_and_token_split_through_to_the_event(monkeypatch):
    """Regression guard for the additive Tracer.log_cost keyword args this feature relies on."""
    captured = {}
    orig_emit = EventStreamer.emit

    def spy(event_type, data):
        if event_type == "cost":
            captured.update(data)
        return orig_emit(event_type, data)

    monkeypatch.setattr(EventStreamer, "emit", staticmethod(spy))
    Tracer.log_cost(30, 0.001, model="openai/gpt-4o", prompt_tokens=20, completion_tokens=10)
    assert captured == {
        "tokens": 30,
        "cost": 0.001,
        "model": "openai/gpt-4o",
        "prompt_tokens": 20,
        "completion_tokens": 10,
    }


def test_cost_log_call_without_new_kwargs_still_works_backward_compatibly(monkeypatch):
    captured = {}
    orig_emit = EventStreamer.emit

    def spy(event_type, data):
        if event_type == "cost":
            captured.update(data)
        return orig_emit(event_type, data)

    monkeypatch.setattr(EventStreamer, "emit", staticmethod(spy))
    Tracer.log_cost(5, 0.0)
    assert captured["tokens"] == 5
    assert captured["model"] is None
