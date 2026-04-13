"""Tests for the @trace decorator."""

import pytest
from agentscope_trace.decorator import trace, trace_context
from agentscope_trace.client import AgentScopeClient, get_client
from agentscope_trace.models import SpanKind, SpanStatus


class TestTraceDecorator:
    def test_trace_decorator_calls_function(self):
        """@trace should not change function behavior."""
        @trace(name="test-add")
        def add(a, b):
            return a + b

        result = add(2, 3)
        assert result == 5

    def test_trace_decorator_with_metadata(self):
        """@trace should attach metadata."""
        @trace(name="test-meta", metadata={"version": "1.0"})
        def greet(name):
            return f"Hello, {name}!"

        result = greet("agentscope-trace")
        assert result == "Hello, agentscope-trace!"

    def test_trace_decorator_preserves_return_value(self):
        """Return value should not be affected by tracing."""
        @trace(name="test-return")
        def compute(x):
            return {"result": x * 2}

        result = compute(21)
        assert result == {"result": 42}

    def test_trace_decorator_error_propagation(self):
        """Errors should still be raised even when traced."""
        @trace(name="test-error")
        def always_fails():
            raise ValueError("expected error")

        with pytest.raises(ValueError, match="expected error"):
            always_fails()


class TestTraceContext:
    def test_trace_context_records_span(self):
        """trace_context should record a span with llm_call set."""
        from agentscope_trace.models import LLMCall

        with trace_context("test-context", kind=SpanKind.LLM) as span:
            span.llm_call = LLMCall(
                model="gpt-4",
                prompt="test prompt",
                completion="test completion",
            )

        # Span should have been finished
        assert span.end_time is not None
        assert span.llm_call is not None
        assert span.llm_call.model == "gpt-4"


class TestDisabledMode:
    def test_disabled_client_skips_tracing(self):
        """When AGENTSCOPE_TRACE_DISABLE=1, spans should be skipped."""
        import os
        original = os.environ.get("AGENTSCOPE_TRACE_DISABLE", "0")

        try:
            os.environ["AGENTSCOPE_TRACE_DISABLE"] = "1"
            # Re-import to pick up the env var
            from agentscope_trace import client as client_module
            client_module._client = None
            client = client_module.get_client()
            assert client.disabled is True
        finally:
            os.environ["AGENTSCOPE_TRACE_DISABLE"] = original
