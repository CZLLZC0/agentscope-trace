"""Tests for AgentScope trace models."""

import time
import pytest
from agentscope_trace.models import (
    Span, SpanKind, SpanStatus, LLMCall, ToolCall, TokenUsage, TraceSession
)


class TestSpan:
    def test_span_creation(self):
        span = Span(name="test", kind=SpanKind.AGENT)
        assert span.name == "test"
        assert span.kind == SpanKind.AGENT
        assert span.status == SpanStatus.OK
        assert span.span_id is not None
        assert len(span.span_id) == 32  # UUID hex without dashes
        assert span.trace_id is not None

    def test_span_finish(self):
        span = Span(name="test")
        time.sleep(0.01)
        span.finish()
        assert span.end_time is not None
        assert span.latency_ms is not None
        assert span.latency_ms > 0

    def test_span_finish_with_error(self):
        span = Span(name="test")
        span.finish(status=SpanStatus.ERROR, error="Something went wrong")
        assert span.status == SpanStatus.ERROR
        assert span.error_message == "Something went wrong"

    def test_double_finish_is_noop(self):
        """Finishing twice should not cause issues."""
        span = Span(name="test")
        span.finish()
        end_time = span.end_time
        span.finish()  # should be noop
        assert span.end_time == end_time

    def test_span_with_llm_call(self):
        span = Span(name="llm-call", kind=SpanKind.LLM)
        span.llm_call = LLMCall(
            model="gpt-4",
            prompt="Hello",
            completion="Hi there!",
            latency_ms=150.0,
        )
        data = span.to_dict()
        assert data["llm_call"]["model"] == "gpt-4"
        assert data["llm_call"]["prompt"] == "Hello"
        assert data["llm_call"]["completion"] == "Hi there!"

    def test_span_with_tool_calls(self):
        span = Span(name="tool-span", kind=SpanKind.TOOL)
        span.add_tool_call(
            name="search",
            arguments={"query": "test"},
            result="search result",
            latency_ms=50.0,
        )
        data = span.to_dict()
        assert len(data["tool_calls"]) == 1
        assert data["tool_calls"][0]["name"] == "search"

    def test_span_to_dict(self):
        span = Span(
            name="test",
            kind=SpanKind.AGENT,
            metadata={"key": "value", "count": 42},
        )
        data = span.to_dict()
        assert "span_id" in data
        assert "trace_id" in data
        assert data["name"] == "test"
        assert data["kind"] == "agent"
        assert data["metadata"]["key"] == "value"

    def test_span_from_dict(self):
        original = Span(name="test", kind=SpanKind.LLM)
        original.finish()
        restored = Span.from_dict(original.to_dict())
        assert restored.name == original.name
        assert restored.span_id == original.span_id
        assert restored.trace_id == original.trace_id

    def test_span_with_parent_id(self):
        parent = Span(name="parent")
        child = Span(name="child", parent_id=parent.span_id)
        assert child.parent_id == parent.span_id
        assert child.trace_id == parent.trace_id  # Same trace


class TestLLMCall:
    def test_llm_call_creation(self):
        llm = LLMCall(model="gpt-4", prompt="test", completion="result")
        assert llm.model == "gpt-4"
        assert llm.prompt == "test"
        assert llm.completion == "result"

    def test_llm_call_with_messages(self):
        llm = LLMCall(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"},
            ],
        )
        assert len(llm.messages) == 2
        assert llm.messages[0]["role"] == "system"

    def test_llm_call_with_usage(self):
        llm = LLMCall(
            model="gpt-4",
            prompt="Hello",
            completion="Hi!",
            usage=TokenUsage(input_tokens=5, output_tokens=6),
        )
        data = llm.to_dict()
        assert data["usage"]["input_tokens"] == 5
        assert data["usage"]["output_tokens"] == 6

    def test_llm_call_to_dict(self):
        llm = LLMCall(model="gpt-4", prompt="Hi", completion="Hello!")
        data = llm.to_dict()
        assert "model" in data
        assert "prompt" in data


class TestToolCall:
    def test_tool_call_creation(self):
        tool = ToolCall(
            name="search",
            arguments={"query": "AI agents"},
            result="Found 10 results",
        )
        assert tool.name == "search"
        assert tool.arguments["query"] == "AI agents"

    def test_tool_call_truncation(self):
        """Long arguments/results should be truncated."""
        tool = ToolCall(
            name="search",
            arguments={"large_field": "x" * 20_000},
            result="y" * 20_000,
        )
        assert len(tool.arguments["large_field"]) <= 10_000  # truncated
        assert len(tool.result) <= 10_000  # truncated

    def test_tool_call_with_error(self):
        tool = ToolCall(
            name="search",
            arguments={"query": "test"},
            result="",
            error="Connection timeout",
        )
        data = tool.to_dict()
        assert data["error"] == "Connection timeout"


class TestTokenUsage:
    def test_token_usage_auto_total(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_token_usage_cost(self):
        usage = TokenUsage(
            input_tokens=1000,
            output_tokens=500,
            input_cost_usd=0.001,
            output_cost_usd=0.002,
        )
        assert usage.total_cost_usd == 0.003


class TestTraceSession:
    def test_session_creation(self):
        session = TraceSession(name="my-session")
        assert session.session_id is not None
        assert len(session.session_id) == 32
        assert session.name == "my-session"
        assert session.created_at is not None

    def test_session_to_dict(self):
        session = TraceSession(name="test", metadata={"tag": "unit-test"})
        data = session.to_dict()
        assert "session_id" in data
        assert data["name"] == "test"
        assert data["metadata"]["tag"] == "unit-test"

    def test_session_from_dict(self):
        original = TraceSession(name="test")
        restored = TraceSession.from_dict(original.to_dict())
        assert restored.session_id == original.session_id
        assert restored.name == original.name
