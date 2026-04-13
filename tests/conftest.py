"""Pytest configuration and fixtures for AgentScope."""

import pytest
from agentscope_trace.client import AgentScopeClient
from agentscope_trace.models import Span, SpanKind, TraceSession


@pytest.fixture
def client():
    """Create a test client pointing to localhost:8000."""
    return AgentScopeClient(url="http://localhost:8000", timeout=5.0)


@pytest.fixture
def sample_span():
    """Create a sample span for testing."""
    return Span(
        name="test-span",
        kind=SpanKind.LLM,
        metadata={"test": True},
    )


@pytest.fixture
def sample_session():
    """Create a sample session for testing."""
    return TraceSession(
        name="test-session",
        metadata={"env": "test"},
    )
