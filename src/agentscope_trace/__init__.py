"""
agentscope-trace — Open-source AI Agent observability platform.

Quick start:
    from agentscope_trace import trace, get_client

    # Auto-trace any function
    @trace(name="my-agent")
    def my_agent(input: str) -> str:
        ...

    # Or use trace_context for more control
    from agentscope_trace import trace_context, SpanKind, LLMCall

    with trace_context("synthesis", kind=SpanKind.LLM) as span:
        span.llm_call = LLMCall(model="gpt-4", prompt="...", completion="...")
        result = llm.invoke("...")
        span.llm_call.completion = result

    # View traces at http://localhost:8000 (or your backend URL)
"""

from __future__ import annotations

__version__ = "0.1.0"

# Core SDK
from .client import (
    AgentScopeClient,
    AgentScopeClientError,
    AgentScopeConfigError,
    AgentScopeConnectionError,
    get_client,
    set_client,
)
from .decorator import AgentScopeCallbackHandler, trace, trace_context
from .models import (
    LLMCall,
    Span,
    SpanKind,
    SpanStatus,
    TokenUsage,
    ToolCall,
    TraceSession,
)

# Public API
__all__ = [
    # Version
    "__version__",
    # Core SDK
    "trace",
    "trace_context",
    "AgentScopeClient",
    "get_client",
    "set_client",
    # Exceptions
    "AgentScopeClientError",
    "AgentScopeConfigError",
    "AgentScopeConnectionError",
    # Models
    "Span",
    "SpanKind",
    "SpanStatus",
    "LLMCall",
    "ToolCall",
    "TokenUsage",
    "TraceSession",
    # Integrations
    "AgentScopeCallbackHandler",
]
