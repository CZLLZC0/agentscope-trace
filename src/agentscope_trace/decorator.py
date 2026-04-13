"""
AgentScope tracing decorators and context managers.

Main entry points:
- @trace(...)        — decorate any function to auto-trace it
- trace_context(...) — context manager for manual span management
- AgentScopeCallbackHandler — LangChain callback handler
"""

from __future__ import annotations

import functools
import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Callable, TypeVar, cast, overload
from typing_extensions import ParamSpec

from .client import AgentScopeClient, get_client
from .models import LLMCall, Span, SpanKind, SpanStatus

logger = logging.getLogger(__name__)


# ── Type helpers ──────────────────────────────────────────────────────────────

P = ParamSpec("P")
F = TypeVar("F", bound=Callable[..., Any])


# ── _TraceContext ─────────────────────────────────────────────────────────────

class _TraceContext:
    """
    Internal context manager that wraps a span's lifecycle.

    Usage:
        with _TraceContext(name="my-span", kind=SpanKind.LLM) as ctx:
            span = ctx.span  # access the Span object
            span.llm_call = LLMCall(...)
    """

    def __init__(
        self,
        name: str,
        kind: SpanKind = SpanKind.AGENT,
        metadata: dict | None = None,
        parent_id: str | None = None,
        client: AgentScopeClient | None = None,
    ):
        self.name = name
        self.kind = kind
        self.metadata = metadata or {}
        self.parent_id = parent_id
        self._client = client or get_client()
        self._span: Span | None = None
        self._exit_exc: BaseException | None = None

    @property
    def span(self) -> Span:
        if self._span is None:
            raise RuntimeError("Span not yet created — use inside `with` block")
        return self._span

    def __enter__(self) -> Span:
        # Build the span
        self._span = Span(
            name=self.name,
            kind=self.kind,
            metadata=self.metadata,
            parent_id=self.parent_id,
        )
        return self._span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self._exit_exc = exc_val

        if self._span is None:
            return

        # Determine status
        if exc_val is not None:
            self._span.finish(status=SpanStatus.ERROR, error=str(exc_val))
        else:
            self._span.finish(status=SpanStatus.OK)

        # Send to backend
        try:
            self._client.create_span(self._span)
        except Exception as e:
            logger.debug("[AgentScope] Failed to send span: %s", e)

        # Re-raise exception (don't suppress it)
        return


# ── Public decorators ─────────────────────────────────────────────────────────

def trace(
    name: str | Callable[..., Any] | None = None,
    kind: SpanKind = SpanKind.AGENT,
    metadata: dict | None = None,
    client: AgentScopeClient | None = None,
) -> Callable[[F], F]:
    """
    Decorator to trace a function's execution as a span.

    Args:
        name: Span name. Defaults to the function's qualified name.
        kind: Span kind (default AGENT).
        metadata: Extra key-value pairs attached to this span.
        client: AgentScopeClient instance (default: global client).

    Usage:
        @trace(name="my-agent")
        def my_agent(input: str) -> str:
            return llm.invoke(input)

        # With metadata:
        @trace(name="rag-pipeline", metadata={"retriever": "pinecone"})
        def rag_search(query: str) -> list[str]:
            ...

    The decorator preserves the original function's signature, return value,
    and exception behavior. Tracing never affects correctness.
    """
    # Support both @trace and @trace() syntax
    if callable(name):
        # Called as @trace without parentheses — name is actually the fn
        fn = name
        return _make_trace_decorator(
            name=fn.__qualname__,
            kind=kind,
            metadata=metadata,
            client=client,
        )(fn)

    # Called as @trace(...) with parentheses
    return _make_trace_decorator(
        name=name or "",
        kind=kind,
        metadata=metadata,
        client=client,
    )


def _make_trace_decorator(
    name: str,
    kind: SpanKind,
    metadata: dict | None,
    client: AgentScopeClient | None,
) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        fn_name = name or fn.__qualname__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Build metadata dict (merge decorator-level + runtime)
            span_metadata = dict(metadata) if metadata else {}
            span_metadata["function"] = fn.__qualname__
            span_metadata["module"] = fn.__module__

            # Allow runtime override via kwarg
            if "trace_metadata" in kwargs:
                span_metadata.update(kwargs.pop("trace_metadata"))

            with _TraceContext(
                name=fn_name,
                kind=kind,
                metadata=span_metadata,
                client=client,
            ):
                return fn(*args, **kwargs)

        # Mark as traced so other tooling can detect it
        wrapper.__dict__["_agentscope_traced"] = True
        wrapper.__dict__["_agentscope_span_kind"] = kind

        return cast(F, wrapper)
    return decorator


# ── Context manager ───────────────────────────────────────────────────────────

@contextmanager
def trace_context(
    name: str,
    kind: SpanKind = SpanKind.AGENT,
    metadata: dict | None = None,
    parent_id: str | None = None,
    client: AgentScopeClient | None = None,
):
    """
    Context manager for manually tracing a block of code.

    Use this when you need more control than the @trace decorator provides,
    or when tracing code that isn't a function.

    Args:
        name: Display name for the span.
        kind: Type of work being traced.
        metadata: Additional metadata.
        parent_id: Explicit parent span ID (auto-detected from context if omitted).
        client: AgentScopeClient to use (default: global client).

    Example:
        with trace_context("synthesize-response", kind=SpanKind.LLM) as span:
            span.llm_call = LLMCall(
                model="gpt-4",
                prompt=prompt,
                completion="...",
                latency_ms=150.0,
            )
    """
    try:
        with _TraceContext(
            name=name,
            kind=kind,
            metadata=metadata,
            parent_id=parent_id,
            client=client,
        ) as span:
            yield span
    except Exception:
        raise


# ── LangChain callback handler ────────────────────────────────────────────────

class AgentScopeCallbackHandler:
    """
    LangChain callback handler — integrates AgentScope with LangChain agents.

    Supports LangChain >= 0.1.0.

    Usage:
        from langchain_openai import ChatOpenAI
        from agentscope_trace import AgentScopeCallbackHandler

        handler = AgentScopeCallbackHandler()
        llm = ChatOpenAI(callbacks=[handler], ...)
    """

    def __init__(
        self,
        client: AgentScopeClient | None = None,
        project: str | None = None,
        tags: list[str] | None = None,
    ):
        self._client = client or get_client()
        self._project = project or os.environ.get("AGENTSCOPE_TRACE_PROJECT", "langchain")
        self._tags = tags or []
        self._span_stack: list[Span] = []
        self._lock = threading.Lock()

    def _start_span(self, name: str, kind: SpanKind, metadata: dict | None = None) -> Span:
        parent = self._span_stack[-1] if self._span_stack else None
        parent_id = parent.span_id if parent else None
        span = Span(
            name=name,
            kind=kind,
            parent_id=parent_id,
            metadata=metadata or {},
        )
        self._span_stack.append(span)
        return span

    def _end_span(self, span: Span, error: str | None = None) -> None:
        span.finish(status=SpanStatus.ERROR if error else SpanStatus.OK, error=error)
        try:
            self._client.create_span(span)
        except Exception as e:
            logger.debug("[AgentScope] Failed to send span: %s", e)
        with self._lock:
            if self._span_stack and self._span_stack[-1].span_id == span.span_id:
                self._span_stack.pop()

    # ── LangChain callback methods ──────────────────────────────────────────

    def on_llm_start(
        self,
        serialized: dict,
        prompts: list[str],
        **kwargs: Any,
    ) -> None:
        model = serialized.get("name", serialized.get("id", ["unknown"])[0])
        metadata = {"model": model, "prompt_count": len(prompts)}
        span = self._start_span(f"llm:{model}", SpanKind.LLM, metadata)
        span.llm_call = LLMCall(
            model=model,
            prompt=prompts[0] if prompts else "",
        )

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        if not self._span_stack:
            return
        span = self._span_stack[-1]
        try:
            # Try to extract completion from LangChain response
            if hasattr(response, "generations") and response.generations:
                gen = response.generations[0][0]
                if span.llm_call:
                    span.llm_call.completion = getattr(gen, "text", str(gen))
                    span.llm_call.latency_ms = getattr(response, "llm_output", {"total_ms": 0.0}).get("total_ms", 0.0) or 0.0
        except Exception:
            pass
        self._end_span(span)

    def on_llm_error(self, error: Exception, **kwargs: Any) -> None:
        if self._span_stack:
            self._end_span(self._span_stack[-1], error=str(error))

    def on_chain_start(
        self,
        serialized: dict,
        inputs: dict,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", "chain")
        metadata = {"input_keys": list(inputs.keys())}
        self._start_span(f"chain:{name}", SpanKind.AGENT, metadata)

    def on_chain_end(self, outputs: dict, **kwargs: Any) -> None:
        if self._span_stack:
            self._end_span(self._span_stack[-1])

    def on_chain_error(self, error: Exception, **kwargs: Any) -> None:
        if self._span_stack:
            self._end_span(self._span_stack[-1], error=str(error))

    def on_tool_start(
        self,
        serialized: dict,
        input_str: str,
        **kwargs: Any,
    ) -> None:
        name = serialized.get("name", serialized.get("description", "tool"))
        metadata = {"input": input_str[:500]}  # truncate for safety
        span = self._start_span(f"tool:{name}", SpanKind.TOOL, metadata)
        span.add_tool_call(name=name, arguments={"input": input_str})

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        if self._span_stack:
            span = self._span_stack[-1]
            if span.tool_calls:
                span.tool_calls[-1].result = output
            self._end_span(span)

    def on_tool_error(self, error: Exception, **kwargs: Any) -> None:
        if self._span_stack:
            span = self._span_stack[-1]
            if span.tool_calls:
                span.tool_calls[-1].error = str(error)
            self._end_span(span, error=str(error))
