"""
AgentScope data models.

Span model — the core unit of observability.
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# ── Enums ─────────────────────────────────────────────────────────────────────

class SpanKind(str, Enum):
    """What kind of span is this?"""
    AGENT = "agent"           # Top-level agent orchestration
    LLM = "llm"               # LLM model call
    TOOL = "tool"             # Tool / function call
    RETRIEVER = "retriever"   # RAG retrieval step
    EMBEDDER = "embedder"     # Embedding generation
    CUSTOM = "custom"          # User-defined span type

    @classmethod
    def all_values(cls) -> list[str]:
        return [e.value for e in cls]


class SpanStatus(str, Enum):
    """Did the span succeed or fail?"""
    OK = "ok"
    ERROR = "error"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# ── Sub-models ────────────────────────────────────────────────────────────────

@dataclass
class TokenUsage:
    """Token consumption breakdown."""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    # Vendor-specific usage metadata
    prompt_tokens: int = 0
    completion_tokens: int = 0
    # Cost tracking (optional)
    input_cost_usd: float | None = None
    output_cost_usd: float | None = None
    total_cost_usd: float | None = None

    def __post_init__(self):
        # Auto-fill total if not set
        if self.total_tokens == 0 and (self.input_tokens or self.output_tokens):
            self.total_tokens = self.input_tokens + self.output_tokens
        if self.prompt_tokens and not self.input_tokens:
            self.input_tokens = self.prompt_tokens
        if self.completion_tokens and not self.output_tokens:
            self.output_tokens = self.completion_tokens
        # Auto-sum costs
        if self.total_cost_usd is None and (self.input_cost_usd is not None or self.output_cost_usd is not None):
            self.total_cost_usd = (self.input_cost_usd or 0.0) + (self.output_cost_usd or 0.0)

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class LLMCall:
    """
    Details of a single LLM invocation captured inside a span.

    Supports both:
    - string prompt / completion (legacy)
    - message arrays (OpenAI chat format)
    """
    model: str = ""
    # String format
    prompt: str = ""
    completion: str = ""
    # Message array format (preferred)
    messages: list[dict[str, str]] = field(default_factory=list)
    # Usage
    usage: TokenUsage | None = None
    # Latency
    latency_ms: float = 0.0
    # Optional: raw model response for debugging
    raw_response: dict | None = None
    # Optional: stop reason from model
    stop_reason: str | None = None

    def __post_init__(self):
        pass  # No auto-computed fields needed; set usage explicitly via TokenUsage

    def to_dict(self) -> dict:
        result = {
            "model": self.model,
            "prompt": self.prompt,
            "completion": self.completion,
            "messages": self.messages,
            "usage": self.usage.to_dict() if self.usage else None,
            "latency_ms": self.latency_ms,
            "stop_reason": self.stop_reason,
        }
        # Only include raw_response if it exists (avoids null noise)
        if self.raw_response:
            result["raw_response"] = self.raw_response
        return result

    @classmethod
    def from_dict(cls, data: dict) -> LLMCall:
        usage = None
        if data.get("usage"):
            usage = TokenUsage(**data["usage"])
        return cls(
            model=data.get("model", ""),
            prompt=data.get("prompt", ""),
            completion=data.get("completion", ""),
            messages=data.get("messages", []),
            usage=usage,
            latency_ms=data.get("latency_ms", 0.0),
            raw_response=data.get("raw_response"),
            stop_reason=data.get("stop_reason"),
        )


@dataclass
class ToolCall:
    """
    Details of a single tool/function invocation inside a span.
    """
    name: str = ""
    arguments: dict[str, Any] = field(default_factory=dict)   # sanitized args
    result: Any = ""                                           # truncated result
    error: str | None = None
    latency_ms: float = 0.0
    # Tool-specific metadata (e.g., API endpoint, HTTP status)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Truncate arguments and result to avoid huge payloads
        max_len = 10_000
        result = self.result
        if isinstance(result, str) and len(result) > max_len:
            self.result = result[:max_len]
        elif not isinstance(result, (str, dict, list, int, float, bool, type(None))):
            self.result = str(result)[:max_len]

        # Sanitize arguments (remove very large values)
        if isinstance(self.arguments, dict):
            for k, v in list(self.arguments.items()):
                if isinstance(v, str) and len(v) > max_len:
                    self.arguments[k] = v[:max_len]
                elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
                    self.arguments[k] = str(v)[:max_len]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result if isinstance(self.result, str) else str(self.result),
            "error": self.error,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ToolCall:
        return cls(
            name=data.get("name", ""),
            arguments=data.get("arguments", {}),
            result=data.get("result", ""),
            error=data.get("error"),
            latency_ms=data.get("latency_ms", 0.0),
            metadata=data.get("metadata", {}),
        )


# ── Core Span model ───────────────────────────────────────────────────────────

@dataclass
class Span:
    """
    A single unit of work in an agent execution trace.

    Spans form a tree: a root agent span can have child LLM spans,
    which can have tool call spans, etc.

    Example:
        with trace_context("my-agent") as span:
            span.llm_call = LLMCall(model="gpt-4", prompt="Hello")
            result = llm.invoke("Hello")
            span.llm_call.completion = result.content
    """
    name: str = ""
    kind: SpanKind = SpanKind.AGENT
    # Auto-generated IDs
    span_id: str = field(default_factory=lambda: _gen_id())
    trace_id: str | None = None
    # Timestamps
    start_time: str = field(default_factory=lambda: _now_iso())
    end_time: str | None = None
    # Hierarchy
    parent_id: str | None = None
    # Status
    status: SpanStatus = SpanStatus.OK
    error_message: str | None = None
    # Annotations
    metadata: dict[str, Any] = field(default_factory=dict)
    # Call details (set by user or callback)
    llm_call: LLMCall | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # Computed
    latency_ms: float | None = None

    def __post_init__(self):
        # Validate kind
        if isinstance(self.kind, str):
            self.kind = SpanKind(self.kind)
        # Generate trace_id if not set
        if self.trace_id is None:
            if self.parent_id:
                # Inherit from parent if parent_id is set
                parent_trace = _get_parent_trace_id(self.parent_id)
                self.trace_id = parent_trace if parent_trace else _gen_id()
            else:
                self.trace_id = _gen_id()
        # Register this span so children can inherit its trace_id
        _register_span(self.span_id, self.trace_id)

    def finish(
        self,
        status: SpanStatus = SpanStatus.OK,
        error: str | None = None,
    ) -> None:
        """
        Mark this span as complete.

        Args:
            status: Final status (default OK)
            error: Error message if status == ERROR
        """
        if self.end_time is not None:
            # Already finished — don't double-finish
            return

        self.end_time = _now_iso()
        self.status = status
        if error:
            self.error_message = error
            self.status = SpanStatus.ERROR

        # Compute latency
        if self.start_time and self.end_time:
            start = _parse_iso(self.start_time)
            end = _parse_iso(self.end_time)
            if start and end:
                self.latency_ms = round((end - start).total_seconds() * 1000, 2)

    def add_tool_call(
        self,
        name: str,
        arguments: dict[str, Any],
        result: Any = "",
        error: str | None = None,
        latency_ms: float = 0.0,
    ) -> ToolCall:
        """Convenience method to add a tool call to this span."""
        tool = ToolCall(
            name=name,
            arguments=arguments,
            result=result,
            error=error,
            latency_ms=latency_ms,
        )
        self.tool_calls.append(tool)
        return tool

    def to_dict(self) -> dict:
        """Serialize to a plain dict (for JSON transmission and storage)."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "kind": self.kind.value if isinstance(self.kind, SpanKind) else self.kind,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "latency_ms": self.latency_ms,
            "status": self.status.value if isinstance(self.status, SpanStatus) else self.status,
            "error_message": self.error_message,
            "metadata": self.metadata,
            "llm_call": self.llm_call.to_dict() if self.llm_call else None,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
        }

    @classmethod
    def from_dict(cls, data: dict) -> Span:
        """Deserialize from a plain dict."""
        llm_call = None
        if data.get("llm_call"):
            llm_call = LLMCall.from_dict(data["llm_call"])

        tool_calls = [ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])]

        return cls(
            span_id=data.get("span_id", _gen_id()),
            trace_id=data.get("trace_id", _gen_id()),
            parent_id=data.get("parent_id"),
            name=data.get("name", ""),
            kind=SpanKind(data.get("kind", "agent")),
            start_time=data.get("start_time", _now_iso()),
            end_time=data.get("end_time"),
            latency_ms=data.get("latency_ms"),
            status=SpanStatus(data.get("status", "ok")),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
            llm_call=llm_call,
            tool_calls=tool_calls,
        )


# ── Trace Session ─────────────────────────────────────────────────────────────

@dataclass
class TraceSession:
    """
    A logical grouping of related spans (e.g., one user conversation or one batch job).

    A session can contain multiple traces (root spans).
    """
    name: str = "default"
    session_id: str = field(default_factory=lambda: _gen_id())
    created_at: str = field(default_factory=lambda: _now_iso())
    metadata: dict[str, Any] = field(default_factory=dict)
    # Runtime tags
    tags: list[str] = field(default_factory=list)
    # Statistics (populated after session ends)
    total_spans: int = 0
    total_llm_calls: int = 0
    total_token_usage: TokenUsage | None = None
    duration_ms: float | None = None

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "name": self.name,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "tags": self.tags,
            "total_spans": self.total_spans,
            "total_llm_calls": self.total_llm_calls,
            "total_token_usage": self.total_token_usage.to_dict() if self.total_token_usage else None,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict) -> TraceSession:
        usage = None
        if data.get("total_token_usage"):
            usage = TokenUsage(**data["total_token_usage"])
        return cls(
            session_id=data.get("session_id", _gen_id()),
            name=data.get("name", "default"),
            created_at=data.get("created_at", _now_iso()),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            total_spans=data.get("total_spans", 0),
            total_llm_calls=data.get("total_llm_calls", 0),
            total_token_usage=usage,
            duration_ms=data.get("duration_ms"),
        )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _gen_id() -> str:
    """Generate a unique ID (UUID v4, without dashes)."""
    return uuid.uuid4().hex


# ── Span registry (for trace_id inheritance) ─────────────────────────────────

_span_registry: dict[str, str] = {}
_span_registry_lock = threading.Lock()


def _register_span(span_id: str, trace_id: str) -> None:
    """Register a span so children can inherit its trace_id."""
    with _span_registry_lock:
        _span_registry[span_id] = trace_id


def _get_parent_trace_id(parent_id: str) -> str | None:
    """Look up the trace_id of a parent span."""
    with _span_registry_lock:
        return _span_registry.get(parent_id)


_UTC = timezone.utc


def _now_iso() -> str:
    """Return current UTC time in ISO-8601 format."""
    return datetime.now(_UTC).isoformat().replace("+00:00", "Z")


def _parse_iso(ts: str) -> datetime | None:
    """Parse ISO-8601 timestamp (handles Z suffix)."""
    if not ts:
        return None
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(ts)
    except ValueError:
        return None
