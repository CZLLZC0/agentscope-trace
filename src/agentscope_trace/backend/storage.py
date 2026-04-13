"""
Thread-safe in-memory storage for AgentScope backend.

Uses a dict of dicts with a reentrant lock.
Supports concurrent reads/writes from multiple FastAPI workers.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class TokenStats:
    """Running token usage statistics for a project."""
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    llm_call_count: int = 0
    span_count: int = 0
    error_count: int = 0

    def to_dict(self) -> dict:
        return {
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "llm_call_count": self.llm_call_count,
            "span_count": self.span_count,
            "error_count": self.error_count,
        }


class Storage:
    """
    Thread-safe in-memory storage.

    Data model:
        spans:       trace_id -> list[Span]
        sessions:    session_id -> TraceSession
        traces:      trace_id -> {session_id, root_span_id, stats}
        stats:       project -> TokenStats
    """

    def __init__(self, max_spans: int = 100_000, max_sessions: int = 10_000):
        self._lock = threading.RLock()

        # trace_id -> list of span dicts
        self._spans: dict[str, list[dict]] = {}
        # session_id -> session dict
        self._sessions: dict[str, dict] = {}
        # trace_id -> trace metadata
        self._traces: dict[str, dict] = {}
        # project_name -> TokenStats
        self._stats: dict[str, TokenStats] = {}

        # Limits to prevent memory exhaustion
        self._max_spans = max_spans
        self._max_sessions = max_sessions
        self._total_spans = 0
        self._total_sessions = 0

        # Rate limiting state: project -> (count, window_start)
        self._rate_limit: dict[str, tuple[int, float]] = {}
        self._rate_limit_lock = threading.Lock()

    # ── Core CRUD ────────────────────────────────────────────────────────────

    def save_span(self, span_data: dict, project: str = "default") -> dict:
        """Save a span, update trace stats, and return it."""
        with self._lock:
            trace_id = span_data["trace_id"]
            session_id = span_data.get("session_id", "default")

            # Create trace entry if new
            if trace_id not in self._traces:
                self._traces[trace_id] = {
                    "trace_id": trace_id,
                    "session_id": session_id,
                    "root_span_id": span_data.get("parent_id") or trace_id,
                    "project": project,
                    "span_count": 0,
                    "llm_call_count": 0,
                    "created_at": span_data.get("start_time"),
                    "last_updated": span_data.get("start_time"),
                }
                self._total_spans = 0  # reset per-trace count

            # Append span
            if trace_id not in self._spans:
                self._spans[trace_id] = []
            self._spans[trace_id].append(span_data)

            # Update trace metadata
            self._traces[trace_id]["span_count"] += 1
            self._traces[trace_id]["last_updated"] = span_data.get("end_time") or span_data.get("start_time")

            # Update project stats
            stats = self._stats.setdefault(project, TokenStats())
            stats.span_count += 1

            status = span_data.get("status", "ok")
            if status == "error":
                stats.error_count += 1

            # Extract LLM call stats
            llm_call = span_data.get("llm_call")
            if llm_call and isinstance(llm_call, dict):
                stats.llm_call_count += 1
                usage = llm_call.get("usage") or {}
                if usage:
                    stats.total_input_tokens += usage.get("input_tokens", 0)
                    stats.total_output_tokens += usage.get("output_tokens", 0)
                    cost = usage.get("total_cost_usd") or usage.get("cost_usd")
                    if cost:
                        stats.total_cost_usd += cost

            # Enforce span limit (drop oldest traces)
            self._maybe_evict()

            return span_data

    def save_spans(self, spans_data: list[dict], project: str = "default") -> list[dict]:
        """Save multiple spans in one batch (atomic)."""
        with self._lock:
            results = []
            for span_data in spans_data:
                results.append(self.save_span(span_data, project))
            return results

    def save_session(self, session_data: dict, project: str = "default") -> dict:
        """Save a trace session."""
        with self._lock:
            session_id = session_data["session_id"]
            self._sessions[session_id] = session_data

            # Enforce session limit
            while len(self._sessions) > self._max_sessions:
                oldest = min(self._sessions, key=lambda s: self._sessions[s].get("created_at", ""))
                del self._sessions[oldest]

            return session_data

    def save_sessions(self, sessions_data: list[dict], project: str = "default") -> list[dict]:
        """Save multiple sessions."""
        with self._lock:
            return [self.save_session(s, project) for s in sessions_data]

    # ── Queries ────────────────────────────────────────────────────────────────

    def get_trace(self, trace_id: str) -> dict | None:
        """Get a trace by ID (includes all spans)."""
        with self._lock:
            if trace_id not in self._traces:
                return None
            return {
                "trace": self._traces.get(trace_id, {}),
                "spans": self._spans.get(trace_id, []),
            }

    def get_traces(
        self,
        project: str = "default",
        limit: int = 50,
        offset: int = 0,
        search: str | None = None,
    ) -> list[dict]:
        """List traces for a project, newest first."""
        with self._lock:
            traces = [
                t for t in self._traces.values()
                if t.get("project") == project
            ]
            if search:
                q = search.lower()
                traces = [t for t in traces if q in t.get("name", "").lower()]
            traces.sort(key=lambda t: t.get("last_updated", ""), reverse=True)
            return traces[offset : offset + limit]

    def get_session(self, session_id: str) -> dict | None:
        """Get a session by ID."""
        with self._lock:
            return self._sessions.get(session_id)

    def get_stats(self, project: str = "default") -> dict:
        """Get usage statistics for a project."""
        with self._lock:
            stats = self._stats.get(project, TokenStats())
            return stats.to_dict()

    def get_all_projects(self) -> list[str]:
        """List all projects that have data."""
        with self._lock:
            return list(self._stats.keys())

    # ── Rate limiting ────────────────────────────────────────────────────────

    def check_rate_limit(
        self,
        project: str,
        max_requests: int = 1000,
        window_seconds: float = 60.0,
    ) -> tuple[bool, dict]:
        """
        Simple in-memory rate limiter.

        Returns (allowed, info_dict).
        """
        import time
        now = time.monotonic()

        with self._rate_limit_lock:
            key = project
            count, window_start = self._rate_limit.get(key, (0, now))

            # Reset window if expired
            if now - window_start >= window_seconds:
                count = 0
                window_start = now

            allowed = count < max_requests
            if allowed:
                count += 1

            self._rate_limit[key] = (count, window_start)

            remaining = max(0, max_requests - count)
            reset_in = max(0, window_seconds - (now - window_start))

            return allowed, {
                "limit": max_requests,
                "remaining": remaining,
                "reset_in_seconds": round(reset_in, 1),
                "window_seconds": window_seconds,
            }

    # ── Memory management ────────────────────────────────────────────────────

    def _maybe_evict(self) -> None:
        """Drop oldest traces if we exceed the span limit."""
        # Count total spans across all traces
        total = sum(len(spans) for spans in self._spans.values())
        if total <= self._max_spans:
            return

        # Sort traces by last_updated, oldest first
        sorted_traces = sorted(
            self._traces.items(),
            key=lambda item: item[1].get("last_updated", ""),
        )

        # Remove oldest traces until under limit
        for trace_id, trace_meta in sorted_traces:
            if total <= self._max_spans * 0.8:  # leave 20% headroom
                break
            # Remove spans for this trace
            span_count = len(self._spans.pop(trace_id, []))
            total -= span_count
            del self._traces[trace_id]

    def clear(self) -> None:
        """Clear all data (for testing)."""
        with self._lock:
            self._spans.clear()
            self._sessions.clear()
            self._traces.clear()
            self._stats.clear()
            self._total_spans = 0
            self._total_sessions = 0


# ── Global singleton ───────────────────────────────────────────────────────────

_storage: Storage | None = None
_storage_lock = threading.Lock()


def get_storage(max_spans: int = 100_000, max_sessions: int = 10_000) -> Storage:
    global _storage
    if _storage is None:
        with _storage_lock:
            if _storage is None:
                _storage = Storage(max_spans=max_spans, max_sessions=max_sessions)
    return _storage
