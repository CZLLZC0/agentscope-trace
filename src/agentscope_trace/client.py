"""
AgentScope HTTP client — sends spans to the backend.

Features:
- Lazy initialization (no backend required at import time)
- Automatic batching of spans
- Retry with exponential backoff
- Thread-safe (uses a queue for background flushing)
- Graceful degradation (spans are dropped on persistent failure, never crash user code)
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any
from urllib.parse import urljoin

import requests
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import RequestException, Timeout

from .models import Span, SpanStatus, TraceSession

logger = logging.getLogger(__name__)


# ── Configuration ─────────────────────────────────────────────────────────────

DEFAULT_URL = os.environ.get("AGENTSCOPE_TRACE_URL", "http://localhost:8000")
DEFAULT_TIMEOUT = 10.0   # seconds
DEFAULT_PROJECT = os.environ.get("AGENTSCOPE_TRACE_PROJECT", "default")
DEFAULT_BATCH_SIZE = 10
DEFAULT_FLUSH_INTERVAL = 2.0  # seconds


# ── Exceptions ────────────────────────────────────────────────────────────────

class AgentScopeClientError(Exception):
    """Base exception for AgentScope client errors."""
    pass


class AgentScopeConfigError(AgentScopeClientError):
    """Misconfiguration (bad URL, missing API key, etc.)."""
    pass


class AgentScopeConnectionError(AgentScopeClientError):
    """Could not connect to the backend."""
    pass


# ── Span Serializer ───────────────────────────────────────────────────────────

def _serialize(obj: Any) -> Any:
    """Convert a dataclass or model to a plain dict suitable for JSON."""
    if obj is None:
        return None
    if is_dataclass(obj):
        return {k: _serialize(v) for k, v in asdict(obj).items()}  # type: ignore[arg-type]
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialize(i) for i in obj]
    # Primitives pass through
    return obj


# ── HTTP Client ───────────────────────────────────────────────────────────────

class AgentScopeClient:
    """
    Sends span data to the AgentScope backend.

    Thread-safe. Non-blocking: spans are queued and flushed in the background
    so they never add latency to your agent's execution.

    Usage:
        client = AgentScopeClient()
        client.create_span(span)
        client.flush()   # optional, periodic flush happens automatically
    """

    _instance: AgentScopeClient | None = None
    _lock = threading.Lock()

    def __init__(
        self,
        url: str = DEFAULT_URL,
        timeout: float = DEFAULT_TIMEOUT,
        project: str = DEFAULT_PROJECT,
        api_key: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        flush_interval: float = DEFAULT_FLUSH_INTERVAL,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
        disabled: bool = False,
    ):
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.project = project
        self.api_key = api_key
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.disabled = disabled

        # Internal state
        self._session: requests.Session | None = None
        self._span_queue: queue.Queue[Span] = queue.Queue()
        self._session_queue: queue.Queue[TraceSession] = queue.Queue()
        self._flush_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._started = False

        # Validate URL format
        if not self.disabled:
            self._validate_url()

    def _validate_url(self) -> None:
        """Check that the URL is a valid http/https URL."""
        if not self.url.startswith(("http://", "https://")):
            raise AgentScopeConfigError(
                f"Invalid URL: {self.url!r} — must start with http:// or https://"
            )

    @property
    def session(self) -> requests.Session:
        """Lazily create a requests Session (connection pooling)."""
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update({
                "Content-Type": "application/json",
                "User-Agent": "AgentScope-Python-SDK/0.1.0",
            })
            if self.api_key:
                self._session.headers["Authorization"] = f"Bearer {self.api_key}"
        return self._session

    # ── Singleton factory ─────────────────────────────────────────────────────

    @classmethod
    def get_instance(
        cls,
        url: str = DEFAULT_URL,
        timeout: float = DEFAULT_TIMEOUT,
        project: str = DEFAULT_PROJECT,
        api_key: str | None = None,
        **kwargs,
    ) -> AgentScopeClient:
        """Get or create the global singleton client (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(
                        url=url,
                        timeout=timeout,
                        project=project,
                        api_key=api_key,
                        **kwargs,
                    )
        return cls._instance

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def start(self) -> None:
        """Start the background flush thread. Called automatically on first span."""
        if self._started or self.disabled:
            return
        self._started = True
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            name="AgentScope-Flush",
            daemon=True,
        )
        self._flush_thread.start()

    def stop(self, timeout: float = 5.0) -> None:
        """
        Stop the flush thread and send any remaining spans.

        Call this at application shutdown to avoid losing data.
        """
        if not self._started:
            return
        self._stop_event.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=timeout)
        # Final flush
        try:
            self._do_flush()
        except Exception:
            pass

    # ── Public API ───────────────────────────────────────────────────────────

    def create_span(self, span: Span) -> None:
        """
        Record a completed span.

        This is non-blocking — the span is queued and flushed asynchronously.
        """
        if self.disabled:
            return
        if span.status == SpanStatus.OK and span.end_time is None:
            # Auto-finish if not already finished
            span.finish()

        self.span_queue.put_nowait(span)
        self.start()

    def create_session(self, session: TraceSession) -> None:
        """Record a trace session."""
        if self.disabled:
            return
        self.session_queue.put_nowait(session)
        self.start()

    def flush(self, timeout: float = 5.0) -> None:
        """
        Force-flush all queued spans to the backend.

        Waits up to `timeout` seconds. Raises on persistent failure.
        """
        if self.disabled:
            return
        try:
            self._do_flush()
        except Exception as e:
            logger.warning("[AgentScope] flush failed: %s", e)

    # ── Internal flush logic ─────────────────────────────────────────────────

    def _flush_loop(self) -> None:
        """Background thread: periodically flush the queue."""
        while not self._stop_event.is_set():
            self._do_flush()
            # Wait for flush interval or stop signal
            self._stop_event.wait(timeout=self.flush_interval)

        # One final flush before exit
        self._do_flush()

    def _do_flush(self) -> None:
        """Drain the queue and send spans to the backend."""
        # Collect spans up to batch_size
        spans_data: list[dict[str, Any]] = []
        while len(spans_data) < self.batch_size:
            try:
                span = self.span_queue.get_nowait()
                serialized = _serialize(span.to_dict())
                if serialized is not None:
                    spans_data.append(serialized)
            except queue.Empty:
                break

        if not spans_data:
            return

        # Send to backend with retries
        try:
            self._post_spans(spans_data)
        except Exception as exc:
            # Put spans back in queue for retry on next flush
            logger.warning(
                "[AgentScope] Failed to flush %d spans: %s — will retry",
                len(spans_data), exc,
            )
            for data in spans_data:
                # Best-effort re-queue (drop if queue is full to avoid memory leak)
                try:
                    self.span_queue.put_nowait(Span.from_dict(data))
                except queue.Full:
                    logger.error("[AgentScope] Span queue full — dropping spans")
                    break

        # Also flush sessions
        sessions_data: list[dict[str, Any]] = []
        while not self.session_queue.empty():
            try:
                session = self.session_queue.get_nowait()
                sessions_data.append(_serialize(session.to_dict()))
            except queue.Empty:
                break

        if sessions_data:
            try:
                self._post_sessions(sessions_data)
            except Exception as exc:
                logger.warning("[AgentScope] Failed to flush sessions: %s", exc)

    def _post_spans(self, spans_data: list[dict]) -> None:
        """POST spans to the backend with retries."""
        payload = {
            "project": self.project,
            "spans": spans_data,
        }
        url = urljoin(self.url + "/", "api/v1/spans")
        self._post_with_retry(url, payload)

    def _post_sessions(self, sessions_data: list[dict]) -> None:
        """POST sessions to the backend with retries."""
        payload = {
            "project": self.project,
            "sessions": sessions_data,
        }
        url = urljoin(self.url + "/", "api/v1/sessions")
        self._post_with_retry(url, payload)

    def _post_with_retry(self, url: str, payload: dict, retries: int | None = None) -> None:
        """
        POST with exponential backoff retry.

        Retries on: connection errors, 5xx server errors, 429 rate limiting.
        Does NOT retry on: 4xx client errors (except 429).
        """
        if retries is None:
            retries = self.max_retries

        last_error: Exception | None = None

        for attempt in range(retries + 1):
            try:
                response = self.session.post(
                    url,
                    json=payload,
                    timeout=self.timeout,
                )

                if response.status_code == 200:
                    return  # Success

                if response.status_code == 429:
                    # Rate limited — wait longer
                    retry_after = float(response.headers.get("Retry-After", 2))
                    logger.info("[AgentScope] Rate limited, waiting %.1fs", retry_after)
                    time.sleep(retry_after)
                    continue

                if 400 <= response.status_code < 500:
                    # Client error — don't retry, raise immediately
                    try:
                        detail = response.json().get("detail", response.text)
                    except Exception:
                        detail = response.text
                    raise AgentScopeClientError(
                        f"Server rejected request ({response.status_code}): {detail}"
                    )

                # 5xx — retry
                logger.warning(
                    "[AgentScope] Server error %d, attempt %d/%d",
                    response.status_code, attempt + 1, retries + 1,
                )

            except (ReqConnectionError, Timeout, RequestException) as e:
                last_error = e
                logger.warning(
                    "[AgentScope] Connection error (attempt %d/%d): %s",
                    attempt + 1, retries + 1, e,
                )

            if attempt < retries:
                delay = self.retry_base_delay * (2 ** attempt)
                time.sleep(delay)

        # All retries exhausted
        raise AgentScopeConnectionError(
            f"Failed after {retries + 1} attempts: {last_error}"
        )

    def health_check(self) -> bool:
        """Ping the backend to verify connectivity."""
        if self.disabled:
            return False
        try:
            url = urljoin(self.url + "/", "health")
            response = self.session.get(url, timeout=3.0)
            return bool(response.status_code == 200)
        except Exception:
            return False

    # ── Queue properties ─────────────────────────────────────────────────────

    @property
    def span_queue(self) -> queue.Queue[Span]:
        return self._span_queue

    @property
    def session_queue(self) -> queue.Queue[TraceSession]:
        return self._session_queue

    def __repr__(self) -> str:
        return (
            f"AgentScopeClient(url={self.url!r}, "
            f"project={self.project!r}, "
            f"disabled={self.disabled})"
        )


# ── Module-level helpers ──────────────────────────────────────────────────────

_client: AgentScopeClient | None = None


def get_client() -> AgentScopeClient:
    """Get the current global client (lazy creation)."""
    global _client
    if _client is None:
        disabled = os.environ.get("AGENTSCOPE_TRACE_DISABLE", "0") == "1"
        url = os.environ.get("AGENTSCOPE_TRACE_URL", DEFAULT_URL)
        project = os.environ.get("AGENTSCOPE_TRACE_PROJECT", DEFAULT_PROJECT)
        api_key = os.environ.get("AGENTSCOPE_TRACE_API_KEY")
        _client = AgentScopeClient(
            url=url,
            project=project,
            api_key=api_key,
            disabled=disabled,
        )
    return _client


def set_client(client: AgentScopeClient) -> None:
    """Replace the global client."""
    global _client
    _client = client
