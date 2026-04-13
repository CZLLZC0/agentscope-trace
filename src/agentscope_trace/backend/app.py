"""
AgentScope Backend — FastAPI application.

Run with:
    uvicorn agentscope.backend.app:app --reload --port 8000
"""

from __future__ import annotations

import logging
import os
import time

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from .storage import get_storage

logger = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────

HOST = os.environ.get("AGENTSCOPE_HOST", "0.0.0.0")
PORT = int(os.environ.get("AGENTSCOPE_PORT", "8000"))
MAX_SPANS = int(os.environ.get("AGENTSCOPE_MAX_SPANS", "100000"))
MAX_SESSIONS = int(os.environ.get("AGENTSCOPE_MAX_SESSIONS", "10000"))
RATE_LIMIT_REQUESTS = int(os.environ.get("AGENTSCOPE_RATE_LIMIT", "1000"))
RATE_LIMIT_WINDOW = float(os.environ.get("AGENTSCOPE_RATE_LIMIT_WINDOW", "60"))

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AgentScope Backend",
    description="Open-source AI Agent observability platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request/Response models ───────────────────────────────────────────────────

class SpanBatchRequest(BaseModel):
    """Request body for posting a batch of spans."""
    model_config = ConfigDict(extra="allow")

    project: str = "default"
    spans: list[dict] = Field(default_factory=list)

    @property
    def validated_spans(self) -> list[dict]:
        return self.spans


class SessionBatchRequest(BaseModel):
    """Request body for posting a batch of sessions."""
    model_config = ConfigDict(extra="allow")

    project: str = "default"
    sessions: list[dict] = Field(default_factory=list)


class SpanResponse(BaseModel):
    """Response after saving spans."""
    saved: int
    project: str


class SessionResponse(BaseModel):
    """Response after saving sessions."""
    saved: int
    project: str


class TraceListResponse(BaseModel):
    """Response for trace list query."""
    traces: list[dict]
    total: int
    limit: int
    offset: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float


# ── App state ─────────────────────────────────────────────────────────────────

_start_time = time.monotonic()
storage = get_storage(max_spans=MAX_SPANS, max_sessions=MAX_SESSIONS)


# ── Dependency: project name ───────────────────────────────────────────────────

def get_project(project: str = Query("default", description="Project name")) -> str:
    return project


def rate_limit(project: str = Depends(get_project)) -> None:
    """Apply rate limiting per project."""
    allowed, info = storage.check_rate_limit(
        project,
        max_requests=RATE_LIMIT_REQUESTS,
        window_seconds=RATE_LIMIT_WINDOW,
    )
    if not allowed:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                **info,
            },
            headers={"Retry-After": str(int(info["reset_in_seconds"]))},
        )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["system"])
async def health() -> HealthResponse:
    """Health check endpoint."""
    uptime = time.monotonic() - _start_time
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        uptime_seconds=round(uptime, 2),
    )


@app.post(
    "/api/v1/spans",
    response_model=SpanResponse,
    tags=["tracing"],
    dependencies=[Depends(rate_limit)],
)
async def create_spans(request: SpanBatchRequest) -> SpanResponse:
    """
    Accept a batch of spans from the SDK.

    Spans are stored in memory. For production, replace the storage layer
    with a persistent database (PostgreSQL, ClickHouse, etc.).
    """
    if not request.spans:
        return SpanResponse(saved=0, project=request.project)

    try:
        saved = storage.save_spans(request.spans, request.project)
        return SpanResponse(saved=len(saved), project=request.project)
    except Exception as exc:
        logger.exception("[AgentScope] Failed to save spans: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post(
    "/api/v1/sessions",
    response_model=SessionResponse,
    tags=["sessions"],
    dependencies=[Depends(rate_limit)],
)
async def create_sessions(request: SessionBatchRequest) -> SessionResponse:
    """Create or update trace sessions."""
    if not request.sessions:
        return SessionResponse(saved=0, project=request.project)

    try:
        saved = storage.save_sessions(request.sessions, request.project)
        return SessionResponse(saved=len(saved), project=request.project)
    except Exception as exc:
        logger.exception("[AgentScope] Failed to save sessions: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/traces", response_model=TraceListResponse, tags=["tracing"])
async def list_traces(
    project: str = Depends(get_project),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    search: str | None = Query(None, description="Search by trace name"),
) -> TraceListResponse:
    """List all traces for a project, newest first."""
    traces = storage.get_traces(project, limit=limit, offset=offset, search=search)
    total = len(traces)
    return TraceListResponse(
        traces=traces,
        total=total,
        limit=limit,
        offset=offset,
    )


@app.get("/api/v1/traces/{trace_id}", tags=["tracing"])
async def get_trace(trace_id: str) -> dict:
    """Get a single trace with all its spans."""
    trace_data = storage.get_trace(trace_id)
    if trace_data is None:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id!r} not found")
    return trace_data


@app.get("/api/v1/sessions/{session_id}", tags=["sessions"])
async def get_session(session_id: str) -> dict:
    """Get a single session."""
    session = storage.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found")
    return session


@app.get("/api/v1/stats", tags=["stats"])
async def get_stats(project: str = Depends(get_project)) -> dict:
    """Get usage statistics for a project."""
    return storage.get_stats(project)


@app.get("/api/v1/projects", tags=["admin"])
async def list_projects() -> dict:
    """List all projects that have data."""
    return {"projects": storage.get_all_projects()}


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
        headers=getattr(exc, "headers", None) or {},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("[AgentScope] Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# ── Startup / shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    logger.info(
        "[AgentScope] Backend starting — port %d, max_spans=%d, max_sessions=%d",
        PORT, MAX_SPANS, MAX_SESSIONS,
    )


@app.on_event("shutdown")
async def shutdown():
    logger.info("[AgentScope] Backend shutting down")


# ── CLI entry point ──────────────────────────────────────────────────────────

def main():
    import uvicorn
    uvicorn.run(
        "agentscope.backend.app:app",
        host=HOST,
        port=PORT,
        reload=os.environ.get("AGENTSCOPE_RELOAD", "0") == "1",
        log_level="info",
    )


if __name__ == "__main__":
    main()
