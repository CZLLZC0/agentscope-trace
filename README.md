# 🔭 AgentScope

<div align="center">

**Open-source AI Agent observability platform.**  
Trace every decision, measure every token, debug every failure.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python: 3.9+](https://img.shields.io/badge/Python-3.9+-green.svg)](pyproject.toml)
[![Stars](https://img.shields.io/github/stars/CZLLZC0/agentscope?style=social)](https://github.com/CZLLZC0/agentscope)
[![Twitter](https://img.shields.io/twitter/follow/agentscope?style=social)](https://twitter.com/agentscope)

*"LangSmith for the rest of us."* — Every AI developer who can't afford $165/month

[Quick Start](#-quick-start) · [Features](#-features) · [Architecture](#-architecture) · [Roadmap](#-roadmap) · [Contributing](#-contributing)

</div>

---

## 🤔 Why AgentScope?

When your AI agent fails in production, you need to know **why**.

LangSmith is great — if you're enterprise.  
If you're building, indie, or cost-conscious: you need something that works
**offline, self-hosted, and free**.

AgentScope gives you:
- Complete execution traces (every LLM call, every tool, every decision)
- Token cost tracking per span and per trace
- Latency analysis (find your bottleneck step)
- Error attribution (which tool failed?)
- Replay with parameter overrides (debug without re-running)

All of this, **under 1500 lines of code**, minimal dependencies, runs anywhere.

---

## ⚡ Quick Start

### 1. Install

```bash
pip install agentscope-trace
```

### 2. Add tracing to your agent

**Option A — Decorator (simplest)**
```python
from agentscope_trace import trace, trace_context, SpanKind, LLMCall

@trace(name="my-agent", kind=SpanKind.AGENT)
def my_agent(user_input: str) -> str:
    context = retrieve_context(user_input)
    response = llm.invoke(context)
    return response

result = my_agent("What's the weather in Tokyo?")
```

**Option B — Context manager (more control)**
```python
from agentscope_trace import trace_context, SpanKind, LLMCall

with trace_context("synthesis", kind=SpanKind.LLM) as span:
    span.llm_call = LLMCall(
        model="gpt-4o-mini",
        prompt="Analyze: " + user_input,
        latency_ms=120.0,
    )
    response = llm.invoke("Analyze: " + user_input)
    span.llm_call.completion = response.content
```

### 3. Start the backend (optional but recommended)

```bash
cd backend
pip install -r requirements.txt
uvicorn agentscope_trace.backend.app:app --reload --port 8000
```

Backend runs at `http://localhost:8000`.  
API docs at `http://localhost:8000/docs`.

### 4. Integrate with LangChain (one line)

```python
from langchain_openai import ChatOpenAI
from agentscope_trace import AgentScopeCallbackHandler

handler = AgentScopeCallbackHandler()
llm = ChatOpenAI(callbacks=[handler], model="gpt-4o-mini")
# Every LangChain chain is now traced automatically
```

---

## 🧩 Features

| Feature | AgentScope | LangSmith | Phoenix |
|---------|-----------|-----------|---------|
| Trace LLM calls | ✅ | ✅ | ✅ |
| Trace tool calls | ✅ | ✅ | ✅ |
| Token cost tracking | ✅ | ✅ | ❌ |
| Self-hosted / offline | ✅ | ❌ | ✅ |
| Open source | ✅ | ❌ | ⚠️ |
| Replay with overrides | 🔜 | ✅ | ❌ |
| Multi-framework hooks | 🔜 | ✅ | ⚠️ |
| Free tier | ✅ | ❌ ($$$) | ✅ |
| Docker one-click deploy | ✅ | ❌ | ⚠️ |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Your Agent Code                  │
│  @trace / trace_context / LangChain Callback        │
└──────────────────────┬──────────────────────────────┘
                       │  HTTP POST (background queue)
                       ▼
┌──────────────────────────────────────────────────────┐
│              AgentScope SDK (client.py)              │
│  Batches spans · Retry with backoff · Graceful deg. │
└──────────────────────┬──────────────────────────────┘
                       │  JSON over HTTP
                       ▼
┌──────────────────────────────────────────────────────┐
│              AgentScope Backend (FastAPI)             │
│  /api/v1/spans · /api/v1/traces · /api/v1/stats     │
└──────────────────────┬──────────────────────────────┘
                       ▼
┌──────────────────────────────────────────────────────┐
│              In-Memory Storage (SQLite-capable)       │
│  Thread-safe, RLock, LRU eviction                    │
└──────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
agentscope/
├── src/agentscope/
│   ├── __init__.py      # Public API
│   ├── models.py        # Span, LLMCall, ToolCall, TraceSession
│   ├── client.py        # HTTP client with retry & batching
│   ├── decorator.py     # @trace, trace_context, LangChain handler
│   └── backend/
│       ├── app.py       # FastAPI application
│       └── storage.py  # Thread-safe in-memory storage
├── examples/
│   ├── langchain_example.py
│   └── custom_agent_example.py
├── tests/               # pytest test suite
├── docker-compose.yml   # One-command start
└── pyproject.toml       # Python package config
```

---

## 🔜 Roadmap

- [ ] **v0.2.0** — Frontend dashboard (trace tree visualization)
- [ ] **v0.3.0** — Replay & parameter override
- [ ] **v0.4.0** — Dify / AutoGen framework hooks
- [ ] **v0.5.0** — PostgreSQL + ClickHouse storage backend
- [ ] **v1.0.0** — Stable API, benchmarks

---

## 🤝 Contributing

```bash
pip install -e ".[dev]"
pytest tests/ -v
ruff check src/
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
