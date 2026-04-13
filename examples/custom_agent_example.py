"""
Example: Custom agent with manual span management.

For agents that don't use a supported framework, use trace_context directly.
"""

import time

from agentscope_trace import trace_context, SpanKind, LLMCall, ToolCall


def mock_llm_call(prompt: str) -> str:
    """Simulated LLM call."""
    time.sleep(0.1)
    return f"Analysis of: {prompt[:50]}... (simulated response)"


def mock_search_tool(query: str) -> str:
    """Simulated web search."""
    time.sleep(0.2)
    return f"Found 42 results for: {query}"


def mock_rag_retriever(query: str) -> list[str]:
    """Simulated RAG retrieval."""
    time.sleep(0.05)
    return [
        "Document 1: Relevant information about the query...",
        "Document 2: Additional context...",
    ]


def research_agent(user_query: str) -> str:
    """
    A multi-step research agent that:
    1. Embeds the query
    2. Retrieves relevant documents
    3. Calls the LLM to synthesize
    4. Optionally searches the web
    """
    # Top-level agent span
    with trace_context(
        name="research-agent",
        kind=SpanKind.AGENT,
        metadata={"user_query": user_query},
    ) as span:

        # Step 1: Retrieve documents
        with trace_context(
            name="retrieve-documents",
            kind=SpanKind.RETRIEVER,
            metadata={"query": user_query},
        ):
            docs = mock_rag_retriever(user_query)

        # Step 2: LLM synthesis
        with trace_context(
            name="synthesize-response",
            kind=SpanKind.LLM,
            metadata={"doc_count": len(docs)},
        ) as llm_span:
            prompt = f"Based on these documents:\n{chr(10).join(docs)}\n\nAnswer: {user_query}"
            response = mock_llm_call(prompt)

            llm_span.llm_call = LLMCall(
                model="gpt-4o-mini",
                prompt=prompt,
                completion=response,
                input_tokens=len(prompt.split()),
                output_tokens=len(response.split()),
                latency_ms=100,
            )

        # Step 3: Web search (conditionally)
        if len(response) < 100:
            with trace_context(
                name="web-search",
                kind=SpanKind.TOOL,
            ) as tool_span:
                search_result = mock_search_tool(user_query)
                tool_span.tool_calls.append(ToolCall(
                    name="web_search",
                    arguments={"query": user_query},
                    result=search_result,
                    latency_ms=200,
                ))

        return response


def main():
    result = research_agent(
        "What are the latest developments in AI agent frameworks?"
    )
    print(f"Agent response:\n{result}")
    print("\n✅ Check http://localhost:3000 to see the full trace tree!")


if __name__ == "__main__":
    main()
