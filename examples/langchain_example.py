"""
Example: Using agentscope-trace with LangChain.

Install: pip install agentscope-trace[langchain]
"""

import os

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage

from agentscope_trace import AgentScopeCallbackHandler, trace


def main():
    # Create LangChain callback handler
    handler = AgentScopeCallbackHandler()

    # Create LLM with tracing
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        callbacks=[handler],
    )

    # Trace a research task
    @trace(name="research-agent", metadata={"task": "topic-analysis"})
    def research_agent(topic: str) -> str:
        messages = [
            HumanMessage(content=f"Give me a brief analysis of: {topic}")
        ]
        response = llm.invoke(messages)
        return response.content

    # Run it
    result = research_agent("the impact of large language models on software development")
    print(f"Result: {result[:100]}...")

    print("\n✅ Check http://localhost:3000 to see the trace!")


if __name__ == "__main__":
    main()
