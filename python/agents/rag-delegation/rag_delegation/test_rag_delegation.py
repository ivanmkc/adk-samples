import asyncio
import os
from unittest.mock import MagicMock

import pytest
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import GoogleSearchTool
from google.adk.tools.retrieval import VertexAiRagRetrieval
from google.generativeai.types import to_content

# Set a dummy environment variable for the test
os.environ["RAG_CORPUS"] = "projects/123/locations/us-central1/ragCorpora/456"

# --- Prompts ---
ROOT_AGENT_PROMPT = """
You are a helpful research assistant.
You have two tools available:
1. A `google_search` tool for general web searches.
2. A sub-agent named `rag_retrieval_agent` for retrieving specific documentation from a RAG corpus.

- If the user asks a general question, use `google_search`.
- If the user asks a question about specific internal documentation, delegate the task to `rag_retrieval_agent`.
"""

RAG_SUBAGENT_PROMPT = """
You are a specialized agent for retrieving information from a RAG corpus.
Use the `retrieve_rag_documentation` tool to answer the user's question based on the provided documentation.
"""


# --- Agents ---
def get_rag_sub_agent(rag_tool: VertexAiRagRetrieval) -> Agent:
    """Returns the RAG sub-agent."""
    return Agent(
        name="rag_retrieval_agent",
        instruction=RAG_SUBAGENT_PROMPT,
        tools=[rag_tool],
        model="gemini-1.5-flash",
    )


def get_root_agent(sub_agents: list[Agent]) -> Agent:
    """Returns the root agent."""
    return Agent(
        name="root_agent",
        instruction=ROOT_AGENT_PROMPT,
        sub_agents=sub_agents,
        tools=[GoogleSearchTool()],
        model="gemini-1.5-flash",
    )


# --- Test ---
def test_rag_delegation(mocker):
    """
    Tests that the root agent correctly delegates a documentation question
    to the RAG sub-agent.
    """
    # 1. Mock the VertexAiRagRetrieval tool
    mock_rag_retrieval = MagicMock(spec=VertexAiRagRetrieval)
    mock_rag_retrieval.name = "retrieve_rag_documentation"
    mock_rag_retrieval.description = "Retrieves RAG documentation."
    # Mock the synchronous run method for the tool
    mocker.patch.object(
        VertexAiRagRetrieval,
        "run",
        return_value={"result": "Retrieved RAG documentation content."},
    )
    # Since the tool is run in an async context by the runner, we also mock the async version
    mocker.patch.object(
        VertexAiRagRetrieval,
        "run_async",
        return_value={"result": "Retrieved RAG documentation content."},
    )

    # 2. Initialize agents
    rag_sub_agent = get_rag_sub_agent(mock_rag_retrieval)
    root_agent = get_root_agent(sub_agents=[rag_sub_agent])

    runner = InMemoryRunner(agent=root_agent)

    # 3. Create a session
    async def create_session():
        await runner.session_service.create_session(
            app_name=runner.app_name, user_id="user1", session_id="session1"
        )

    asyncio.run(create_session())

    # 4. Create the message that should trigger delegation
    message = to_content(
        {"role": "user", "parts": ["What is the process for filing a bug report according to the internal documentation?"]}
    )

    # 5. Run the agent
    events = runner.run(
        user_id="user1",
        session_id="session1",
        new_message=message,
    )

    final_response = ""
    for event in events:
        if event.is_final_response():
            for part in event.content.parts:
                final_response += part.text

    # 6. Assert the response is based on the mocked RAG tool output
    assert "rag documentation content" in final_response.lower()

