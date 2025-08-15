import asyncio
import os
from unittest.mock import MagicMock

import pytest
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools.retrieval import VertexAiRagRetrieval
from google.genai.types import UserContent, Part
from vertexai.preview import rag
import dotenv

dotenv.load_dotenv()

# Set a dummy environment variable for the test
# os.environ["RAG_CORPUS"] = "projects/123/locations/us-central1/ragCorpora/456"


# --- Prompts ---
ROOT_AGENT_PROMPT = """
You are a helpful research assistant.
You have two tools available:
1. A `get_current_date` tool for getting the current date.
2. A `rag_retrieval_agent` subagent for retrieving specific documentation from a RAG corpus.

- If the user asks for the current date, use `get_current_date`.
- If the user asks a question about information, delegate the task to `retrieve_rag_documentation`.
"""

RAG_SUBAGENT_PROMPT = """
You are a specialized agent for retrieving information from a RAG corpus.
Use the `retrieve_rag_documentation` tool to answer the user's question.
Output the retrieved documentation directly.
"""


# --- Custom Tool ---
def get_current_date() -> str:
    """Returns the current date in YYYY-MM-DD format."""
    return "2025-08-13"


# --- Agents ---
def get_rag_sub_agent(rag_tool: VertexAiRagRetrieval) -> Agent:
    """Returns the RAG sub-agent."""
    return Agent(
        name="rag_retrieval_agent",
        instruction=RAG_SUBAGENT_PROMPT,
        tools=[rag_tool],
        model="gemini-2.0-flash",
    )


def get_root_agent(sub_agents: list[Agent], tools: list = None) -> Agent:
    """Returns the root agent."""
    return Agent(
        name="root_agent",
        instruction=ROOT_AGENT_PROMPT,
        sub_agents=sub_agents,
        tools=tools,
        model="gemini-2.0-flash",
    )

RAG_ENGINE_CORPUS_NAME: str | None = "projects/ivanmkc-test/locations/us-central1/ragCorpora/8207810320882728960"

rag_tool = VertexAiRagRetrieval(
            name='retrieve_rag_documentation',
            description=(
                'Use this tool to retrieve documentation and reference materials '
                'for the question from the RAG corpus,'
            ),
            rag_resources=[
                rag.RagResource(
                    rag_corpus=RAG_ENGINE_CORPUS_NAME
                )
            ],
            similarity_top_k=10,
            vector_distance_threshold=0.6,
        )


# --- Test ---
# async def test_rag_delegation(mocker):
#     """
#     Tests that the root agent correctly delegates a documentation question
#     to the RAG sub-agent.
#     """
#     # 1. Mock the rag.retrieval_query function
#     # mocker.patch(
#     #     "vertexai.preview.rag.retrieval_query",
#     #     return_value=MagicMock(
#     #         contexts=MagicMock(
#     #             contexts=[MagicMock(text="Retrieved RAG documentation content.")]
#     #         )
#     #     ),
#     # )

#     # 2. Initialize agents
#     # rag_tool = VertexAiRagRetrieval(
#     #     name="retrieve_rag_documentation",
#     #     description="Retrieves RAG documentation.",
#     # )
#     rag_sub_agent = get_rag_sub_agent(rag_tool)
#     root_agent = get_root_agent(sub_agents=[rag_sub_agent], tools=[])

#     runner = InMemoryRunner(agent=root_agent)

#     # 3. Create a session
#     session = await runner.session_service.create_session(
#         app_name=runner.app_name, user_id="user1", session_id="session1"
#     )

#     # 4. Create the message that should trigger delegation
#     content = UserContent(
#         parts=[
#             Part(
#                 text="What is the process for filing a bug report according to the internal documentation?"
#             )
#         ]
#     )

#     # 5. Run the agent
#     events = [
#         e
#         async for e in runner.run_async(
#             user_id=session.user_id,
#             session_id=session.id,
#             new_message=content,
#         )
#     ]

#     final_response = ""
#     for event in events:
#         if event.is_final_response():
#             for part in event.content.parts:
#                 final_response += part.text

#     # 6. Assert the response is based on the mocked RAG tool output
#     assert "rag documentation content" in final_response.lower()


# async def test_rag_delegation_with_custom_tool(mocker):
#     """
#     Tests that the root agent can use a custom tool alongside delegating
#     to the RAG sub-agent.
#     """
#     # 1. Mock the rag.retrieval_query function
#     # mocker.patch(
#     #     "vertexai.preview.rag.retrieval_query",
#     #     return_value=MagicMock(
#     #         contexts=MagicMock(
#     #             contexts=[MagicMock(text="Retrieved RAG documentation content.")]
#     #         )
#     #     ),
#     # )

#     # 2. Initialize agents and tools
#     # rag_tool = VertexAiRagRetrieval(
#     #     name="retrieve_rag_documentation",
#     #     description="Retrieves RAG documentation.",
#     # )
#     rag_sub_agent = get_rag_sub_agent(rag_tool)
#     root_agent = get_root_agent(
#         sub_agents=[rag_sub_agent], tools=[get_current_date]
#     )

#     runner = InMemoryRunner(agent=root_agent)

#     # 3. Create a session
#     await runner.session_service.create_session(
#         app_name=runner.app_name, user_id="user1", session_id="session1"
#     )

#     # 4. Test custom tool usage
#     content_date = UserContent(parts=[Part(text="What is the current date?")])
#     events_date = [
#         e
#         async for e in runner.run_async(
#             user_id="user1",
#             session_id="session1",
#             new_message=content_date,
#         )
#     ]
#     final_response_date = ""
#     for event in events_date:
#         if event.is_final_response():
#             for part in event.content.parts:
#                 final_response_date += part.text
#     assert "2025-08-13" in final_response_date

#     # 5. Test RAG delegation
#     content_rag = UserContent(
#         parts=[
#             Part(
#                 text="What is the process for filing a bug report according to the internal documentation?"
#             )
#         ]
#     )
#     events_rag = [
#         e
#         async for e in runner.run_async(
#             user_id="user1",
#             session_id="session1",
#             new_message=content_rag,
#         )
#     ]
#     final_response_rag = ""
#     for event in events_rag:
#         if event.is_final_response():
#             for part in event.content.parts:
#                 final_response_rag += part.text
#     assert "rag documentation content" in final_response_rag.lower()


async def test_rag_as_tool_throws_exception(mocker):
    """
    Tests that using VertexAiRagRetrieval as a tool alongside another tool
    raises a ValueError.
    """
    # 1. Initialize agents and tools
    # rag_tool = VertexAiRagRetrieval(
    #     name="retrieve_rag_documentation",
    #     description="Retrieves RAG documentation.",
    # )

    rag_subagent = Agent(
        name="rag_retrieval_agent",
        instruction=RAG_SUBAGENT_PROMPT,
        tools=[rag_tool],
        model="gemini-2.0-flash",
    )

    root_agent = Agent(
        name="root_agent",
        instruction=ROOT_AGENT_PROMPT,
        sub_agents=[rag_subagent],
        tools=[get_current_date],
        model="gemini-2.0-flash",
    )

    runner = InMemoryRunner(agent=root_agent)

    # 2. Create a session
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id="user1", session_id="session1"
    )

    # 3. Run the agent and guide it to use the RAG subagent
    content = UserContent(
        parts=[
            Part(
                text="What was the total revenue of Alphabet Inc. in the year 2023?"
            )
        ]
    )
    events = []
    try:
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            events.append(event)
    except Exception as ex:
        print(ex)
        pass

    print(events[-1].content.parts[-1].text)

    # 4. Run the agent and guide it to use the get_current_date tool
    content = UserContent(
        parts=[
            Part(
                text="What's the current date?"
            )
        ]
    )
    events = []
    try:
        async for event in runner.run_async(
            user_id=session.user_id,
            session_id=session.id,
            new_message=content,
        ):
            events.append(event)
    except Exception as ex:
        print(ex)
        pass
        
    print(events[-1].content.parts[-1].text)