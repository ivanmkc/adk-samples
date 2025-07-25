# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Critic agent for identifying and verifying statements using search tools."""

from google.adk import Agent
from google.adk.agents import SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models import LlmResponse
from google.adk.tools import google_search, VertexAiSearchTool
from google.adk.tools.agent_tool import AgentTool

from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

from google.genai import types

from . import prompt


def _render_reference(
    callback_context: CallbackContext,
    llm_response: LlmResponse,
) -> LlmResponse:
    """Appends grounding references to the response."""
    del callback_context
    if (
        not llm_response.content or
        not llm_response.content.parts or
        not llm_response.grounding_metadata
    ):
        return llm_response
    references = []
    for chunk in llm_response.grounding_metadata.grounding_chunks or []:
        title, uri, text = '', '', ''
        if chunk.retrieved_context:
            title = chunk.retrieved_context.title
            uri = chunk.retrieved_context.uri
            text = chunk.retrieved_context.text
        elif chunk.web:
            title = chunk.web.title
            uri = chunk.web.uri
        parts = [s for s in (title, text) if s]
        if uri and parts:
            parts[0] = f'[{parts[0]}]({uri})'
        if parts:
            references.append('* ' + ': '.join(parts) + '\n')
    if references:
        reference_text = ''.join(['\n\nReference:\n\n'] + references)
        llm_response.content.parts.append(types.Part(text=reference_text))
    if all(part.text is not None for part in llm_response.content.parts):
        all_text = '\n'.join(part.text for part in llm_response.content.parts)
        llm_response.content.parts[0].text = all_text
        del llm_response.content.parts[1:]
    return llm_response


def create_critic_agent(
    agent_name: str,
    rag_corpus_id: str | None = None,
    critic_agent_model: str = "gemini-2.0-flash",
    format_agent_model: str = "gemini-2.0-flash"
) -> SequentialAgent:
    """
    Creates a sequential critic agent with configurable models and an optional RAG tool.

    Args:
        rag_corpus_id: The full resource name of the RAG corpus. If provided,
                       a RAG retrieval tool will be included.
        critic_agent_model: The model name for the main critic agent.
        format_agent_model: The model name for the final formatting agent.

    Returns:
        An instance of the configured SequentialAgent.
    """
    # Search agent and tool definition
    _search_agent = Agent(
        model="gemini-2.0-flash",
        name="google_search_grounding",
        description="An agent providing Google-search grounding capability",
        instruction=""",
        Answer the user's question directly using google_search grounding tool; Provide a brief but concise response. 
        Rather than a detail response, provide the immediate actionable item for a tourist or traveler, in a single sentence.
        Do not ask the user to check or look up information for themselves, that's your role; do your best to be informative.
        """,
        tools=[google_search],
    )    
    
    # Initialize the list of tools for the critic agent
    google_search_grounding = AgentTool(agent=_search_agent)
    critic_tools = [google_search_grounding]

    # Conditionally create and add the RAG tool if an ID is provided
    if rag_corpus_id:
        ask_vertex_retrieval = VertexAiRagRetrieval(
            name='retrieve_rag_documentation',
            description=(
                'Use this tool to retrieve documentation and reference materials '
                'for the question from the RAG corpus,'
            ),
            rag_resources=[
                rag.RagResource(
                    rag_corpus=rag_corpus_id
                )
            ],
            similarity_top_k=10,
            vector_distance_threshold=0.6,
        )
        critic_tools.append(ask_vertex_retrieval)

    # Critic agent that uses the configured list of tools and model
    critic_agent_with_tools = Agent(
        model=critic_agent_model,
        name='critic_agent_with_tools',
        instruction=prompt.CRITIC_PROMPT,
        tools=critic_tools,
    )

    # Formatting agent with its configured model
    format_agent = Agent(
        name="format_agent",
        model=format_agent_model,
        description="This is an agent that formats the answers from the agent 'critic_agent_with_tools'.",
        instruction="You are an agent that formats the answers from the critique agent.",
        output_schema=prompt.CriticOutput
    )

    # Root sequential agent
    critic_agent = SequentialAgent(
        name=agent_name,
        description="This is the root agent that coordinates the critiquing and formatting.",
        sub_agents=[critic_agent_with_tools, format_agent],
    )

    return critic_agent

critic_agent = create_critic_agent(agent_name="critic_agent")