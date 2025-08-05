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

"""LLM Auditor for verifying & refining LLM-generated answers using the web."""

from google.adk.agents import SequentialAgent

from .sub_agents.critic import create_critic_agent
from .sub_agents.reviser import create_reviser_agent


def create_llm_auditor(
    agent_name: str,
    critic_agent_name: str,
    reviser_agent_name: str,
    rag_corpus_id: str | None = None,
    critic_agent_model: str = "gemini-2.0-flash",
    format_agent_model: str = "gemini-2.0-flash"
) -> SequentialAgent:
    """Constructs the complete llm_auditor sequential agent.

    This root agent is designed to critically evaluate LLM-generated answers.
    It encapsulates a `critic_agent` for evaluation and a `reviser_agent`
    for refinement, chaining them together to ensure responses are accurate
    and aligned with real-world knowledge.

    Args:
        rag_corpus_id: Optional. The full resource name of the RAG corpus to be
            used by the underlying critic agent. If None, the RAG tool will not
            be used.

            e.g. projects/ivanmkc-test/locations/us-central1/ragCorpora/8207810320882728960
        critic_agent_model: The model name for the core critiquing sub-agent
            within the `critic_agent`. This agent uses tools like Google Search
            and (optionally) RAG to perform the evaluation.
        format_agent_model: The model name for the formatting sub-agent
            within the `critic_agent`. This explicitly controls the model used
            for the final formatting step of the critique.

    Returns:
        An instance of the fully configured `llm_auditor` SequentialAgent.
    """
    # Create the nested critic agent, passing along the specified models.
    critic_agent = create_critic_agent(
        agent_name=critic_agent_name,
        rag_corpus_id=rag_corpus_id,
        critic_agent_model=critic_agent_model,
        format_agent_model=format_agent_model
    )

    reviser_agent = create_reviser_agent(
        agent_name=reviser_agent_name
    )

    # Assemble the final auditor by chaining the critic and reviser.
    llm_auditor = SequentialAgent(
        name=agent_name,
        description=(
            'Evaluates LLM-generated answers, verifies actual accuracy using the'
            ' web, and refines the response to ensure alignment with real-world'
            ' knowledge.'
        ),
        sub_agents=[critic_agent, reviser_agent],
    )

    return llm_auditor

root_agent = create_llm_auditor(
    agent_name="llm_auditor",
    critic_agent_name="critic_agent",
    reviser_agent_name="reviser_agent",
)
