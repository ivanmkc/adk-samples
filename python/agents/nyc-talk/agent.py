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

"""This script defines a multi-agent system for an company info assistant.

The system consists of two agents:
1. user_info_collector: Collects the user's first name and age.
2. root_agent: The main agent that orchestrates the conversation, using the
   user_info_collector and a RAG tool to answer questions about companies.
"""

from google.adk.agents import Agent
from google.adk.tools.retrieval.vertex_ai_rag_retrieval import VertexAiRagRetrieval
from vertexai.preview import rag

# The large language model to use for the agents.
llm = "gemini-2.0-flash"  # Or your preferred model

# The agent responsible for collecting user information.
user_info_collector = Agent(
    name="user_info_collector",
    model=llm,
    instruction=(
        "Ask the user questions to collect user information. If they don't answer, reiterate that the info is"
        " needed before you can start helping. Be firm and don't give in. Once you collect the info, summarize it and pass it to the root_agent."
    ),
    description=(
        "An agent that collects user information, including first name and"
        " age."
    ),
)

# The ID of the RAG corpus to use for retrieval.
rag_corpus_id = "projects/169190568756/locations/us-central1/ragCorpora/1152921504606846976"

# The RAG retrieval tool for fetching documentation.
ask_vertex_retrieval = VertexAiRagRetrieval(
    name="retrieve_rag_documentation",
    description=(
        "Use this tool to retrieve documentation and reference materials about various companies."
    ),
    rag_resources=[
        rag.RagResource(rag_corpus=rag_corpus_id)
    ],
    similarity_top_k=10,
    vector_distance_threshold=0.6,
)

# The main agent of the system.
root_agent = Agent(
    name="company_info_assistant",
    model=llm,
    instruction=(
        "You are a helpful agent for Blue Ridge Outfitters. Introduce yourself as such and always provide sample questions when you prompt the user. You should only answer questions about this particular company."
        " Before you do anything, make sure you use user_info_collector to"
        " collect user information. Assume all ambiguous questions are about Blue Ridge Outfitters. For questions about the company, use the RAG tool to find relevant information, making sure you specify that you are asking about Blue Ridge Outfitters and not other companies. After gathering user info,"
        " you can prompt the user for their question."
    ),
    description=(
        "An assistant for gathering information and looking up info for Blue Ridge Outfitters."
    ),
    tools=[ask_vertex_retrieval],
    sub_agents=[user_info_collector],
)
