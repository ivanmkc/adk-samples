# Company Info Assistant

This agent is a multi-agent system designed to assist users with information about the company **Blue Ridge Outfitters**. It first collects the user's name and age, and then uses a Retrieval Augmented Generation (RAG) tool to answer questions about the company.

## Setup

1.  **Create a virtual environment:**

    ```bash
    python3 -m venv env
    ```

2.  **Activate the virtual environment:**

    ```bash
    source env/bin/activate
    ```

3.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Create a `.env` file** and add the following environment variables:

    ```
    PROJECT_ID="your-gcp-project-id"
    LOCATION="your-gcp-location"
    ```

## Running the Agent

To run the agent, use the following command:

```bash
adk run agent:root_agent
```

## Agents

This system consists of two agents:

*   `user_info_collector`: This agent is responsible for collecting the user's first name and age before the conversation begins.
*   `root_agent`: This is the main agent that orchestrates the conversation. It uses the `user_info_collector` to gather user information and then utilizes a RAG tool to answer questions specifically about Blue Ridge Outfitters.
