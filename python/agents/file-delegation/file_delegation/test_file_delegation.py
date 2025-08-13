import os
import asyncio
import base64
from PIL import Image
import pytest
from google.adk import Agent
from google.adk.runners import InMemoryRunner
from google.genai.types import Content, Part, Blob

# Prompts
ROOT_AGENT_PROMPT = """
You are a root agent that can delegate tasks to sub-agents.
You have an image processing sub-agent that can analyze images.
If the user asks you to process an image, delegate the task to the image processing sub-agent.
The user has provided an image. Delegate the task to the image processing sub-agent.
"""

IMAGE_PROCESSING_SUB_AGENT_PROMPT = """
You are an expert in image analysis.
Describe the image provided by the user in detail.
"""

# Agents
def get_image_processing_sub_agent() -> Agent:
    """Returns the image processing sub-agent."""
    return Agent(
        name="image_processing_sub_agent",
        instruction=IMAGE_PROCESSING_SUB_AGENT_PROMPT,
        model="gemini-2.5-flash",
    )

def get_root_agent(sub_agents: list[Agent]) -> Agent:
    """Returns the root agent."""
    return Agent(
        name="root_agent",
        instruction=ROOT_AGENT_PROMPT,
        sub_agents=sub_agents,
        model="gemini-2.5-flash",
    )

# Test
def test_file_delegation():
    # Create a dummy image for testing
    if not os.path.exists("dummy_image.png"):
        img = Image.new("RGB", (60, 30), color="red")
        img.save("dummy_image.png")

    image_processing_sub_agent = get_image_processing_sub_agent()

    root_agent = get_root_agent(
        sub_agents=[image_processing_sub_agent]
    )

    runner = InMemoryRunner(agent=root_agent)

    # Create a session
    async def create_session():
        await runner.session_service.create_session(
            app_name=runner.app_name, user_id="user1", session_id="session1"
        )

    asyncio.run(create_session())

    # Create the message content
    with open("dummy_image.png", "rb") as f:
        file_bytes = f.read()

    message = Content(
        role="user",
        parts=[
            Part(text="Please process this image and tell me what you see."),
            Part(inline_data=Blob(data=file_bytes, mime_type="image/png")),
        ]
    )

    # Run the agent
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

    # Clean up the dummy image
    os.remove("dummy_image.png")

    print(f"Response: {final_response}")

    assert "red" in final_response.lower()