import asyncio
from typing import Type
from async_lru import alru_cache

from google.adk.agents import SequentialAgent
from google.adk.runners import InMemoryRunner
from google.genai.types import Part, UserContent
from google.adk.events import Event
from google.adk.plugins import LogCollectorPlugin

class ClaimReviser:
    """
    A service class to revise claims using a provided LLM auditor agent.

    This class encapsulates the logic for running an agent to evaluate and
    revise a given text claim. It manages the runner and session lifecycle.
    """

    def __init__(self, llm_auditor: SequentialAgent):
        """
        Initializes the ClaimReviser with a specific LLM auditor agent.

        Args:
            llm_auditor: An instance of a sequential agent configured for
                         auditing and revising claims.
        """
        if not llm_auditor:
            raise ValueError("An llm_auditor agent instance is required.")
        
        # TODO: Check if logger needs to be type-safe or not
        self.logger = LogCollectorPlugin()
        self.auditor_runner = InMemoryRunner(agent=llm_auditor, plugins=[self.logger])

    def _create_verification_prompt(self, claim: str) -> str:
        """Creates the prompt to send to the agent."""
        return f"Verify this claim: {claim}"

    @alru_cache(maxsize=None)
    async def revise_claim(self, claim: str) -> list[Event]:
        """
        Revises a single claim using the configured runner session.

        This method is cached to avoid re-evaluating the same claim.

        Args:
            claim: The claim string to be evaluated.

        Returns:
            The rewritten claim provided by the agent.
        """
        session = await self.auditor_runner.session_service.create_session(
            app_name=self.auditor_runner.app_name, user_id="test_user"
        )
        content = UserContent(parts=[Part(text=self._create_verification_prompt(claim))])
        events = []
        async for event in self.auditor_runner.run_async(
            user_id=session.user_id, session_id=session.id, new_message=content
        ):
            events.append(event)

        # Assuming the final event contains the revised text
        # raw_text = events[-1].content.parts[0].text
        # return raw_text
        return events

    async def revise_claim_async(self, claim: str, semaphore: asyncio.Semaphore) -> list[Event] | None:
        """
        A concurrent-safe wrapper for revising a single claim.

        This method uses a semaphore to limit the number of concurrent
        revisions and includes basic error handling.

        Args:
            claim: The claim string to evaluate.
            semaphore: An asyncio.Semaphore to limit concurrent access.

        Returns:
            The revised claim string, or None if an error occurs.
        """
        async with semaphore:
            try:
                result = await self.revise_claim(claim=claim)
                return result
            except Exception as e:
                print(f"An error occurred during evaluation of '{claim}': {e}")
                return None
