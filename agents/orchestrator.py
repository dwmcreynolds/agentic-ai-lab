"""Orchestrator agent – top-level coordinator for the agent hierarchy.

Hierarchy
---------
Level 0  OrchestratorAgent   (this module)
Level 1  PlannerAgent        – decomposes the research question
Level 1  ResearcherAgent     – one instance per sub-question
Level 1  SynthesizerAgent    – merges findings into a final report
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from openai import OpenAI

from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.synthesizer import SynthesizerAgent
from memory.store import MemoryStore

logger = logging.getLogger(__name__)


class OrchestratorAgent:
    """Coordinates the full research pipeline.

    Parameters
    ----------
    search_fn:
        A callable ``(query: str) -> Any`` used by every ResearcherAgent.
        Inject a stub/mock here during testing.
    client:
        Optional shared OpenAI client.  A new client is created if omitted.
    model:
        Model name forwarded to every sub-agent.
    max_sub_questions:
        Hard cap on the number of sub-questions the Planner may return.
    """

    def __init__(
        self,
        search_fn: Callable[[str], Any],
        client: OpenAI | None = None,
        model: str = "gpt-4o-mini",
        max_sub_questions: int = 6,
    ):
        self._search_fn = search_fn
        self._client = client
        self._model = model
        self._max_sub_questions = max_sub_questions
        self._memory = MemoryStore()

        # Instantiate sub-agents (Level 1)
        shared = dict(client=self._client, model=self._model)
        self._planner = PlannerAgent(**shared)
        self._synthesizer = SynthesizerAgent(**shared)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, research_question: str) -> str:
        """Execute the full research pipeline and return the final report.

        Steps
        -----
        1. Planner decomposes the question into sub-questions.
        2. One ResearcherAgent per sub-question is dispatched sequentially.
        3. Synthesizer merges all findings into a structured report.
        """
        logger.info("Orchestrator: starting research for %r", research_question)
        self._memory.clear()

        # --- Step 1: Planning ---
        logger.info("Orchestrator → Planner: decomposing question")
        sub_questions = self._planner.decompose(research_question)
        sub_questions = sub_questions[: self._max_sub_questions]
        logger.info("Planner produced %d sub-questions", len(sub_questions))
        self._memory.store("sub_questions", sub_questions)

        # --- Step 2: Research (one agent per sub-question) ---
        findings: list[dict[str, Any]] = []
        for idx, sub_q in enumerate(sub_questions, start=1):
            logger.info("Orchestrator → Researcher %d/%d: %r", idx, len(sub_questions), sub_q)
            researcher = ResearcherAgent(
                search_fn=self._search_fn,
                client=self._client,
                model=self._model,
            )
            finding = researcher.research(sub_q)
            findings.append(finding)
            self._memory.store(f"finding_{idx}", finding)
            logger.info("Researcher %d complete: %d source(s)", idx, len(finding["sources"]))

        self._memory.store("findings", findings)

        # --- Step 3: Synthesis ---
        logger.info("Orchestrator → Synthesizer: generating final report")
        report = self._synthesizer.synthesize(research_question, findings)
        self._memory.store("report", report)

        logger.info("Orchestrator: pipeline complete")
        return report

    @property
    def memory(self) -> MemoryStore:
        """Access the shared memory store (useful for inspection / testing)."""
        return self._memory
