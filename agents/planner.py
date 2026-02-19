"""Planner agent – decomposes a broad research question into sub-questions."""

from __future__ import annotations

import json
import re

from agents.base import BaseAgent


class PlannerAgent(BaseAgent):
    """Level-1 agent responsible for question decomposition.

    Given a broad research question it returns a JSON list of 3–6 focused
    sub-questions that together cover the original question.
    """

    system_prompt = (
        "You are a research planning expert. "
        "Your sole task is to break a broad research question into 3 to 6 "
        "focused, non-overlapping sub-questions that together fully cover the topic. "
        "Return ONLY a JSON array of strings – no markdown, no explanation. "
        "Example output: [\"Sub-question 1\", \"Sub-question 2\"]"
    )

    def decompose(self, research_question: str) -> list[str]:
        """Return a list of sub-questions for *research_question*."""
        raw = self.run(f"Research question: {research_question}")

        # Strip possible markdown fences before parsing
        raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`")

        try:
            sub_questions = json.loads(raw)
            if isinstance(sub_questions, list) and all(
                isinstance(q, str) for q in sub_questions
            ):
                return sub_questions
        except json.JSONDecodeError:
            pass

        # Fallback: split on newlines and treat each non-empty line as a sub-question
        lines = [line.strip(" -•0123456789.)") for line in raw.splitlines()]
        return [line for line in lines if line]
