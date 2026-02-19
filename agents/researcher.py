"""Researcher agent – investigates a single sub-question using a search tool."""

from __future__ import annotations

from typing import Any, Callable

from agents.base import BaseAgent


_SEARCH_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": (
                "Search for information relevant to the query. "
                "Returns a list of text excerpts with source URLs."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "A concise search query (max 10 words).",
                    }
                },
                "required": ["query"],
            },
        },
    }
]


class ResearcherAgent(BaseAgent):
    """Level-1 agent that researches a single sub-question.

    It is given a ``search`` callable at construction time so that the
    underlying search backend can be swapped (real API, stub, mock …).
    """

    system_prompt = (
        "You are a rigorous research assistant. "
        "You will be given a focused research sub-question. "
        "Use the 'search' tool to retrieve relevant information. "
        "Summarise what you found in 2–4 sentences and list every source URL "
        "you relied on. Do not invent facts or sources that were not returned "
        "by the search tool. "
        "Format your answer as:\n"
        "SUMMARY: <your summary>\n"
        "SOURCES: <comma-separated list of URLs>"
    )

    tools = _SEARCH_TOOL_SCHEMA

    def __init__(self, search_fn: Callable[[str], Any], **kwargs: Any):
        super().__init__(**kwargs)
        self._search_fn = search_fn

    def research(self, sub_question: str) -> dict[str, Any]:
        """Investigate *sub_question* and return a structured finding dict."""
        tool_registry = {"search": self._search_fn}
        raw = self.run(sub_question, tool_registry=tool_registry)

        summary = raw
        sources: list[str] = []

        for line in raw.splitlines():
            if line.startswith("SUMMARY:"):
                summary = line[len("SUMMARY:"):].strip()
            elif line.startswith("SOURCES:"):
                sources = [s.strip() for s in line[len("SOURCES:"):].split(",") if s.strip()]

        return {
            "sub_question": sub_question,
            "summary": summary,
            "sources": sources,
        }
