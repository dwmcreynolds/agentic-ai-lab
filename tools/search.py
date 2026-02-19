"""Search tool abstraction used by ResearcherAgent instances.

Two implementations are provided:

* ``SearchTool``     – wraps a real external search API (configurable).
* ``StubSearchTool`` – returns deterministic fake results; useful for
                        offline development and unit testing.
"""

from __future__ import annotations

import os
from typing import Any


class SearchTool:
    """Thin wrapper around an external search API.

    By default the tool expects a ``SEARCH_API_KEY`` environment variable and
    calls a generic search endpoint.  Subclass or monkey-patch ``_fetch`` to
    adapt to your preferred provider (SerpAPI, Tavily, Bing, …).

    The callable interface (``__call__``) is what ResearcherAgent injects.
    """

    def __init__(self, api_key: str | None = None, max_results: int = 5):
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY", "")
        self.max_results = max_results

    def __call__(self, query: str) -> str:
        """Run *query* and return a plain-text result block."""
        results = self._fetch(query)
        if not results:
            return "No results found."
        lines: list[str] = []
        for r in results[: self.max_results]:
            lines.append(f"[{r.get('title', 'No title')}] {r.get('snippet', '')}")
            lines.append(f"URL: {r.get('url', 'N/A')}")
            lines.append("")
        return "\n".join(lines).strip()

    def _fetch(self, query: str) -> list[dict[str, Any]]:
        """Call the external search API and return a list of result dicts.

        Each dict should contain at least 'title', 'snippet', and 'url'.
        Override this method to integrate a real search provider.
        """
        raise NotImplementedError(
            "SearchTool._fetch() must be implemented for a real search provider. "
            "Use StubSearchTool for offline/testing use cases."
        )


class StubSearchTool:
    """Deterministic stub search tool for offline development and testing.

    Returns plausible-looking but entirely fake results so that the full
    agent pipeline can be exercised without any API keys.
    """

    _STUBS: list[dict[str, Any]] = [
        {
            "title": "Overview of the topic",
            "snippet": "This article provides a comprehensive introduction to the subject.",
            "url": "https://example.com/overview",
        },
        {
            "title": "Recent advances",
            "snippet": "Researchers have made significant progress in this area over the past five years.",
            "url": "https://example.com/recent-advances",
        },
        {
            "title": "Key challenges",
            "snippet": "Several open problems remain, including scalability and interpretability.",
            "url": "https://example.com/challenges",
        },
    ]

    def __call__(self, query: str) -> str:
        lines: list[str] = [f"Results for: {query}", ""]
        for r in self._STUBS:
            lines.append(f"[{r['title']}] {r['snippet']}")
            lines.append(f"URL: {r['url']}")
            lines.append("")
        return "\n".join(lines).strip()
