"""Search tool abstraction used by ResearcherAgent instances.

Three implementations are provided:

* ``ExaSearchTool`` – production search via the Exa neural search API.
* ``SearchTool``    – generic base class; subclass and override ``_fetch``
                       to adapt to any other provider.
* ``StubSearchTool``– deterministic fake results for offline dev / testing.
"""

from __future__ import annotations

import os
from typing import Any


class ExaSearchTool:
    """Neural search powered by the Exa API (https://exa.ai).

    Exa is purpose-built for AI agents: it returns semantically relevant
    results with pre-extracted text highlights rather than raw HTML snippets,
    which gives the ResearcherAgent richer context for summarisation.

    Environment variable: ``EXA_API_KEY``

    Args:
        api_key:     Exa API key.  Falls back to ``EXA_API_KEY`` env var.
        max_results: Maximum number of results to return per query (default 5).
        num_sentences: Sentences per highlight excerpt (default 3).
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        num_sentences: int = 3,
    ):
        from exa_py import Exa  # imported lazily so stub mode never needs the package

        self._api_key = api_key or os.environ.get("EXA_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "Exa API key is required. Set EXA_API_KEY or pass api_key=..."
            )
        self._client = Exa(api_key=self._api_key)
        self.max_results = max_results
        self.num_sentences = num_sentences

    def __call__(self, query: str) -> str:
        """Run *query* and return a plain-text result block."""
        results = self._fetch(query)
        if not results:
            return "No results found."
        lines: list[str] = []
        for r in results:
            lines.append(f"[{r.get('title', 'No title')}] {r.get('snippet', '')}")
            lines.append(f"URL: {r.get('url', 'N/A')}")
            lines.append("")
        return "\n".join(lines).strip()

    def _fetch(self, query: str) -> list[dict[str, Any]]:
        """Call Exa's search_and_contents endpoint and normalise results."""
        response = self._client.search_and_contents(
            query,
            num_results=self.max_results,
            highlights={"num_sentences": self.num_sentences},
            use_autoprompt=True,
        )
        results: list[dict[str, Any]] = []
        for r in response.results:
            highlights = getattr(r, "highlights", None) or []
            snippet = " ".join(highlights) if highlights else ""
            results.append(
                {"title": r.title or "", "snippet": snippet, "url": r.url or ""}
            )
        return results


class SearchTool:
    """Generic base class for search tools.

    Subclass and override ``_fetch`` to integrate a provider other than Exa.
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
        """Call the external search API and return normalised result dicts.

        Each dict must contain at least 'title', 'snippet', and 'url'.
        """
        raise NotImplementedError(
            "SearchTool._fetch() must be implemented. "
            "Use ExaSearchTool for the Exa provider or StubSearchTool for offline use."
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
