"""Short-term in-process memory store used by the Orchestrator.

The store is intentionally simple: a key-value dict that lives for the
duration of a single pipeline run.  It provides a structured way for the
Orchestrator to pass intermediate results between sub-agents without
coupling them directly to each other.
"""

from __future__ import annotations

from typing import Any


class MemoryStore:
    """Lightweight key-value store for inter-agent data sharing.

    Usage
    -----
    >>> mem = MemoryStore()
    >>> mem.store("sub_questions", ["Q1", "Q2"])
    >>> mem.retrieve("sub_questions")
    ['Q1', 'Q2']
    >>> mem.retrieve("missing_key")   # returns None, never raises
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        """Persist *value* under *key*, overwriting any existing entry."""
        self._data[key] = value

    def retrieve(self, key: str, default: Any = None) -> Any:
        """Return the value stored under *key*, or *default* if absent."""
        return self._data.get(key, default)

    def clear(self) -> None:
        """Remove all stored entries (called at the start of each run)."""
        self._data.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a shallow copy of the current store contents."""
        return dict(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __contains__(self, key: object) -> bool:
        return key in self._data
