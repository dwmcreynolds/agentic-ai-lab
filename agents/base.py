"""Base agent class shared by all agents in the hierarchy."""

from __future__ import annotations

import json
import os
from typing import Any

from openai import OpenAI


class BaseAgent:
    """Common interface and helpers for every agent in the hierarchy.

    Subclasses set ``system_prompt`` and optionally ``tools`` (OpenAI
    function-calling schema).  Calling ``run()`` sends a user message,
    handles one round of tool calls, and returns the assistant's final
    text response.
    """

    system_prompt: str = ""
    tools: list[dict[str, Any]] = []

    def __init__(self, client: OpenAI | None = None, model: str = "gpt-4o-mini"):
        self.client = client or OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))
        self.model = model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _chat(self, messages: list[dict[str, Any]]) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if self.tools:
            kwargs["tools"] = self.tools
            kwargs["tool_choice"] = "auto"
        return self.client.chat.completions.create(**kwargs)

    def _handle_tool_calls(
        self,
        messages: list[dict[str, Any]],
        response: Any,
        tool_registry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Append tool-call results to the message list and return it."""
        choice = response.choices[0]
        messages.append(choice.message.model_dump(exclude_unset=True))

        for tool_call in choice.message.tool_calls or []:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            fn = tool_registry.get(fn_name)
            if fn is None:
                result = f"Error: unknown tool '{fn_name}'"
            else:
                try:
                    result = fn(**fn_args)
                except Exception as exc:  # noqa: BLE001
                    result = f"Error: {exc}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                }
            )
        return messages

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(
        self,
        user_message: str,
        tool_registry: dict[str, Any] | None = None,
    ) -> str:
        """Send *user_message*, handle at most one round of tool calls, return text."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        response = self._chat(messages)
        choice = response.choices[0]

        if choice.finish_reason == "tool_calls" and tool_registry:
            messages = self._handle_tool_calls(messages, response, tool_registry)
            response = self._chat(messages)
            choice = response.choices[0]

        return choice.message.content or ""
