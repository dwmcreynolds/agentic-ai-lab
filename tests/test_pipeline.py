"""Tests for the Orchestrated Agent Hierarchy pipeline.

All OpenAI calls are mocked so the suite runs with no API keys.
The StubSearchTool is used for every search interaction.
"""

from __future__ import annotations

import json
import types
import unittest
from unittest.mock import MagicMock, patch

from memory.store import MemoryStore
from tools.search import StubSearchTool
from agents.planner import PlannerAgent
from agents.researcher import ResearcherAgent
from agents.synthesizer import SynthesizerAgent
from agents.orchestrator import OrchestratorAgent


# ---------------------------------------------------------------------------
# Helpers – build fake OpenAI response objects
# ---------------------------------------------------------------------------

def _make_text_response(content: str) -> MagicMock:
    """Return a minimal fake ChatCompletion whose first choice is a text reply."""
    msg = MagicMock()
    msg.content = content
    msg.tool_calls = None
    msg.model_dump.return_value = {"role": "assistant", "content": content}

    choice = MagicMock()
    choice.finish_reason = "stop"
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_tool_call_response(tool_name: str, tool_args: dict, call_id: str = "call_1") -> MagicMock:
    """Return a fake ChatCompletion that requests one tool call."""
    tool_call = MagicMock()
    tool_call.id = call_id
    tool_call.function.name = tool_name
    tool_call.function.arguments = json.dumps(tool_args)

    msg = MagicMock()
    msg.content = None
    msg.tool_calls = [tool_call]
    msg.model_dump.return_value = {
        "role": "assistant",
        "tool_calls": [{"id": call_id, "function": {"name": tool_name, "arguments": json.dumps(tool_args)}}],
    }

    choice = MagicMock()
    choice.finish_reason = "tool_calls"
    choice.message = msg

    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ---------------------------------------------------------------------------
# MemoryStore
# ---------------------------------------------------------------------------

class TestMemoryStore(unittest.TestCase):
    def setUp(self):
        self.mem = MemoryStore()

    def test_store_and_retrieve(self):
        self.mem.store("key", [1, 2, 3])
        self.assertEqual(self.mem.retrieve("key"), [1, 2, 3])

    def test_retrieve_missing_returns_default(self):
        self.assertIsNone(self.mem.retrieve("nope"))
        self.assertEqual(self.mem.retrieve("nope", "fallback"), "fallback")

    def test_clear(self):
        self.mem.store("k", "v")
        self.mem.clear()
        self.assertEqual(len(self.mem), 0)

    def test_contains(self):
        self.mem.store("x", 1)
        self.assertIn("x", self.mem)
        self.assertNotIn("y", self.mem)

    def test_snapshot_is_copy(self):
        self.mem.store("a", 1)
        snap = self.mem.snapshot()
        snap["a"] = 99
        self.assertEqual(self.mem.retrieve("a"), 1)  # original unchanged


# ---------------------------------------------------------------------------
# StubSearchTool
# ---------------------------------------------------------------------------

class TestStubSearchTool(unittest.TestCase):
    def setUp(self):
        self.tool = StubSearchTool()

    def test_returns_string(self):
        result = self.tool("climate change")
        self.assertIsInstance(result, str)

    def test_includes_query(self):
        result = self.tool("ocean acidification")
        self.assertIn("ocean acidification", result)

    def test_includes_stub_urls(self):
        result = self.tool("anything")
        self.assertIn("https://example.com", result)


# ---------------------------------------------------------------------------
# PlannerAgent
# ---------------------------------------------------------------------------

class TestPlannerAgent(unittest.TestCase):
    def _make_planner(self, llm_response: str) -> PlannerAgent:
        client = MagicMock()
        client.chat.completions.create.return_value = _make_text_response(llm_response)
        return PlannerAgent(client=client)

    def test_decompose_json_array(self):
        planner = self._make_planner('["Q1", "Q2", "Q3"]')
        result = planner.decompose("Tell me about climate change")
        self.assertEqual(result, ["Q1", "Q2", "Q3"])

    def test_decompose_json_with_fences(self):
        planner = self._make_planner('```json\n["A", "B"]\n```')
        result = planner.decompose("Some question")
        self.assertEqual(result, ["A", "B"])

    def test_decompose_fallback_lines(self):
        # If the model doesn't return valid JSON, fall back to line splitting
        planner = self._make_planner("- Sub Q1\n- Sub Q2\n- Sub Q3")
        result = planner.decompose("Some question")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_decompose_calls_llm_once(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_text_response('["X"]')
        planner = PlannerAgent(client=client)
        planner.decompose("question")
        self.assertEqual(client.chat.completions.create.call_count, 1)


# ---------------------------------------------------------------------------
# ResearcherAgent
# ---------------------------------------------------------------------------

class TestResearcherAgent(unittest.TestCase):
    def _make_researcher(self, llm_responses: list) -> tuple[ResearcherAgent, MagicMock]:
        client = MagicMock()
        client.chat.completions.create.side_effect = llm_responses
        search = StubSearchTool()
        researcher = ResearcherAgent(search_fn=search, client=client)
        return researcher, client

    def test_research_plain_text(self):
        resp = _make_text_response(
            "SUMMARY: Ocean acidification is caused by CO2.\nSOURCES: https://example.com/overview"
        )
        researcher, _ = self._make_researcher([resp])
        finding = researcher.research("What causes ocean acidification?")
        self.assertEqual(finding["sub_question"], "What causes ocean acidification?")
        self.assertIn("CO2", finding["summary"])
        self.assertIn("https://example.com/overview", finding["sources"])

    def test_research_with_tool_call(self):
        # First response: requests search tool; second: text answer
        tool_resp = _make_tool_call_response("search", {"query": "ocean acidification causes"})
        text_resp = _make_text_response(
            "SUMMARY: Dissolved CO2 lowers ocean pH.\nSOURCES: https://example.com/overview"
        )
        researcher, client = self._make_researcher([tool_resp, text_resp])
        finding = researcher.research("What causes ocean acidification?")
        # Two LLM calls: tool call + follow-up
        self.assertEqual(client.chat.completions.create.call_count, 2)
        self.assertIn("pH", finding["summary"])

    def test_research_returns_dict_keys(self):
        resp = _make_text_response("SUMMARY: Some finding.\nSOURCES: https://a.com")
        researcher, _ = self._make_researcher([resp])
        finding = researcher.research("A question")
        self.assertIn("sub_question", finding)
        self.assertIn("summary", finding)
        self.assertIn("sources", finding)


# ---------------------------------------------------------------------------
# SynthesizerAgent
# ---------------------------------------------------------------------------

class TestSynthesizerAgent(unittest.TestCase):
    def _make_synthesizer(self, llm_response: str) -> SynthesizerAgent:
        client = MagicMock()
        client.chat.completions.create.return_value = _make_text_response(llm_response)
        return SynthesizerAgent(client=client)

    def test_synthesize_returns_string(self):
        synth = self._make_synthesizer("# Final Report\n\nExecutive summary here.")
        findings = [
            {"sub_question": "Q1", "summary": "Finding 1", "sources": ["https://example.com"]},
            {"sub_question": "Q2", "summary": "Finding 2", "sources": []},
        ]
        report = synth.synthesize("Big research question", findings)
        self.assertIsInstance(report, str)
        self.assertIn("Final Report", report)

    def test_synthesize_includes_findings_in_prompt(self):
        client = MagicMock()
        client.chat.completions.create.return_value = _make_text_response("report")
        synth = SynthesizerAgent(client=client)
        findings = [{"sub_question": "SQ", "summary": "S", "sources": ["https://x.com"]}]
        synth.synthesize("Question", findings)

        call_args = client.chat.completions.create.call_args
        messages = call_args.kwargs.get("messages") or call_args.args[0]
        user_content = next(m["content"] for m in messages if m["role"] == "user")
        self.assertIn("SQ", user_content)
        self.assertIn("https://x.com", user_content)


# ---------------------------------------------------------------------------
# OrchestratorAgent (integration)
# ---------------------------------------------------------------------------

class TestOrchestratorAgent(unittest.TestCase):
    """End-to-end pipeline test with all LLM calls mocked."""

    def _make_orchestrator(self) -> OrchestratorAgent:
        """Build an Orchestrator whose sub-agents all use mocked OpenAI clients."""
        search = StubSearchTool()
        orchestrator = OrchestratorAgent(search_fn=search)

        # Patch the internal sub-agents with lightweight mocks
        orchestrator._planner = MagicMock()
        orchestrator._planner.decompose.return_value = [
            "What are the causes?",
            "What are the effects?",
            "What are the solutions?",
        ]

        orchestrator._synthesizer = MagicMock()
        orchestrator._synthesizer.synthesize.return_value = (
            "# Research Report\n\n## Executive Summary\nThis is a synthesized report."
        )

        # ResearcherAgent is instantiated per-sub-question; patch the class
        self._researcher_patch = patch("agents.orchestrator.ResearcherAgent")
        MockResearcher = self._researcher_patch.start()
        instance = MockResearcher.return_value
        instance.research.side_effect = lambda q: {
            "sub_question": q,
            "summary": f"Finding for: {q}",
            "sources": ["https://example.com/overview"],
        }

        return orchestrator

    def tearDown(self):
        try:
            self._researcher_patch.stop()
        except Exception:
            pass

    def test_full_pipeline_returns_report(self):
        orch = self._make_orchestrator()
        report = orch.run("What is ocean acidification?")
        self.assertIsInstance(report, str)
        self.assertIn("Research Report", report)

    def test_planner_called_once(self):
        orch = self._make_orchestrator()
        orch.run("Some question")
        orch._planner.decompose.assert_called_once_with("Some question")

    def test_synthesizer_called_once_with_all_findings(self):
        orch = self._make_orchestrator()
        orch.run("Some question")
        orch._synthesizer.synthesize.assert_called_once()
        _, findings = orch._synthesizer.synthesize.call_args.args
        self.assertEqual(len(findings), 3)

    def test_memory_populated_after_run(self):
        orch = self._make_orchestrator()
        orch.run("Some question")
        mem = orch.memory
        self.assertIn("sub_questions", mem)
        self.assertIn("findings", mem)
        self.assertIn("report", mem)
        self.assertEqual(len(mem.retrieve("sub_questions")), 3)

    def test_max_sub_questions_respected(self):
        search = StubSearchTool()
        orch = OrchestratorAgent(search_fn=search, max_sub_questions=2)
        orch._planner = MagicMock()
        orch._planner.decompose.return_value = ["Q1", "Q2", "Q3", "Q4", "Q5"]
        orch._synthesizer = MagicMock()
        orch._synthesizer.synthesize.return_value = "report"

        with patch("agents.orchestrator.ResearcherAgent") as MockR:
            MockR.return_value.research.side_effect = lambda q: {
                "sub_question": q, "summary": "s", "sources": []
            }
            orch.run("question")

        findings = orch.memory.retrieve("findings")
        self.assertEqual(len(findings), 2)  # capped at max_sub_questions


if __name__ == "__main__":
    unittest.main(verbosity=2)
