"""Synthesizer agent – combines researcher findings into a final report."""

from __future__ import annotations

from typing import Any

from agents.base import BaseAgent


class SynthesizerAgent(BaseAgent):
    """Level-1 agent that aggregates all research findings.

    Receives the original research question plus a list of per-sub-question
    findings and produces a coherent, fully-cited final report.
    """

    system_prompt = (
        "You are an expert technical writer specialising in research synthesis. "
        "You will receive a research question and a collection of findings, each "
        "with a summary and a list of source URLs. "
        "Produce a structured final report with:\n"
        "  1. An executive summary (3–5 sentences)\n"
        "  2. A section for each sub-question with key insights\n"
        "  3. A numbered references section listing every unique source URL\n"
        "Do NOT introduce facts, claims, or citations that are not present in "
        "the provided findings. Clearly cite in-text references by number."
    )

    def synthesize(
        self,
        research_question: str,
        findings: list[dict[str, Any]],
    ) -> str:
        """Generate a final report from *findings* for *research_question*."""
        findings_text = "\n\n".join(
            f"Sub-question: {f['sub_question']}\n"
            f"Summary: {f['summary']}\n"
            f"Sources: {', '.join(f['sources']) if f['sources'] else 'none'}"
            for f in findings
        )
        prompt = (
            f"Research question: {research_question}\n\n"
            f"Findings:\n{findings_text}"
        )
        return self.run(prompt)
