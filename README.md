# Research Decomposition Agent

## Overview
This project implements a lightweight agentic system that takes a broad research question,
breaks it into focused sub-questions, gathers information using search tools, and produces a
structured summary with citations.

The goal is not to maximize creativity, but to explore controllability, evaluation, and failure
modes in multi-step LLM workflows.

---

## Problem Statement
Single-shot prompts often produce shallow or uneven research summaries when given broad questions.
They may skip important subtopics, hallucinate sources, or fail to explain their reasoning process.

This project investigates whether a simple task-decomposition loop improves coverage and
reliability compared to a single prompt approach.

---

## Objectives
In scope:
- Decompose a research question into 3–6 focused sub-questions
- Perform tool-assisted research for each sub-question
- Generate a final summary with explicit source attribution
- Halt execution once coverage criteria are met

Out of scope:
- Real-time web crawling
- Long-term memory persistence
- Autonomous goal generation

---

## System Architecture
The system uses a single agent operating in a controlled loop.

Components:
- Planner module to generate sub-questions
- Tool interface for external search
- Aggregation step to combine findings
- Evaluation step to check completeness

The agent terminates when all sub-questions have been addressed or when a retry limit is reached.

---

## How It Works
1. User provides a research question
2. Agent generates a list of sub-questions
3. Each sub-question is processed sequentially
4. Search tool is called with constrained queries
5. Results are summarized and stored in short-term memory
6. Final synthesis is generated with citations

---

## Prompt Design
Prompts are split into separate roles:
- Planner prompt for question decomposition
- Research prompt for summarization
- Synthesizer prompt for final output

Strict instructions are used to:
- Avoid unsupported claims
- Require explicit citations
- Prevent the model from answering beyond retrieved content

Despite these constraints, hallucinated citations remain a known risk.

---

## Tools and Dependencies
- OpenAI API for language model access
- External search API for document retrieval
- Python standard library for orchestration

The agent only calls tools when instructed by the planner prompt.

---

## Setup and Usage
Requirements:
- Python 3.10+
- API keys stored as environment variables

Installation:
```bash
pip install -r requirements.txt
