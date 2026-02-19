# Research Decomposition Agent

## Overview
This project implements an **Orchestrated Agent Hierarchy** that takes a broad
research question, decomposes it into focused sub-questions, gathers information
using search tools, and produces a structured summary with citations.

The system is designed around controllability, evaluation, and failure-mode
analysis in multi-step LLM workflows.

---

## Problem Statement
Single-shot prompts often produce shallow or uneven research summaries when given
broad questions. They may skip important subtopics, hallucinate sources, or fail
to explain their reasoning process.

This project investigates whether a **multi-agent task-decomposition hierarchy**
improves coverage and reliability compared to a single-prompt approach.

---

## Objectives
In scope:
- Decompose a research question into 3–6 focused sub-questions
- Perform tool-assisted research for each sub-question via dedicated agents
- Generate a final summary with explicit source attribution
- Halt execution once all sub-questions have been addressed or a retry limit
  is reached

Out of scope:
- Real-time web crawling
- Long-term memory persistence
- Autonomous goal generation

---

## System Architecture

The system uses a **three-level orchestrated agent hierarchy**.

```
Level 0  ┌─────────────────────────────┐
         │      OrchestratorAgent      │  ← top-level coordinator
         └────────┬────────┬───────────┘
                  │        │
Level 1  ┌────────▼──┐  ┌──▼────────┐  ┌──────────────────┐
         │  Planner  │  │ Researcher│  │   Synthesizer    │
         │  Agent    │  │ Agent ×N  │  │   Agent          │
         └───────────┘  └───────────┘  └──────────────────┘
```

| Agent | Role |
|---|---|
| `OrchestratorAgent` | Receives the research question, manages the pipeline, owns the MemoryStore |
| `PlannerAgent` | Decomposes the broad question into 3–6 focused sub-questions |
| `ResearcherAgent` | One instance per sub-question; calls the search tool and summarises findings |
| `SynthesizerAgent` | Merges all findings into a structured report with citations |

### Supporting Components

| Module | Purpose |
|---|---|
| `memory/store.py` | In-process key-value store; decouples agents from each other |
| `tools/search.py` | `SearchTool` (real API) and `StubSearchTool` (offline/testing) |

---

## How It Works

1. **User** provides a research question via CLI
2. **Orchestrator** invokes the **Planner** to decompose it into sub-questions
3. **Orchestrator** dispatches one **Researcher** agent per sub-question
4. Each **Researcher** calls the search tool and returns a structured finding
5. **Orchestrator** stores every finding in the shared **MemoryStore**
6. **Orchestrator** invokes the **Synthesizer** with all findings
7. **Synthesizer** returns a final cited report, which the Orchestrator returns
   to the user

---

## Project Structure

```
agentic-ai-lab/
├── main.py                  # CLI entry point
├── requirements.txt
├── agents/
│   ├── __init__.py
│   ├── base.py              # BaseAgent with OpenAI call helpers
│   ├── orchestrator.py      # Level-0 coordinator
│   ├── planner.py           # Level-1: question decomposition
│   ├── researcher.py        # Level-1: single-sub-question research
│   └── synthesizer.py       # Level-1: report generation
├── tools/
│   ├── __init__.py
│   └── search.py            # SearchTool + StubSearchTool
└── memory/
    ├── __init__.py
    └── store.py             # Short-term MemoryStore
```

---

## Prompt Design

Each agent has a tightly-scoped system prompt:

| Agent | Prompt focus |
|---|---|
| Planner | Output a JSON array of sub-questions; no extra text |
| Researcher | Use the search tool; format output as SUMMARY / SOURCES |
| Synthesizer | Produce a structured report; only cite sources present in findings |

Constraints applied across all agents:
- No unsupported claims
- Explicit citations required
- Output beyond retrieved content is prohibited

Despite these constraints, hallucinated citations remain a known risk when using
language models.

---

## Tools and Dependencies

- `openai>=1.0.0` – language model access (function calling / tool use)
- `python-dotenv>=1.0.0` – environment variable management
- Standard library only for orchestration logic

The `StubSearchTool` allows the full pipeline to run offline without any search
API keys, which is useful during development and in CI environments.

---

## Setup and Usage

**Requirements:** Python 3.10+

```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and fill in your API keys:

```
OPENAI_API_KEY=sk-...
SEARCH_API_KEY=...
```

**Run with a real search API:**

```bash
python main.py "What are the main causes and consequences of ocean acidification?"
```

**Run in offline / stub mode (no API keys required):**

```bash
python main.py --stub "What are the main causes of ocean acidification?"
```

**Additional options:**

```
--model gpt-4o          Override the model (default: gpt-4o-mini)
--max-sub-questions 4   Limit sub-questions (default: 6)
--verbose               Enable debug logging
```

---

## Extending the System

- **Swap the search backend:** implement `SearchTool._fetch()` or pass any
  callable `(query: str) -> str` as `search_fn` to `OrchestratorAgent`.
- **Add parallel research:** replace the sequential loop in
  `OrchestratorAgent.run()` with `concurrent.futures.ThreadPoolExecutor`.
- **Plug in a different LLM:** change the `model` parameter or subclass
  `BaseAgent` to wrap a non-OpenAI client.
