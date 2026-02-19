"""Entry point for the Orchestrated Agent Hierarchy research system.

Usage
-----
    # With real API keys (Exa for search, OpenAI for LLM):
    export OPENAI_API_KEY=sk-...
    export EXA_API_KEY=...
    python main.py "What are the main causes and consequences of ocean acidification?"

    # Offline / stub mode (no API keys required):
    python main.py --stub "What are the main causes of ocean acidification?"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

from agents import OrchestratorAgent
from tools import ExaSearchTool, StubSearchTool


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Research Decomposition via Orchestrated Agent Hierarchy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "question",
        help="The broad research question to investigate.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        default=False,
        help=(
            "Use the StubSearchTool instead of a real search API. "
            "Useful for offline testing."
        ),
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name to use for all agents (default: gpt-4o-mini).",
    )
    parser.add_argument(
        "--max-sub-questions",
        type=int,
        default=6,
        dest="max_sub_questions",
        help="Maximum number of sub-questions the planner may generate (default: 6).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    return parser


def main() -> None:
    load_dotenv()

    parser = build_arg_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Validate environment when using real search
    if not args.stub:
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("Error: OPENAI_API_KEY is not set. Use --stub for offline mode.")
        if not os.environ.get("EXA_API_KEY"):
            sys.exit(
                "Error: EXA_API_KEY is not set. "
                "Use --stub for offline mode or set the environment variable."
            )

    search_fn = StubSearchTool() if args.stub else ExaSearchTool()

    orchestrator = OrchestratorAgent(
        search_fn=search_fn,
        model=args.model,
        max_sub_questions=args.max_sub_questions,
    )

    print(f"\nResearch question: {args.question}\n")
    print("=" * 72)

    report = orchestrator.run(args.question)

    print(report)
    print("=" * 72)


if __name__ == "__main__":
    main()
