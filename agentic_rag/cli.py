import argparse
import logging
from collections.abc import Sequence
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import ValidationError

from .agents import (
    AgentOutputError,
    DataRetrieverAgent,
    ReportGeneratorAgent,
    build_search_tool,
)
from .config import Settings
from .retrieval import KnowledgeBaseError
from .workflow import build_workflow, invoke_workflow

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ask grounded questions about a local knowledge base."
    )
    parser.add_argument("query", nargs="*", help="Question to ask")
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Display retrieved chunk IDs and semantic similarity scores",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Keep accepting questions until 'exit' or 'quit'",
    )
    parser.add_argument("--verbose", action="store_true", help="Show unexpected errors")
    return parser


def build_application(settings: Settings):
    """Wire production dependencies without making an OpenAI request."""

    api_key = settings.openai_api_key.get_secret_value()
    embeddings = OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=api_key,
    )
    search_tool = build_search_tool(
        settings.knowledge_base_path,
        embeddings,
        min_score=settings.retrieval_min_score,
    )
    model = ChatOpenAI(
        model=settings.openai_model,
        api_key=api_key,
        max_retries=2,
    )
    return build_workflow(
        DataRetrieverAgent(model, search_tool),
        ReportGeneratorAgent(model),
    )


def print_result(result, *, show_sources: bool) -> None:
    print("\nAnswer\n------")
    print(result.report.answer)

    if not show_sources:
        return

    print("\nRetrieved sources\n-----------------")
    if not result.retrieval.chunks:
        print("No relevant chunks found.")
    for chunk in result.retrieval.chunks:
        status = "used" if chunk.chunk_id in result.report.used_chunk_ids else "retrieved"
        print(f"- {chunk.chunk_id} | score={chunk.score:.4f} | {status}")


def ask_once(graph, query: str, *, show_sources: bool) -> None:
    print_result(invoke_workflow(graph, query), show_sources=show_sources)


def interactive_loop(graph, *, show_sources: bool) -> None:
    print("Interactive mode. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if query.casefold() in {"exit", "quit"}:
            return
        if not query:
            print("Please enter a non-empty question.")
            continue
        ask_once(graph, query, show_sources=show_sources)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and translate expected failures into stable exit codes."""

    parser = build_parser()
    args = parser.parse_args(argv)
    query = " ".join(args.query).strip()
    if not args.interactive and not query:
        parser.error("provide a question or use --interactive")

    try:
        graph = build_application(Settings())
        if query:
            ask_once(graph, query, show_sources=args.show_sources)
        if args.interactive:
            interactive_loop(graph, show_sources=args.show_sources)
        return 0
    except (AgentOutputError, KnowledgeBaseError, ValidationError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except Exception:
        if args.verbose:
            raise
        LOGGER.error("Unexpected application failure. Re-run with --verbose for details.")
        return 1
