import json
from pathlib import Path
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, StructuredTool
from pydantic import ValidationError
from .models import GeneratedReport, QueryText, RetrievalResult, UserQuery
from .retrieval import SemanticRetriever, load_knowledge_chunks

RETRIEVER_PROMPT = """You are a data-retrieval agent.
Call search_knowledge_base exactly once using the user's complete question.
Do not answer, interpret, summarize, or invent facts."""

REPORT_PROMPT = """You are a report-generation agent.
Answer only from retrieved_evidence. Produce a comprehensive, cohesive, non-redundant,
well-formatted answer for the end user. Return the required structured output.
List only chunk IDs that directly support the answer. If the evidence is insufficient,
say so, set insufficient_context to true, and return no used_chunk_ids."""


class AgentOutputError(RuntimeError):
    pass


def build_search_tool(
    knowledge_base_path: Path,
    embeddings: Embeddings,
    *,
    min_score: float,
) -> BaseTool:
    """Load the knowledge file and expose search as the agent's only approved tool."""

    retriever = SemanticRetriever(load_knowledge_chunks(knowledge_base_path), embeddings)

    def search(query: QueryText) -> dict:
        result = retriever.search(query, min_score=min_score)
        return result.model_dump(mode="json")

    return StructuredTool.from_function(
        func=search,
        name="search_knowledge_base",
        description=(
            "Semantically search the local knowledge base using the user's complete question. "
            "Returns raw relevant chunks and never answers the question."
        ),
    )


class DataRetrieverAgent:
    """Force an LLM to retrieve evidence without producing an answer."""

    def __init__(self, model: BaseChatModel, search_tool: BaseTool) -> None:
        self._search_tool = search_tool
        self._model = model.bind_tools(
            [search_tool],
            tool_choice=search_tool.name,
            strict=True,
            parallel_tool_calls=False,
        )

    def retrieve(self, user_query: UserQuery) -> RetrievalResult:
        """Run exactly one approved search and validate its raw result."""
        response = self._model.invoke(
            [
                SystemMessage(content=RETRIEVER_PROMPT),
                HumanMessage(content=user_query.query),
            ]
        )
        tool_calls = response.tool_calls
        if len(tool_calls) != 1:
            raise AgentOutputError("Data retriever must make exactly one tool call")

        tool_call = tool_calls[0]
        if tool_call["name"] != self._search_tool.name:
            raise AgentOutputError("Data retriever attempted to call an unapproved tool")

        try:
            return RetrievalResult.model_validate(self._search_tool.invoke(tool_call["args"]))
        except ValidationError as exc:
            raise AgentOutputError("Data retriever returned an invalid retrieval result") from exc


def validate_report_sources(report: GeneratedReport, retrieval: RetrievalResult) -> None:
    """Reject reports that cite chunks absent from the retrieval result."""
    unknown_ids = set(report.used_chunk_ids) - {chunk.chunk_id for chunk in retrieval.chunks}
    if unknown_ids:
        unknown = ", ".join(sorted(unknown_ids))
        raise AgentOutputError(f"Report generator cited unknown chunk IDs: {unknown}")


class ReportGeneratorAgent:
    """Generate a structured answer using only validated retrieved evidence."""

    def __init__(self, model: BaseChatModel) -> None:
        self._model = model.with_structured_output(
            GeneratedReport,
            method="json_schema",
            strict=True,
        )

    def generate(self, user_query: UserQuery, retrieval: RetrievalResult) -> GeneratedReport:
        """Create a grounded report and verify every claimed source ID."""
        payload = {
            "user_query": user_query.query,
            "retrieved_evidence": retrieval.model_dump(mode="json"),
        }
        raw_report = self._model.invoke(
            [
                SystemMessage(content=REPORT_PROMPT),
                HumanMessage(content=json.dumps(payload, ensure_ascii=False, indent=2)),
            ]
        )
        try:
            report = GeneratedReport.model_validate(raw_report)
        except ValidationError as exc:
            raise AgentOutputError("Report generator returned an invalid report") from exc

        validate_report_sources(report, retrieval)
        return report
