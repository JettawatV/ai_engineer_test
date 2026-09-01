from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict
from .models import GeneratedReport, RetrievalResult, UserQuery, WorkflowResult


class WorkflowState(TypedDict, total=False):
    query: str
    retrieval: RetrievalResult
    report: GeneratedReport


def build_workflow(data_retriever, report_generator) -> CompiledStateGraph:
    """Compile the sequential retrieval and report-generation workflow."""

    def retrieve(state: WorkflowState) -> dict[str, RetrievalResult]:
        query = UserQuery(query=state["query"])
        return {"retrieval": data_retriever.retrieve(query)}

    def write_report(state: WorkflowState) -> dict[str, GeneratedReport]:
        query = UserQuery(query=state["query"])
        return {"report": report_generator.generate(query, state["retrieval"])}

    graph = StateGraph(WorkflowState)
    graph.add_node("data_retriever_agent", retrieve)
    graph.add_node("report_generator_agent", write_report)
    graph.add_edge(START, "data_retriever_agent")
    graph.add_edge("data_retriever_agent", "report_generator_agent")
    graph.add_edge("report_generator_agent", END)
    return graph.compile()


def invoke_workflow(graph: CompiledStateGraph, query: str) -> WorkflowResult:
    """Run a validated query and validate the graph's public output."""

    validated_query = UserQuery(query=query)
    result = graph.invoke({"query": validated_query.query})
    return WorkflowResult.model_validate(result)
