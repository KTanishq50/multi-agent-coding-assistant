from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from .state import AgentState
from .intent_classifier import intent_node
from .supervisor import supervisor_node, supervisor_retry_node
from .planner import planner_node
from .retriever import retriever_node
from .rag_grader import rag_grader_node, corrective_rag_node
from .coder import coder_node
from .verifier import verifier_node
from .reviewer import reviewer_node
from app.services.learning import process_learning

MAX_SYNTAX_RETRIES = 1


def route_after_supervisor(state: AgentState) -> str:
    return "skip" if state.get("route") == "skip_retrieval" else "retrieve"


def route_after_grader(state: AgentState) -> str:
    if not state.get("retrieval_ok", True) and not state.get("corrective_attempted"):
        return "corrective"
    return "coder"


def route_after_verifier(state: AgentState) -> str:
    if not state.get("syntax_ok", True):
        syntax_retries = state.get("syntax_retry_count", 0)
        if syntax_retries < MAX_SYNTAX_RETRIES:
            return "fix_syntax"
        else:
            print("[GRAPH] Syntax still failing after retry — proceeding to reviewer")
            return "review"
    return "review"


def route_after_review(state: AgentState) -> str:
    return "retry" if state.get("needs_retry") else "learn"


def route_after_retry_decision(state: AgentState) -> str:
    return "coder" if state.get("needs_retry") else "learn"


def increment_syntax_retry(state: AgentState) -> AgentState:
    state["syntax_retry_count"] = state.get("syntax_retry_count", 0) + 1
    state["syntax_ok"]    = None
    state["syntax_error"] = None
    state["coder_iterations"] = 0
    print(f"[GRAPH] Syntax retry #{state['syntax_retry_count']}")
    return state


@traceable(name="learning_node", run_type="chain")
def learning_node(state: AgentState) -> AgentState:
    process_learning(
        state.get("review", ""),
        answer=state.get("answer", ""),
        intent=state.get("intent", ""),
        query=state.get("query", ""),
    )
    return state


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("intent",                 intent_node)
    workflow.add_node("supervisor",             supervisor_node)
    workflow.add_node("planner",                planner_node)
    workflow.add_node("retriever",              retriever_node)
    workflow.add_node("rag_grader",             rag_grader_node)
    workflow.add_node("corrective_rag",         corrective_rag_node)
    workflow.add_node("coder",                  coder_node)
    workflow.add_node("verifier",               verifier_node)
    workflow.add_node("increment_syntax_retry", increment_syntax_retry)
    workflow.add_node("reviewer",               reviewer_node)
    workflow.add_node("supervisor_retry",       supervisor_retry_node)
    workflow.add_node("learn",                  learning_node)

    workflow.add_edge(START,    "intent")
    workflow.add_edge("intent", "supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {"skip": "planner", "retrieve": "planner"}
    )

    workflow.add_edge("planner",   "retriever")
    workflow.add_edge("retriever", "rag_grader")

    workflow.add_conditional_edges(
        "rag_grader",
        route_after_grader,
        {"corrective": "corrective_rag", "coder": "coder"}
    )

    workflow.add_edge("corrective_rag", "coder")
    workflow.add_edge("coder",          "verifier")

    workflow.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {"fix_syntax": "increment_syntax_retry", "review": "reviewer"}
    )

    workflow.add_edge("increment_syntax_retry", "coder")

    workflow.add_conditional_edges(
        "reviewer",
        route_after_review,
        {"retry": "supervisor_retry", "learn": "learn"}
    )

    workflow.add_conditional_edges(
        "supervisor_retry",
        route_after_retry_decision,
        {"coder": "coder", "learn": "learn"}
    )

    workflow.add_edge("learn", END)

    return workflow.compile()


graph = build_graph()