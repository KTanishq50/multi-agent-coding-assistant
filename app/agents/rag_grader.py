from __future__ import annotations
import numpy as np
from .state import AgentState, AgentMessage
from app.services.embeddings import get_embedding
from app.services.rag import query_codebase
from app.services.memory import retrieve_memory, format_memory
from langsmith import traceable

LOW_CONFIDENCE = 0.22


def _cosine(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d > 0 else 0.0


def _score_context(query: str, chunks: list[str]) -> float:
    if not chunks:
        return 0.0
    q_emb = get_embedding(query)
    if not q_emb:
        return 0.0
    scores = []
    for chunk in chunks[:3]:
        c_emb = get_embedding(chunk[:400])
        if c_emb:
            scores.append(_cosine(q_emb, c_emb))
    return float(np.mean(scores)) if scores else 0.0


@traceable(name="rag_grader_agent", run_type="chain")
def rag_grader_node(state: AgentState) -> AgentState:
    context = state.get("context") or {}
    chunks  = context.get("code", []) if isinstance(context, dict) else []

    if state.get("route") == "skip_retrieval":
        state["retrieval_confidence"] = 1.0
        state["retrieval_ok"]         = True
        return state

    score = _score_context(state["query"], chunks)
    state["retrieval_confidence"] = round(score, 3)
    state["retrieval_ok"]         = score >= LOW_CONFIDENCE

    print(f"[RAG GRADER] confidence={score:.3f} ok={state['retrieval_ok']}")

    messages = list(state.get("messages") or [])
    messages.append(AgentMessage(
        agent="rag_grader",
        type="decision",
        content=(
            f"Retrieval confidence: {score:.3f} "
            f"({'OK' if state['retrieval_ok'] else 'LOW — corrective RAG needed'})"
        ),
        metadata={"score": score, "ok": state["retrieval_ok"]},
    ))
    state["messages"] = messages
    return state


@traceable(name="corrective_rag_agent", run_type="chain")
def corrective_rag_node(state: AgentState) -> AgentState:
    query      = state["query"]
    intent     = state.get("intent", "general")
    project_id = state["project_id"]

    intent_prefix = {
        "write":   "function class definition example similar",
        "fix":     "bug error exception handler fix",
        "explain": "implementation logic flow structure",
        "general": "code usage pattern",
    }
    prefix       = intent_prefix.get(intent, "")
    reformulated = f"{prefix} {query}".strip()
    print(f"[CORRECTIVE RAG] Reformulated: {reformulated!r}")

    new_chunks = query_codebase(reformulated, project_id, n_results=5)
    new_score  = _score_context(query, new_chunks) if new_chunks else 0.0
    print(f"[CORRECTIVE RAG] New confidence: {new_score:.3f}")

    if new_chunks:
        memories = retrieve_memory(query, k=2)
        state["context"] = {
            "code":   new_chunks,
            "memory": format_memory(memories),
        }
        state["retrieval_confidence"] = round(new_score, 3)
        state["retrieval_ok"]         = new_score >= LOW_CONFIDENCE

    state["corrective_attempted"] = True

    messages = list(state.get("messages") or [])
    messages.append(AgentMessage(
        agent="corrective_rag",
        type="context",
        content=f"Corrective retrieval: {len(new_chunks)} chunks, confidence={new_score:.3f}",
        metadata={"reformulated": reformulated, "score": new_score},
    ))
    state["messages"] = messages
    return state