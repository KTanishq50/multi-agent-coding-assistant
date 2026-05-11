from __future__ import annotations
from app.services.llm import call_llm_raw
from .state import AgentState, AgentMessage
from langsmith import traceable


@traceable(name="supervisor_agent", run_type="chain")
def supervisor_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "general")
    query  = state["query"]

    messages = state.get("messages") or []
    history  = "\n".join(
        f"[{m['agent']}] {m['content'][:120]}" for m in messages[-6:]
    )

    prompt = (
        f"You are a supervisor for a coding agent system.\n"
        f"User query: {query}\n"
        f"Intent: {intent}\n"
        f"Recent agent messages:\n{history}\n\n"
        f"Decide the next action. Reply with ONE of:\n"
        f"PROCEED — everything is on track, continue normally\n"
        f"NEEDS_MORE_CONTEXT — retrieval was poor, need broader search\n"
        f"SKIP_RETRIEVAL — this is a general coding question, no project context needed\n"
        f"Answer with one of those exact phrases:"
    )

    try:
        decision = call_llm_raw(prompt, max_tokens=10).strip().upper()
    except Exception:
        decision = "PROCEED"

    if "SKIP" in decision:
        route = "skip_retrieval"
    elif "MORE_CONTEXT" in decision or "CONTEXT" in decision:
        route = "needs_more_context"
    else:
        route = "proceed"

    state["route"]            = route
    state["supervisor_notes"] = decision
    print(f"[SUPERVISOR] Decision: {route} (raw: {decision!r})")

    messages = list(state.get("messages") or [])
    messages.append(AgentMessage(
        agent="supervisor",
        type="decision",
        content=f"Routing decision: {route}",
        metadata={"route": route, "raw": decision},
    ))
    state["messages"] = messages
    return state


@traceable(name="supervisor_retry_agent", run_type="chain")
def supervisor_retry_node(state: AgentState) -> AgentState:
    """
    Called after reviewer rejects.
    Reads reviewer feedback and gives coder specific guidance on retry.
    The guidance is injected into state so coder sees it in the next pass.
    """
    answer = state.get("answer", "")
    review = state.get("review", "")
    query  = state["query"]

    prompt = (
        f"A coding agent produced an answer that was rejected by the reviewer.\n"
        f"Query: {query}\n"
        f"Reviewer feedback: {review[:400]}\n\n"
        f"First line: RETRY or ACCEPT\n"
        f"If RETRY: on the next lines, give the coder 2-3 specific instructions to fix the answer.\n"
        f"Be concrete — reference what was wrong and what to do instead."
    )

    try:
        decision = call_llm_raw(prompt, max_tokens=120).strip()
    except Exception:
        decision = "ACCEPT"

    decision_upper = decision.upper()

    if "RETRY" in decision_upper and state.get("retry_count", 0) < 1:
        state["needs_retry"]  = True
        state["retry_count"]  = state.get("retry_count", 0) + 1
        # extract the guidance lines and store as supervisor note for coder
        lines    = decision.strip().split("\n")
        guidance = "\n".join(l for l in lines[1:] if l.strip())
        state["supervisor_notes"] = guidance if guidance else decision
        print(f"[SUPERVISOR RETRY] RETRY — guidance: {guidance[:80]}")
    else:
        state["needs_retry"] = False
        if not state.get("final_answer"):
            state["final_answer"] = answer
        print(f"[SUPERVISOR RETRY] ACCEPT")

    print(f"[SUPERVISOR RETRY] Decision: {decision[:60]!r} → needs_retry={state['needs_retry']}")

    messages = list(state.get("messages") or [])
    messages.append(AgentMessage(
        agent="supervisor",
        type="decision",
        content=f"Retry decision: {'retry' if state['needs_retry'] else 'accept'}. {state.get('supervisor_notes', '')[:150]}",
        metadata={"decision": decision[:100], "retry_count": state.get("retry_count", 0)},
    ))
    state["messages"] = messages
    return state