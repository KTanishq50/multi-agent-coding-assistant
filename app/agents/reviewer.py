from __future__ import annotations
import re
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm import call_llm
from .state import AgentState, AgentMessage

_REVIEW_CRITERIA = {
    "write": (
        "Evaluate the generated code on these dimensions:\n"
        "1. Correctness — does it actually implement what was requested?\n"
        "2. Completeness — are there missing cases, imports, or error handling?\n"
        "3. Style — does it follow patterns from the codebase context?\n"
        "4. Quality — is it clean, readable, properly documented?\n"
        "If all pass: APPROVED. If any fail: REJECTED with specific fixes."
    ),
    "fix": (
        "Evaluate the fix on these dimensions:\n"
        "1. Does it actually address the described bug?\n"
        "2. Does it show the corrected code clearly?\n"
        "3. Could the fix introduce new bugs or regressions?\n"
        "4. Is the explanation of what was wrong accurate?\n"
        "If all pass: APPROVED. If any fail: REJECTED with specific fixes."
    ),
    "explain": (
        "Evaluate the explanation on these dimensions:\n"
        "1. Is it grounded in actual code from the context (not generic)?\n"
        "2. Does it reference real function/class names?\n"
        "3. Is it accurate — no hallucinated behavior?\n"
        "4. Is it clear and well structured?\n"
        "If all pass: APPROVED. If any fail: REJECTED with specific fixes."
    ),
    "general": (
        "Evaluate the answer on these dimensions:\n"
        "1. Does it answer the actual question?\n"
        "2. Is it grounded in the codebase context?\n"
        "3. Are there any hallucinations or invented details?\n"
        "4. Is it specific and actionable?\n"
        "If all pass: APPROVED. If any fail: REJECTED with specific fixes."
    ),
}

_NOISE_MARKERS = [
    "answer to review:", "codebase context:", "query:", "review criterion:",
    "agent context:", "if the answer", "does it write", "does it fix",
    "first line:", "then write", "evaluate the", "on these dimensions",
]


def _clean_reviewer_body(text: str) -> str:
    """
    Extract improved answer from reviewer output after APPROVED/REJECTED line.
    Stops at first line that echoes prompt content.
    """
    lines = text.strip().split("\n")
    body_lines = lines[1:] if lines else []

    clean = []
    for line in body_lines:
        line_lower = line.strip().lower()
        if any(marker in line_lower for marker in _NOISE_MARKERS):
            break
        clean.append(line)

    return "\n".join(clean).strip()

from langsmith import traceable
@traceable(name="reviewer_agent")
def reviewer_node(state: AgentState) -> AgentState:
    answer = state.get("answer", "")

    if not answer or not answer.strip():
        state["needs_retry"] = False
        state["final_answer"] = "No output produced. Try rephrasing your prompt."
        state["review"] = "SKIPPED — empty answer"
        return state

    # after one retry accept what we have — don't loop forever
    if state.get("retry_count", 0) >= 1:
        state["needs_retry"] = False
        state["final_answer"] = answer
        state["review"] = "Accepted after retry."
        return state

    intent = state.get("intent", "general")
    criteria = _REVIEW_CRITERIA.get(intent, _REVIEW_CRITERIA["general"])

    context = state.get("context") or {}
    chunks = context.get("code", []) if isinstance(context, dict) else []
    context_str = "\n\n---\n\n".join(chunks[:2])[:600]

    # read agent communications — reviewer should know what others decided
    messages = state.get("messages") or []
    agent_notes = "\n".join(
        f"[{m['agent']}/{m['type']}] {m['content'][:120]}"
        for m in messages[-6:]
        if m["agent"] in ("planner", "retriever", "rag_grader", "verifier", "supervisor")
    )

    syntax_ok = state.get("syntax_ok", True)
    syntax_note = ""
    if syntax_ok is False:
        syntax_note = f"\nNote: syntax verifier found errors — check code carefully.\n"

    retrieval_ok = state.get("retrieval_ok", True)
    retrieval_note = ""
    if not retrieval_ok:
        conf = state.get("retrieval_confidence", 0)
        retrieval_note = f"\nNote: retrieval confidence was low ({conf:.2f}) — watch for hallucinated details.\n"

    prompt = [
        SystemMessage(content=(
            f"You are a senior code reviewer performing a thorough review.\n\n"
            f"Review criteria for '{intent}' task:\n{criteria}\n"
            f"{syntax_note}{retrieval_note}\n"
            f"Agent pipeline context:\n{agent_notes}\n\n"
            f"Output format (strict):\n"
            f"Line 1: APPROVED or REJECTED\n"
            f"If REJECTED: explain each specific issue, then provide the corrected version.\n"
            f"If APPROVED: provide the final clean version of the answer."
        )),
        HumanMessage(content=(
            f"Query: {state['query']}\n\n"
            f"Codebase context:\n{context_str}\n\n"
            f"Answer to review:\n{answer}"
        ))
    ]

    review = call_llm(prompt)

    # store only the first 300 chars of review as the review record
    # this is the verdict + brief notes, not the full echoed answer
    state["review"] = review

    first_line = review.strip().split("\n")[0].strip().upper()
    state["needs_retry"] = "REJECTED" in first_line

    if state["needs_retry"]:
        # reviewer rejected — try to extract the corrected version it wrote
        body = _clean_reviewer_body(review)
        # use reviewer's correction if substantial, otherwise send back to coder
        state["final_answer"] = body if len(body) > 30 else answer
    else:
        # approved — use coder's answer directly
        # reviewer's rewrite of an approved answer is usually worse with weak models
        state["final_answer"] = answer

    messages = list(state.get("messages") or [])
    messages.append(AgentMessage(
        agent="reviewer",
        type="review",
        content=f"{'REJECTED' if state['needs_retry'] else 'APPROVED'}: {review[:300]}",
        metadata={
            "needs_retry": state["needs_retry"],
            "intent": intent,
            "syntax_ok": syntax_ok,
            "retrieval_ok": retrieval_ok,
        },
    ))
    state["messages"] = messages

    print(f"[REVIEWER] {'REJECTED → retry' if state['needs_retry'] else 'APPROVED'}")
    return state