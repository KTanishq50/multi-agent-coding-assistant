"""
Intent classifier.
Uses regex patterns first (fast, no LLM cost).
Falls back to LLM only when patterns are ambiguous (short queries, mixed signals).
This is how real systems do it — cheap classifier first, LLM only when needed.
"""
from __future__ import annotations
import re
from app.services.llm import call_llm_raw
from .state import AgentState

_WRITE_PATTERNS = [
    r"\bwrite\b", r"\bcreate\b", r"\bbuild\b", r"\bgenerate\b",
    r"\badd\b", r"\bimplement\b", r"\bmake\b", r"\bnew function\b",
    r"\bnew class\b", r"\bnew endpoint\b", r"\bcode for\b",
    r"\bdefine\b", r"\bset up\b",
]
_FIX_PATTERNS = [
    r"\bfix\b", r"\bbug\b", r"\berror\b", r"\bbroken\b", r"\bfailing\b",
    r"\bdoesn.t work\b", r"\bnot working\b", r"\bcrash\b", r"\bexception\b",
    r"\btraceback\b", r"\bdebug\b", r"\bwrong\b", r"\bissue\b",
    r"\brefactor\b", r"\bimprove\b", r"\boptimize\b", r"\bproblem\b",
]
_EXPLAIN_PATTERNS = [
    r"\bexplain\b", r"\bwhat does\b", r"\bhow does\b", r"\bwhat is\b",
    r"\bdescribe\b", r"\bunderstand\b", r"\bwalk me through\b",
    r"\bwhy does\b", r"\bwhere is\b", r"\bshow me\b", r"\btell me\b",
    r"\bwhat.s happening\b",
]


def _regex_classify(query: str) -> tuple[str, int]:
    """Returns (intent, match_count). Higher match_count = more confident."""
    q = query.lower()
    scores = {
        "write":   sum(1 for p in _WRITE_PATTERNS   if re.search(p, q)),
        "fix":     sum(1 for p in _FIX_PATTERNS     if re.search(p, q)),
        "explain": sum(1 for p in _EXPLAIN_PATTERNS if re.search(p, q)),
    }
    best = max(scores, key=scores.get)
    return (best if scores[best] > 0 else "general"), scores[best]


def _llm_classify(query: str) -> str:
    """LLM fallback for ambiguous queries."""
    prompt = (
        f"Classify this coding query into exactly one category.\n"
        f"Categories: write, fix, explain, general\n"
        f"Query: {query}\n"
        f"Answer with one word only:"
    )
    try:
        result = call_llm_raw(prompt, max_tokens=5).strip().lower()
        for cat in ("write", "fix", "explain", "general"):
            if cat in result:
                return cat
    except Exception:
        pass
    return "general"


def classify_intent(query: str) -> str:
    intent, confidence = _regex_classify(query)
    # if very short query or low confidence, use LLM
    if confidence == 0 or len(query.split()) <= 3:
        llm_intent = _llm_classify(query)
        print(f"[INTENT] Regex low confidence → LLM says: {llm_intent!r}")
        return llm_intent
    return intent

from langsmith import traceable
@traceable(name="intent_agent")
def intent_node(state: AgentState) -> AgentState:
    intent = classify_intent(state["query"])
    state["intent"] = intent
    print(f"[INTENT] {state['query']!r} → {intent}")

    # add to message bus
    messages = list(state.get("messages") or [])
    messages.append({
        "agent": "intent_classifier",
        "type": "decision",
        "content": f"Query intent classified as: {intent}",
        "metadata": {"intent": intent},
    })
    state["messages"] = messages
    return state