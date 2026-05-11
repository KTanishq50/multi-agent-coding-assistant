"""
Syntax verifier — runs between coder and reviewer.
Model-independent: uses ast.parse() for Python, basic checks for JS/TS.
If syntax fails, sends back to coder with the exact error.
No LLM call — fast and cheap.
"""
from __future__ import annotations
import ast
import re
from .state import AgentState, AgentMessage


def _extract_python_blocks(text: str) -> list[str]:
    """Extract Python code blocks from markdown or raw text."""
    # try markdown fences first
    fenced = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.DOTALL)
    if fenced:
        return fenced

    # if no fences, check if whole text looks like Python
    stripped = text.strip()
    if stripped.startswith(("def ", "class ", "import ", "from ", "#")):
        return [stripped]

    # try to find any code-looking block
    all_fenced = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
    return all_fenced if all_fenced else []


def _verify_python(code: str) -> tuple[bool, str]:
    """Returns (is_valid, error_message)."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)


def _verify_js_basic(code: str) -> tuple[bool, str]:
    """Basic JS/TS checks — bracket balance."""
    opens = code.count("{") + code.count("(") + code.count("[")
    closes = code.count("}") + code.count(")") + code.count("]")
    if abs(opens - closes) > 2:
        return False, f"Unbalanced brackets: {opens} opens vs {closes} closes"
    return True, ""

from langsmith import traceable
@traceable(name="verifier_agent")
def verifier_node(state: AgentState) -> AgentState:
    """
    Check syntax of generated code.
    Sets state["syntax_ok"] and state["syntax_error"].
    If syntax fails, marks for re-send to coder (not reviewer).
    """
    answer = state.get("answer", "")
    intent = state.get("intent", "general")

    # only verify for write/fix intents — no point checking explanation text
    if intent not in ("write", "fix"):
        state["syntax_ok"] = True
        state["syntax_error"] = ""
        return state

    if not answer or not answer.strip():
        state["syntax_ok"] = True  # nothing to verify
        state["syntax_error"] = ""
        return state

    # try to extract Python blocks
    python_blocks = _extract_python_blocks(answer)

    if not python_blocks:
        # no extractable code block — not a syntax error, just prose
        state["syntax_ok"] = True
        state["syntax_error"] = ""
        print("[VERIFIER] No code blocks found — skipping syntax check")
        return state

    # check each block
    errors = []
    for block in python_blocks:
        ok, err = _verify_python(block)
        if not ok:
            errors.append(err)

    if errors:
        error_str = "; ".join(errors)
        state["syntax_ok"] = False
        state["syntax_error"] = error_str
        print(f"[VERIFIER] SYNTAX FAIL: {error_str}")

        # inject error into state so coder sees it on retry
        # we do this by adding to the message bus
        messages = list(state.get("messages") or [])
        messages.append(AgentMessage(
            agent="verifier",
            type="tool_result",
            content=f"SYNTAX ERROR in generated code: {error_str}. Fix this before proceeding.",
            metadata={"errors": errors, "block_count": len(python_blocks)},
        ))
        state["messages"] = messages

        # also add to session memory so system learns from this
        from app.services.session_memory import session_memory
        session_memory.add(f"Syntax error in generated code: {error_str}")

    else:
        state["syntax_ok"] = True
        state["syntax_error"] = ""
        print(f"[VERIFIER] Syntax OK — {len(python_blocks)} block(s) verified")

        messages = list(state.get("messages") or [])
        messages.append(AgentMessage(
            agent="verifier",
            type="decision",
            content=f"Syntax verified OK — {len(python_blocks)} Python block(s) passed",
            metadata={"block_count": len(python_blocks)},
        ))
        state["messages"] = messages

    return state