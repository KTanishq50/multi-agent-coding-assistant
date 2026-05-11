"""
Coder agent with ReAct loop.
- Executes planner steps one by one
- Can request read_full_file or search_code mid-task
- Validates tool requests (catches placeholder values)
- Special minimal prompt for syntax retries
- Reads supervisor guidance on reviewer retry
"""
from __future__ import annotations
import re
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm import call_llm
from app.services.session_memory import session_memory
from app.services.tools import read_full_file, search_code, grep_search
from .state import AgentState, AgentMessage
from langsmith import traceable

MAX_CODER_ITERATIONS = 3

_SYSTEM = {
    "write":   "You are an expert software developer. Write clean, complete, working code.",
    "fix":     "You are an expert debugger. Fix the bug and explain the change.",
    "explain": "You are a senior engineer. Explain the code clearly and precisely.",
    "general": "You are a helpful code assistant.",
}

_PLACEHOLDER_PATTERNS = [
    r"^<.*>$", r"^your[_\s]", r"^example",
    r"^path/to", r"^\.\.\.", r"^<path",
    r"filename", r"relative_path",
]


def _is_placeholder(value: str) -> bool:
    v = value.strip().lower()
    for pattern in _PLACEHOLDER_PATTERNS:
        if re.search(pattern, v):
            return True
    if "." not in v and "/" not in v and "\\" not in v:
        return True
    return False


def _parse_tool_request(text: str) -> dict | None:
    text = text.strip()

    need_file = re.search(r"NEED_FILE:\s*([^\n]+)", text)
    if need_file:
        path = need_file.group(1).strip()
        if _is_placeholder(path):
            print(f"[CODER] Rejected placeholder file request: {path!r}")
            return None
        return {"tool": "read_full_file", "arg": path}

    need_search = re.search(r"NEED_SEARCH:\s*([^\n]+)", text)
    if need_search:
        term = need_search.group(1).strip()
        if _is_placeholder(term) or len(term) < 2:
            print(f"[CODER] Rejected placeholder search: {term!r}")
            return None
        return {"tool": "search_code", "arg": term}

    return None


def _execute_tool_request(project_id: str, request: dict) -> str:
    tool = request["tool"]
    arg  = request["arg"]

    if tool == "read_full_file":
        result = read_full_file(project_id, arg)
        return f"# Contents of {arg}:\n{result}"

    if tool == "search_code":
        hits = search_code(project_id, arg) or grep_search(project_id, arg)
        if not hits:
            return f"# No results for: {arg}"
        lines = [f"# Search results for '{arg}':"]
        for h in hits[:10]:
            lines.append(f"  {h['relative_path']}:{h['line_number']}: {h['line_content']}")
        return "\n".join(lines)

    return f"[ERROR] Unknown tool: {tool}"


def _build_step_instructions(steps: list[str], current_step: int) -> str:
    if not steps:
        return ""
    lines = ["Execution plan:"]
    for i, step in enumerate(steps):
        if i == current_step:
            prefix = "→ EXECUTE NOW:"
        elif i < current_step:
            prefix = "✓ Done:"
        else:
            prefix = "  Pending:"
        lines.append(f"{prefix} Step {i+1}: {step}")
    return "\n".join(lines)


def _build_coder_prompt(state: AgentState, additional_context: list[str]) -> list:
    intent = state.get("intent", "general")
    context = state.get("context") or {}
    code_chunks = context.get("code", []) if isinstance(context, dict) else []
    memory_text = context.get("memory", "") if isinstance(context, dict) else ""

    syntax_error = state.get("syntax_error", "")
    is_syntax_retry = bool(syntax_error)

    #  syntax retry — minimal prompt
    # keeps tinyllama from echoing the full prompt
    if is_syntax_retry:
        previous_answer = state.get("answer", "")
        return [
            SystemMessage(content=(
                "You are a Python developer. Fix the syntax error in the code below.\n"
                "Output ONLY the corrected Python code. Nothing else."
            )),
            HumanMessage(content=(
                f"Syntax error: {syntax_error}\n\n"
                f"Code to fix:\n{previous_answer[:800]}\n\n"
                f"Task: {state['query']}\n\n"
                f"Output only the fixed code:"
            ))
        ]

    #  normal path
    all_context  = code_chunks + additional_context
    context_str  = "\n\n---\n\n".join(all_context)[:2000]
    session_ctx  = "\n".join(session_memory.get())[:300]

    # compact agent summary — prevents tinyllama from summarising pipeline
    retrieval_note = next(
        (m["metadata"].get("score") for m in (state.get("messages") or [])
         if m["agent"] == "rag_grader"),
        None
    )
    retriever_chunk_count = next(
        (m["metadata"].get("chunk_count") for m in (state.get("messages") or [])
         if m["agent"] == "retriever"),
        0
    )
    agent_summary = (
        f"Context retrieved: {retriever_chunk_count} chunk(s). "
        f"Retrieval confidence: {retrieval_note or state.get('retrieval_confidence', '?')}"
    )

    steps         = state.get("plan_steps") or []
    current_step  = state.get("current_step", 0)
    step_directive = _build_step_instructions(steps, current_step)

    # supervisor guidance from retry (this is what makes retry smarter)
    supervisor_guidance = state.get("supervisor_notes", "")
    retry_note = ""
    review_feedback = state.get("review", "")
    if (review_feedback
            and "REJECTED" in str(review_feedback).upper()
            and state.get("retry_count", 0) > 0):
        retry_note = (
            f"\nPrevious attempt REJECTED.\n"
            f"Supervisor guidance: {supervisor_guidance[:300]}\n"
            f"Reviewer said: {review_feedback[:200]}\n"
            f"Fix these issues in your response.\n"
        )

    confidence_note = ""
    if not state.get("retrieval_ok", True):
        conf = state.get("retrieval_confidence", 0)
        confidence_note = (
            f"\nNote: retrieval confidence low ({conf:.2f}). State assumptions.\n"
        )

    system_prompt = _SYSTEM.get(intent, _SYSTEM["general"])

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=(
            f"Pipeline status: {agent_summary}\n\n"
            f"{step_directive}\n\n"
            f"Codebase context:\n{context_str}\n\n"
            f"Past mistakes to avoid:\n{memory_text}\n\n"
            f"Session history:\n{session_ctx}\n"
            f"{confidence_note}{retry_note}\n"
            f"Task: {state['query']}"
        ))
    ]


def _clean_coder_output(text: str) -> str:
    """Extract code from markdown fences if present."""
    text = text.strip()
    code_block = re.search(r"```python\s*(.*?)\s*```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    code_block = re.search(r"```(.*?)```", text, re.DOTALL)
    if code_block:
        return code_block.group(1).strip()
    return text


@traceable(name="coder_agent", run_type="chain")
def coder_node(state: AgentState) -> AgentState:
    project_id         = state["project_id"]
    additional_context = list(state.get("additional_context") or [])
    coder_tool_requests = list(state.get("coder_tool_requests") or [])
    iterations         = state.get("coder_iterations", 0)
    tool_calls         = list(state.get("tool_calls") or [])

    final_answer = None

    for i in range(MAX_CODER_ITERATIONS):
        iterations += 1
        print(f"[CODER] Iteration {iterations}")

        prompt = _build_coder_prompt(state, additional_context)
        output = call_llm(prompt)
        output = _clean_coder_output(output)

        tool_request = _parse_tool_request(output)

        if tool_request and iterations < MAX_CODER_ITERATIONS:
            tool_result = _execute_tool_request(project_id, tool_request)
            additional_context.append(tool_result)
            coder_tool_requests.append(tool_request)
            tool_calls.append(
                f"coder→{tool_request['tool']}({tool_request['arg']!r})"
            )
            print(f"[CODER] Tool request: {tool_request['tool']}({tool_request['arg']!r})")

            steps        = state.get("plan_steps") or []
            current_step = state.get("current_step", 0)
            state["current_step"]       = min(current_step + 1, len(steps))
            state["additional_context"] = additional_context
            state["coder_tool_requests"] = coder_tool_requests
            state["tool_calls"]         = tool_calls
            continue
        else:
            final_answer = output
            break

    if not final_answer:
        final_answer = "Could not produce output after maximum iterations."

    state["answer"]              = final_answer
    state["coder_iterations"]    = iterations
    state["additional_context"]  = additional_context
    state["coder_tool_requests"] = coder_tool_requests
    state["tool_calls"]          = tool_calls

    messages = list(state.get("messages") or [])
    messages.append(AgentMessage(
        agent="coder",
        type="code",
        content=final_answer[:300] + ("..." if len(final_answer) > 300 else ""),
        metadata={
            "intent":        state.get("intent"),
            "iterations":    iterations,
            "tool_requests": len(coder_tool_requests),
            "retry_count":   state.get("retry_count", 0),
        },
    ))
    state["messages"] = messages

    print(f"[CODER] Done in {iterations} iterations, {len(coder_tool_requests)} tool requests")
    return state