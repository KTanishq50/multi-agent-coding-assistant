from __future__ import annotations
import re
from langchain_core.messages import SystemMessage, HumanMessage
from app.services.llm import call_llm
from app.services.tools import get_project_summary
from .state import AgentState, AgentMessage


def _parse_steps(plan_text: str) -> list[str]:
    """
    Extract numbered steps from plan text.
    
    Quality filters (not count filters):
    - Must start with a number
    - Must be a concrete action (short enough to be a step)
    - Skip code blocks, prose paragraphs, examples
    - Stop when numbered sequence breaks (e.g. 1,2,3 then prose)
    - Hard max of 6 to prevent runaway plans
    
    With a good model this will return 4-5 meaningful steps.
    With tinyllama it will still cap the garbage.
    """
    lines = plan_text.strip().split("\n")
    steps = []
    last_step_num = 0
    in_code_block = False

    for line in lines:
        raw = line.strip()

        # track code fences
        if raw.startswith("```"):
            in_code_block = not in_code_block
            continue
        if in_code_block:
            continue

        # must start with a number
        match = re.match(r"^(\d+)[.):\-]\s*(.+)", raw)
        if not match:
            # a blank line after we have steps doesn't stop us
            # but a non-numbered non-blank line that comes after
            # numbered steps might be prose continuation — skip it
            continue

        step_num = int(match.group(1))
        content = match.group(2).strip()

        # numbered sequence must be contiguous (1,2,3 not 1,2,5,8)
        # if it jumps more than 1, it's a new section not a continuation
        if steps and step_num > last_step_num + 2:
            break

        # skip code-like content even if numbered
        if content.startswith(("def ", "class ", "import ", "from ", "```")):
            continue

        # skip example/test code patterns
        if re.match(r"^(import |assert |print\(|if __name__)", content):
            continue

        # skip lines that are clearly prose not actions
        # actions start with a verb or file reference
        # prose starts with "I ", "The ", "This ", "We ", "In "
        if re.match(r"^(I |The |This |We |In |Note|Here|For|Also|Then|So )", content, re.IGNORECASE):
            continue

        # reasonable length for a step — not a one-word label, not a paragraph
        if len(content) < 5 or len(content) > 150:
            continue

        steps.append(content)
        last_step_num = step_num

        # hard max to prevent completely runaway models
        if len(steps) >= 6:
            break

    if not steps:
        return [
            "Identify relevant code in the project",
            "Write the requested code following project patterns",
            "Verify the output is correct",
        ]

    return steps

from langsmith import traceable
@traceable(name="planner_agent")
def planner_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "general")
    route = state.get("route", "proceed")
    summary = get_project_summary(state["project_id"])
    state["file_structure"] = summary

    _INTENT_HINTS = {
        "write":   "User wants new code written.",
        "fix":     "User wants a bug fixed.",
        "explain": "User wants code explained.",
        "general": "General question about the codebase.",
    }
    hint = _INTENT_HINTS.get(intent, _INTENT_HINTS["general"])

    route_note = ""
    if route == "skip_retrieval":
        route_note = "\nNote: answer from general knowledge, no retrieval needed."
    elif route == "needs_more_context":
        route_note = "\nNote: retrieval may be weak, plan for broader search."

    prompt = [
        SystemMessage(content=(
            f"You are a planning agent. {hint}{route_note}\n\n"
            f"Project files:\n{summary}\n\n"
            f"Output ONLY a numbered list of exactly 3 steps.\n"
            f"Each step must be a single concrete action.\n"
            f"Reference actual file names when relevant.\n"
            f"No code. No explanations. Steps only.\n"
            f"Format:\n1. <action>\n2. <action>\n3. <action>"
        )),
        HumanMessage(content=state["query"])
    ]

    plan = call_llm(prompt)
    steps = _parse_steps(plan)

    state["plan"] = plan
    state["plan_steps"] = steps
    state["current_step"] = 0

    messages = list(state.get("messages") or [])
    messages.append(AgentMessage(
        agent="planner",
        type="plan",
        content=plan,
        metadata={"intent": intent, "step_count": len(steps), "steps": steps},
    ))
    state["messages"] = messages

    print(f"[PLANNER] {len(steps)} steps for intent={intent}")
    return state