from __future__ import annotations
from typing import TypedDict, List, Optional, Dict, Any


class AgentMessage(TypedDict):
    agent: str
    type: str
    content: str
    metadata: Dict


class AgentState(TypedDict):
    # core
    project_id: str
    query: str

    # agent communication bus
    messages: List[AgentMessage]

    # intent + routing
    intent: Optional[str]
    route: Optional[str]

    # planning — now structured as list of steps
    plan: Optional[str]
    plan_steps: Optional[List[str]]        # NEW — parsed list of steps
    current_step: Optional[int]            # NEW — which step coder is on
    file_structure: Optional[str]

    # retrieval
    context: Optional[Dict]
    retrieval_confidence: Optional[float]
    retrieval_ok: Optional[bool]
    corrective_attempted: Optional[bool]

    # tool calls
    tool_calls: Optional[List[str]]
    tool_results: Optional[Dict[str, Any]]

    # ReAct loop — coder requesting more context
    coder_tool_requests: Optional[List[Dict]]   # NEW — tools coder asked for
    coder_iterations: Optional[int]             # NEW — how many times coder has run
    additional_context: Optional[List[str]]     # NEW — context fetched by coder mid-task

    # generation
    answer: Optional[str]
    final_answer: Optional[str]

    # syntax verification
    syntax_ok: Optional[bool]               # NEW
    syntax_error: Optional[str]             # NEW — exact error from ast.parse

    # review + retry
    review: Optional[str]
    needs_retry: Optional[bool]
    retry_count: Optional[int]

    # supervisor
    supervisor_notes: Optional[str]

    # errors
    error: Optional[str]