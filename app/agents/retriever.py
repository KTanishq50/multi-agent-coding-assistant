"""
Retriever agent — handles all context gathering.
Uses tools actively: RAG, search_code, read_full_file.
Reads planner's message to understand what to look for.
Communicates results back via message bus.
"""
from __future__ import annotations
from app.services.rag import query_codebase
from app.services.memory import retrieve_memory, format_memory
from app.services.tools import search_code, read_full_file, grep_search
from .state import AgentState, AgentMessage

from app.services.knowledge import retrieve_knowledge

_MAX_CONTEXT_CHARS = 1500

from langsmith import traceable
@traceable(name="retriver_agent")
def retriever_node(state: AgentState) -> AgentState:
    query = state["query"]
    project_id = state["project_id"]
    intent = state.get("intent", "general")
    route = state.get("route", "proceed")
    tool_calls = list(state.get("tool_calls") or [])
    tool_results = dict(state.get("tool_results") or {})

    # read planner's message to understand what files to look at
    messages = state.get("messages") or []
    planner_plan = next(
        (m["content"] for m in reversed(messages) if m["agent"] == "planner"),
        ""
    )

    # step 1: semantic RAG 
    if route == "skip_retrieval":
        chunks = []
        print("[RETRIEVER] Skipping retrieval (supervisor decision)")
    else:
        chunks = query_codebase(query, project_id, n_results=5)
        tool_calls.append(f"query_codebase({query!r})")

    # step 2: tool-based retrieval for fix/explain
    extra_context = []

    if intent in ("fix", "explain") and route != "skip_retrieval":
        # extract identifiers from query and plan
        terms = _extract_terms(query + " " + planner_plan)
        for term in terms[:3]:
            results = search_code(project_id, term)
            tool_calls.append(f"search_code({term!r}) → {len(results)} hits")
            tool_results[f"search:{term}"] = results

            # read the first file that defines it
            for r in results[:2]:
                if r.get("definition_type"):  # it's a definition, not just usage
                    content = read_full_file(project_id, r["relative_path"])
                    if not content.startswith("[ERROR]"):
                        extra_context.append(f"# {r['relative_path']}\n{content[:1000]}")
                        tool_calls.append(f"read_full_file({r['relative_path']!r})")
                    break

    #step 3: for write intent, grep for related patterns 
    if intent == "write":
        # look for similar functions that already exist
        write_terms = _extract_write_targets(query)
        for term in write_terms[:2]:
            hits = grep_search(project_id, term)
            if hits:
                tool_calls.append(f"grep_search({term!r}) → {len(hits)} hits")
                tool_results[f"grep:{term}"] = hits
                # read the file with the most hits
                from collections import Counter
                file_counts = Counter(h["relative_path"] for h in hits)
                top_file = file_counts.most_common(1)[0][0] if file_counts else None
                if top_file:
                    content = read_full_file(project_id, top_file)
                    if not content.startswith("[ERROR]"):
                        extra_context.append(f"# {top_file} (related patterns)\n{content[:800]}")
                        tool_calls.append(f"read_full_file({top_file!r})")

    # step 4: memory retrieval 
    memories = retrieve_memory(query, k=2)

    # step 5: knowledge base retrieval 
    
    knowledge_chunks = retrieve_knowledge(query, k=2)
    if knowledge_chunks:
        tool_calls.append(f"retrieve_knowledge({query!r}) → {len(knowledge_chunks)} chunks")
        # prepend with source label
        knowledge_context = [f"# From knowledge base:\n{c}" for c in knowledge_chunks]
    else:
        knowledge_context = []

    # merge and dedup 
    all_code = chunks + extra_context + knowledge_context
    seen = set()
    deduped = []
    for c in all_code:
        if c not in seen:
            seen.add(c)
            deduped.append(c)

    state["context"] = {
        "code": deduped[:7],
        "memory": format_memory(memories),
    }
    state["tool_calls"] = tool_calls
    state["tool_results"] = tool_results

    # communicate to other agents via message bus
    context_summary = f"Retrieved {len(deduped)} context pieces. Tools used: {', '.join(tool_calls[-5:])}"
    messages = list(state.get("messages") or [])
    messages.append(AgentMessage(
        agent="retriever",
        type="context",
        content=context_summary,
        metadata={
            "chunk_count": len(deduped),
            "tool_count": len(tool_calls),
            "memory_count": len(memories),
        },
    ))
    state["messages"] = messages

    print(f"[RETRIEVER] {len(deduped)} chunks, {len(tool_calls)} tool calls")
    return state


def _extract_terms(text: str) -> list[str]:
    """Extract likely code identifiers from text."""
    import re
    snake = re.findall(r"\b[a-z][a-z0-9_]{2,}\b", text.lower())
    camel = re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", text)
    stop = {
        "the", "fix", "bug", "code", "function", "class", "file", "how",
        "why", "what", "does", "make", "write", "create", "add", "get",
        "this", "that", "with", "for", "and", "not", "use", "new", "list",
        "return", "import", "from", "def", "var", "let", "const", "perform",
    }
    terms = [t for t in snake if t not in stop] + camel
    # deduplicate preserving order
    seen = set()
    result = []
    for t in terms:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:6]


def _extract_write_targets(query: str) -> list[str]:
    """Extract what the user wants to write — function names, patterns."""
    import re
    # look for quoted names, snake_case names, class names
    quoted = re.findall(r'["\']([^"\']+)["\']', query)
    snake = re.findall(r"\b[a-z][a-z0-9_]{3,}\b", query.lower())
    stop = {"write", "create", "function", "class", "that", "with", "which", "from", "like", "perform"}
    targets = quoted + [s for s in snake if s not in stop]
    seen = set()
    result = []
    for t in targets:
        if t not in seen:
            seen.add(t)
            result.append(t)
    return result[:3]