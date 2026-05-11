from __future__ import annotations
import re
import json
import chromadb
from collections import defaultdict, Counter
from app.services.llm import call_llm_raw
from langsmith import traceable

MAX_EXPANSION_WORDS = 20

GENERAL_VOCAB = {
    "csv":    "pandas read_csv csv reader file",
    "json":   "json load dump parse requests",
    "http":   "requests get post api fetch",
    "file":   "open read write pathlib",
    "api":    "endpoint fastapi route request response",
    "class":  "class init self method",
    "async":  "async await asyncio",
    "numpy":  "np array ndarray reshape zeros ones dtype",
    "pandas": "pd DataFrame read_csv groupby merge apply",
    "torch":  "tensor model forward backward optimizer loss",
    "sql":    "query select insert update delete database",
    "auth":   "login authenticate token jwt password user",
    "test":   "pytest unittest assert mock fixture",
    "error":  "exception traceback raise try except",
    "socket": "socket connect bind listen recv send",
    "thread": "threading Thread Lock queue concurrent",
    "regex":  "re compile match search findall pattern",
}

_PROMPT_LEAK_PHRASES = [
    "coding search", "search query", "related technical",
    "no explanation", "space-separated", "output only",
    "expand this", "list 4", "one per line", "terms only",
]

_chroma           = chromadb.PersistentClient(path="./chroma_db")
_exp_collection   = _chroma.get_or_create_collection("project_expansions")
_global_collection = _chroma.get_or_create_collection("global_patterns")


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z_]\w*", text.lower())


def index_chunks_for_expansion(chunks: list[dict], project_id: str) -> None:
    project_learned = defaultdict(Counter)
    for chunk in chunks:
        tokens = _tokenize(chunk.get("content", ""))
        for i, token in enumerate(tokens):
            if len(token) < 3:
                continue
            window = tokens[max(0, i - 5):i + 6]
            for neighbor in window:
                if neighbor != token and len(neighbor) >= 3:
                    project_learned[token][neighbor] += 1

    serializable = {
        t: [n for n, _ in c.most_common(10)]
        for t, c in project_learned.items()
    }
    _exp_collection.upsert(ids=[project_id], documents=[json.dumps(serializable)])

    for token, neighbors in serializable.items():
        try:
            existing = _global_collection.get(ids=[token])
            curr     = json.loads(existing.get("documents", ["[]"])[0]) if existing.get("documents") else []
            merged   = list(dict.fromkeys(curr + neighbors))[:15]
            _global_collection.upsert(ids=[token], documents=[json.dumps(merged)])
        except Exception:
            _global_collection.upsert(ids=[token], documents=[json.dumps(neighbors)])


def _load_project(project_id: str) -> dict:
    try:
        result = _exp_collection.get(ids=[project_id])
        return json.loads(result.get("documents", ["{}"])[0])
    except Exception:
        return {}


def _load_global(token: str) -> list[str]:
    try:
        result = _global_collection.get(ids=[token])
        return json.loads(result.get("documents", ["[]"])[0])
    except Exception:
        return []


def _llm_expand(query: str) -> str | None:
    prompt = (
        f"Python programming terms related to: {query}\n"
        f"List 4 terms only, one per line:"
    )
    try:
        result     = call_llm_raw(prompt, max_tokens=30, timeout=15).strip()
        if not result:
            return None
        first_line = result.split("\n")[0].strip()
        if len(first_line.split()) > 8:
            return None
        first_lower = first_line.lower()
        for phrase in _PROMPT_LEAK_PHRASES:
            if phrase in first_lower:
                return None
        if first_line.lower() == query.lower():
            return None
        return first_line
    except Exception:
        return None


@traceable(name="rewrite_query", run_type="tool")
def rewrite_query(query: str, project_id: str) -> str:
    query = query.strip()
    if not query:
        return query

    parts  = [query]
    tokens = _tokenize(query)

    # 1. static vocab
    for key, val in GENERAL_VOCAB.items():
        if key in query.lower():
            parts.append(val)
            break

    # 2. global co-occurrence
    for tok in tokens[:2]:
        global_n = _load_global(tok)
        if global_n:
            parts.append(" ".join(global_n[:4]))
            break

    # 3. project co-occurrence
    project_data = _load_project(project_id)
    for tok in tokens:
        if tok in project_data:
            parts.append(" ".join(project_data[tok][:5]))
            break

    # 4. LLM expansion
    llm_exp = _llm_expand(query)
    if llm_exp:
        parts.append(llm_exp)

    final  = " ".join(parts).split()[:MAX_EXPANSION_WORDS]
    result = " ".join(final)
    print(f"[REWRITE] {query} → {result}")
    return result