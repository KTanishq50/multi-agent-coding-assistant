from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
import re
import os
import uuid

from app.services.ingestion import extract_zip, process_project
from app.services.rag import store_chunks, query_codebase, collection
from app.services.tools import (
    list_project_files, read_full_file, write_file,
    get_project_summary, extract_code_blocks
)
from app.agents.graph import graph
from app.services.llm import generate_answer
from app.services.session_memory import session_memory

router = APIRouter()
BASE_DIR = "workspace"


class QueryRequest(BaseModel):
    project_id: str
    question: str


class WriteFileRequest(BaseModel):
    project_id: str
    relative_path: str
    content: str


# helpers 

def _extract_plan_steps(plan_text: str) -> str:
    """
    Extract only the step lines from planner output.
    Strips code blocks, prose, examples — returns only numbered steps.
    """
    if not plan_text:
        return ""

    # remove code fences entirely
    cleaned = re.sub(r"```.*?```", "", plan_text, flags=re.DOTALL).strip()

    lines = cleaned.split("\n")
    steps = []
    for line in lines:
        line = line.strip()
        if re.match(r"^\d+[.):\-]\s*.+", line):
            content = re.sub(r"^\d+[.):\-]\s*", "", line).strip()
            if content and len(content) < 150:
                steps.append(f"{len(steps)+1}. {content}")

    if steps:
        return "\n".join(steps)

    # planner wrote prose/code instead of steps — show summary
    summary = cleaned[:200].strip()
    if summary:
        return f"(Planner output — steps not extracted)\n{summary}"
    return "(No plan generated)"


def _trim_review(review_text: str) -> str:
    """
    Extract just the verdict and brief critique from reviewer output.
    Strips echoed code/context that tinyllama repeats back.
    """
    if not review_text:
        return ""

    lines = review_text.strip().split("\n")
    if not lines:
        return ""

    verdict = lines[0].strip()  # APPROVED or REJECTED

    noise_markers = [
        "query:", "codebase context:", "answer to review:",
        "```python", "```", "def ", "import ", "---",
        "review criterion:", "agent context:", "if the answer",
    ]

    kept = [verdict]
    for line in lines[1:10]:  # max 10 lines from reviewer
        line_lower = line.strip().lower()
        if any(marker in line_lower for marker in noise_markers):
            break
        if line.strip():
            kept.append(line.strip())

    return "\n".join(kept)


#  upload

@router.post("/upload-zip")
async def upload_zip(file: UploadFile = File(...)):
    project_id = str(uuid.uuid4())
    project_path = os.path.join(BASE_DIR, project_id)
    os.makedirs(project_path, exist_ok=True)
    session_memory.clear()

    zip_path = os.path.join(project_path, file.filename)
    with open(zip_path, "wb") as f:
        f.write(await file.read())

    extract_zip(zip_path, project_path)
    os.remove(zip_path)

    chunks = process_project(project_path)
    print(f"[UPLOAD] {len(chunks)} chunks")

    if not chunks:
        return {"project_id": project_id, "chunks": 0, "status": "no_code_files_found"}

    store_chunks(chunks, project_id)
    files = list_project_files(project_id)

    return {
        "project_id": project_id,
        "chunks": len(chunks),
        "status": "embedded",
        "files": [f["relative_path"] for f in files],
    }


# query 

@router.post("/query")
async def query_project(request: QueryRequest):
    if not request.question or not request.question.strip():
        return {"answer": "Please provide a question.", "success": False}

    project_path = os.path.join(BASE_DIR, request.project_id)
    if not os.path.exists(project_path):
        return {"error": "Invalid project_id", "success": False}

    initial_state = {
        "project_id":          request.project_id,
        "query":               request.question.strip(),
        "messages":            [],
        "retry_count":         0,
        "needs_retry":         False,
        "tool_calls":          [],
        "tool_results":        {},
        "corrective_attempted": False,
        "coder_iterations":    0,
        "syntax_retry_count":  0,
        "additional_context":  [],
        "coder_tool_requests": [],
    }

    try:
        result = graph.invoke(initial_state, {"recursion_limit": 40})

        context = result.get("context") or {}
        if not isinstance(context, dict):
            context = {}

        agent_log = [
            {"agent": m["agent"], "type": m["type"], "content": m["content"][:200]}
            for m in (result.get("messages") or [])
        ]

        return {
            "success":              True,
            "answer":               result.get("final_answer") or result.get("answer") or "",
            "plan":                 _extract_plan_steps(result.get("plan") or ""),
            "review":               _trim_review(result.get("review") or ""),
            "intent":               result.get("intent") or "general",
            "route":                result.get("route") or "proceed",
            "retrieval_confidence": result.get("retrieval_confidence", 0),
            "retrieval_ok":         result.get("retrieval_ok", True),
            "tool_calls":           result.get("tool_calls") or [],
            "context_chunks":       len(context.get("code", [])),
            "agent_log":            agent_log,
            "supervisor_notes":     result.get("supervisor_notes") or "",
            "syntax_ok":            result.get("syntax_ok", True),
            "syntax_error":         result.get("syntax_error") or "",
            "coder_iterations":     result.get("coder_iterations", 1),
        }

    except Exception as e:
        print(f"[QUERY ERROR] {e}")
        chunks = query_codebase(request.question, request.project_id)
        fallback = generate_answer("\n\n".join(chunks[:3]), request.question)
        return {
            "success":              False,
            "answer":               fallback or "",
            "plan":                 "",
            "review":               "",
            "intent":               "general",
            "route":                "proceed",
            "retrieval_confidence": 0,
            "retrieval_ok":         False,
            "tool_calls":           [],
            "context_chunks":       0,
            "agent_log":            [],
            "error":                str(e)[:300],
            "syntax_ok":            True,
            "syntax_error":         "",
            "coder_iterations":     0,
        }


# file tools

@router.get("/project-files")
def project_files(project_id: str):
    files = list_project_files(project_id)
    return {"files": [f["relative_path"] for f in files]}


@router.get("/read-file")
def read_file_endpoint(project_id: str, path: str):
    content = read_full_file(project_id, path)
    return {"content": content, "path": path}


@router.post("/write-file")
async def write_file_endpoint(request: WriteFileRequest):
    clean = extract_code_blocks(request.content)
    result = write_file(request.project_id, request.relative_path, clean)
    return result


@router.get("/project-summary")
def project_summary_endpoint(project_id: str):
    summary = get_project_summary(project_id)
    return {"summary": summary}


@router.get("/debug")
def debug():
    return {
        "total_chunks_in_db": collection.count(),
        "session_memory_size": len(session_memory.get()),
    }