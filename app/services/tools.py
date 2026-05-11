"""
Tools layer — all file system operations sandboxed to workspace/{project_id}/.
These are callable by any agent.
Tools log themselves to state["tool_calls"] and state["tool_results"].
"""
from __future__ import annotations
import os
import re
import ast
from pathlib import Path
from typing import Any

WORKSPACE = "workspace"
ALLOWED_EXTENSIONS = (
    ".py", ".js", ".ts", ".tsx", ".jsx",
    ".cpp", ".c", ".h", ".java", ".go", ".rs",
    ".md", ".txt", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini",
    ".html", ".css", ".sh",
)
MAX_FILE_READ_CHARS = 8000
MAX_SEARCH_RESULTS = 40


def _safe_path(project_id: str, relative_path: str) -> str | None:
    base = os.path.abspath(os.path.join(WORKSPACE, project_id))
    full = os.path.abspath(os.path.join(base, relative_path.lstrip("/")))
    if not full.startswith(base):
        return None
    return full


#  list_project_files

def list_project_files(project_id: str) -> list[dict]:
    """
    Returns all code files as list of dicts.
    Each: {relative_path, size_lines, extension, type}
    """
    base = os.path.join(WORKSPACE, project_id)
    if not os.path.exists(base):
        return []

    results = []
    for root, dirs, files in os.walk(base):
        dirs[:] = [
            d for d in dirs
            if not d.startswith(".") and d not in ("__pycache__", "node_modules", ".git", "dist", "build")
        ]
        for fname in sorted(files):
            if not fname.endswith(ALLOWED_EXTENSIONS):
                continue
            full = os.path.join(root, fname)
            rel = os.path.relpath(full, base)
            try:
                content = Path(full).read_text(encoding="utf-8", errors="replace")
                lines = content.count("\n") + 1
            except Exception:
                lines = 0
                content = ""
            results.append({
                "relative_path": rel,
                "size_lines": lines,
                "extension": os.path.splitext(fname)[1],
                "type": _guess_file_type(fname, content),
            })
    return results


def _guess_file_type(fname: str, content: str) -> str:
    if fname.endswith(".py"):
        if "class " in content and "def " in content:
            return "module"
        if "def test_" in content or "import pytest" in content:
            return "test"
        return "script"
    if fname.endswith((".js", ".ts", ".tsx", ".jsx")):
        return "frontend"
    if fname.endswith((".md", ".txt")):
        return "docs"
    return "config"


def get_file_structure(project_id: str) -> str:
    """Compact string representation of project structure for planner."""
    files = list_project_files(project_id)
    if not files:
        return "No files found in project."
    lines = [f"  {f['relative_path']} ({f['size_lines']} lines, {f['type']})" for f in files]
    return "\n".join(lines)


#  read_full_file 

def read_full_file(project_id: str, relative_path: str) -> str:
    """Read complete file. Returns content or error string."""
    full = _safe_path(project_id, relative_path)
    if not full:
        return f"[ERROR] Access denied: {relative_path}"
    if not os.path.exists(full):
        return f"[ERROR] File not found: {relative_path}"
    try:
        content = Path(full).read_text(encoding="utf-8", errors="replace")
        if len(content) > MAX_FILE_READ_CHARS:
            content = content[:MAX_FILE_READ_CHARS] + "\n... [truncated — file too large]"
        return content
    except Exception as e:
        return f"[ERROR] Could not read {relative_path}: {e}"


# get_project_summary

def get_project_summary(project_id: str) -> str:
    """
    Compact overview: file list + key function/class names from Python files.
    Used by planner to understand project without reading all files.
    """
    files = list_project_files(project_id)
    if not files:
        return "Empty project."

    lines = ["Project structure:"]
    for f in files:
        lines.append(f"  {f['relative_path']} ({f['size_lines']} lines)")

        # for Python files, extract top-level definitions
        if f["extension"] == ".py":
            full = _safe_path(project_id, f["relative_path"])
            if full and os.path.exists(full):
                defs = _extract_python_defs(full)
                if defs:
                    lines.append(f"    → {', '.join(defs[:8])}")

    return "\n".join(lines)


def _extract_python_defs(filepath: str) -> list[str]:
    """Extract top-level function and class names from a Python file."""
    try:
        content = Path(filepath).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content)
        defs = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if isinstance(node, ast.ClassDef):
                    defs.append(f"class {node.name}")
                else:
                    defs.append(f"def {node.name}()")
        return defs
    except Exception:
        return []


#  grep_search 

def grep_search(project_id: str, pattern: str, case_sensitive: bool = False) -> list[dict]:
    """
    Regex/literal search across all project files.
    Returns list of {relative_path, line_number, line_content}.
    """
    files = list_project_files(project_id)
    results = []
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        regex = re.compile(pattern, flags)
    except re.error:
        # fall back to literal search
        regex = re.compile(re.escape(pattern), flags)

    for f in files:
        full = _safe_path(project_id, f["relative_path"])
        if not full:
            continue
        try:
            lines = Path(full).read_text(encoding="utf-8", errors="replace").splitlines()
            for i, line in enumerate(lines, start=1):
                if regex.search(line):
                    results.append({
                        "relative_path": f["relative_path"],
                        "line_number": i,
                        "line_content": line.strip()[:200],
                    })
                    if len(results) >= MAX_SEARCH_RESULTS:
                        return results
        except Exception:
            continue
    return results


#  search_code 

def search_code(project_id: str, term: str) -> list[dict]:
    """
    Identifier-aware search. Finds function/class/variable definitions and usages.
    Combines grep with AST-level definition lookup for Python files.
    """
    # basic grep first
    grep_results = grep_search(project_id, term)

    # AST-level: find which Python file defines this as a function/class
    files = list_project_files(project_id)
    ast_results = []
    for f in files:
        if f["extension"] != ".py":
            continue
        full = _safe_path(project_id, f["relative_path"])
        if not full:
            continue
        try:
            content = Path(full).read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(content)
            lines = content.splitlines()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if term.lower() in node.name.lower():
                        lineno = node.lineno
                        ast_results.append({
                            "relative_path": f["relative_path"],
                            "line_number": lineno,
                            "line_content": lines[lineno - 1].strip()[:200] if lineno <= len(lines) else "",
                            "definition_type": "class" if isinstance(node, ast.ClassDef) else "function",
                        })
        except Exception:
            continue

    # merge, AST results first (more precise)
    seen = set()
    merged = []
    for r in ast_results + grep_results:
        key = (r["relative_path"], r["line_number"])
        if key not in seen:
            seen.add(key)
            merged.append(r)
        if len(merged) >= MAX_SEARCH_RESULTS:
            break

    return merged


#  write_file 

def write_file(project_id: str, relative_path: str, content: str) -> dict:
    """Write content to a file. Creates parent dirs if needed."""
    full = _safe_path(project_id, relative_path)
    if not full:
        return {"success": False, "error": "Access denied — path outside project"}
    try:
        os.makedirs(os.path.dirname(full), exist_ok=True)
        Path(full).write_text(content, encoding="utf-8")
        return {"success": True, "path": relative_path, "chars": len(content)}
    except Exception as e:
        return {"success": False, "error": str(e)}


#  extract_code_blocks 

def extract_code_blocks(text: str) -> str:
    """
    Extract code from markdown code fences.
    Falls back to raw text if no fences found.
    """
    pattern = r"```(?:\w+)?\n(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)
    if matches:
        return "\n\n".join(m.strip() for m in matches)
    return text.strip()


#  tool dispatcher 

TOOL_REGISTRY = {
    "list_project_files": list_project_files,
    "read_full_file":     read_full_file,
    "get_project_summary": get_project_summary,
    "grep_search":        grep_search,
    "search_code":        search_code,
    "write_file":         write_file,
}


def dispatch_tool(tool_name: str, project_id: str, **kwargs) -> Any:
    """
    Central dispatcher for tool calls.
    Used by retriever and coder when they decide to call tools.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"[ERROR] Unknown tool: {tool_name}"
    try:
        fn = TOOL_REGISTRY[tool_name]
        return fn(project_id=project_id, **kwargs)
    except Exception as e:
        return f"[ERROR] Tool {tool_name} failed: {e}"