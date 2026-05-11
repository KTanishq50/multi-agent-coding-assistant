import ast


def split_large_block(lines: list[str], start: int, end: int, max_lines: int) -> list[str]:
    block = lines[start:end]
    return [
        "\n".join(block[i: i + max_lines])
        for i in range(0, len(block), max_lines)
    ]


def chunk_code(file_content: str, file_path: str = "", max_lines: int = 80) -> list[dict]:
    """
    Hybrid chunker:
      1. AST-based for .py files (functions / classes), split if too large
      2. Line-based fallback for all other files
    Always returns list[dict] with keys: content, file, type, name
    """
    chunks: list[dict] = []

    if file_path.endswith(".py"):
        try:
            tree = ast.parse(file_content)
            lines = file_content.splitlines()

            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start = node.lineno - 1
                    end = getattr(node, "end_lineno", start + max_lines)
                    sub_chunks = split_large_block(lines, start, end, max_lines)

                    for sub in sub_chunks:
                        if sub.strip():
                            chunks.append({
                                "content": sub,
                                "file": file_path,
                                "type": "class" if isinstance(node, ast.ClassDef) else "function",
                                "name": node.name,
                            })

            if chunks:
                return chunks

        except Exception as e:
            print(f"[AST ERROR] {file_path}: {e}")

    # Fallback: line-based
    lines = file_content.splitlines()
    for i in range(0, len(lines), max_lines):
        sub = "\n".join(lines[i: i + max_lines])
        if sub.strip():
            chunks.append({
                "content": sub,
                "file": file_path,
                "type": "generic",
                "name": f"chunk_{i}",
            })

    return chunks