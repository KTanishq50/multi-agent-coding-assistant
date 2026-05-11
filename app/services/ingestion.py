import zipfile
from pathlib import Path
from app.services.indexer import get_code_files
from app.services.chunker import chunk_code


def extract_zip(zip_path: str, extract_to: str) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)


def process_project(project_path: str) -> list[dict]:
    """
    Walk all code files in project_path and return a flat list of chunks.
    Each chunk is already a dict with keys: content, file, type, name.
    """
    files = get_code_files(project_path)
    all_chunks: list[dict] = []

    for file_path in files:
        try:
            text = Path(file_path).read_text(encoding="utf-8", errors="replace")
            chunks = chunk_code(text, file_path=file_path)
            # chunk_code already returns list[dict] — just extend, don't re-wrap
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"[ERROR reading {file_path}] {e}")

    print(f"[INGEST] {len(files)} files → {len(all_chunks)} chunks")
    return all_chunks