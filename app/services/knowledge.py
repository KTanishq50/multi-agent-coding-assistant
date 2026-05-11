"""
Knowledge base — domain docs ingested once, used across all projects.
Separate from project-specific codebase chunks.
Lives in ./knowledge/ folder at project root.
"""
from __future__ import annotations
import os
from pathlib import Path
import chromadb
from app.services.embeddings import get_embedding

KNOWLEDGE_DIR = "./knowledge"
_chroma = chromadb.PersistentClient(path="./chroma_db")
knowledge_collection = _chroma.get_or_create_collection("knowledge_base")

ALLOWED_EXTENSIONS = (".md", ".txt", ".rst")


def ingest_knowledge_base() -> int:
    if not os.path.exists(KNOWLEDGE_DIR):
        os.makedirs(KNOWLEDGE_DIR)
        return 0

    # check what's already ingested by listing existing IDs
    try:
        existing = knowledge_collection.get()
        existing_ids = set(existing["ids"]) if existing["ids"] else set()
    except Exception:
        existing_ids = set()

    ingested = 0
    for root, _, files in os.walk(KNOWLEDGE_DIR):
        for fname in files:
            if not fname.endswith(ALLOWED_EXTENSIONS):
                continue
            full_path = os.path.join(root, fname)
            rel_path = os.path.relpath(full_path, KNOWLEDGE_DIR)
            doc_id_prefix = f"kb_{rel_path.replace(os.sep, '_')}"

            # skip if any chunk from this doc is already ingested
            if any(eid.startswith(doc_id_prefix) for eid in existing_ids):
                continue

            try:
                content = Path(full_path).read_text(encoding="utf-8", errors="replace")
                chunks = _chunk_doc(content, doc_id_prefix, rel_path)
                for chunk_id, chunk_text in chunks:
                    emb = get_embedding(chunk_text)
                    if emb:
                        knowledge_collection.add(
                            ids=[chunk_id],
                            documents=[chunk_text],
                            embeddings=[emb],
                            metadatas=[{"source": rel_path, "chunk": 0}]
                        )
                        ingested += 1
                print(f"[KNOWLEDGE] Ingested {rel_path} → {len(chunks)} chunks")
            except Exception as e:
                print(f"[KNOWLEDGE] Failed {rel_path}: {e}")

    return ingested


def _chunk_doc(content: str, doc_id: str, source: str) -> list[tuple[str, str]]:
    """Better chunking with overlap."""
    # You can keep your current logic or switch to recursive splitter / langchain TextSplitter
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks = []
    current = ""
    chunk_idx = 0
    overlap = 100  # characters

    for para in paragraphs:
        if len(current) + len(para) > 600 and current:
            chunks.append((f"{doc_id}_chunk{chunk_idx}", current.strip()))
            chunk_idx += 1
            # overlap
            current = current[-overlap:] + "\n\n" + para
        else:
            current += "\n\n" + para

    if current.strip():
        chunks.append((f"{doc_id}_chunk{chunk_idx}", current.strip()))

    return chunks


def retrieve_knowledge(query: str, k: int = 2) -> list[str]:
    """
    Retrieve relevant knowledge base chunks for a query.
    Returns list of text chunks.
    """
    try:
        count = knowledge_collection.count()
        if count == 0:
            return []
    except Exception:
        return []

    emb = get_embedding(query)
    if not emb:
        return []

    try:
        results = knowledge_collection.query(
            query_embeddings=[emb],
            n_results=min(k, count)
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        print(f"[KNOWLEDGE] Retrieve error: {e}")
        return []