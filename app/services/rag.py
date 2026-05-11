from __future__ import annotations
import numpy as np
import chromadb
import os
from langsmith import traceable

from app.services.embeddings import get_embedding
from app.services.bm25 import BM25Index
from app.services.query_rewriter import rewrite_query

_chroma    = chromadb.PersistentClient(path="./chroma_db")
collection = _chroma.get_or_create_collection("codebase")

_bm25_indices: dict[str, BM25Index] = {}
BM25_DIR = "./bm25_indices"
os.makedirs(BM25_DIR, exist_ok=True)


def _get_bm25(project_id: str) -> BM25Index:
    if project_id not in _bm25_indices:
        idx  = BM25Index()
        path = os.path.join(BM25_DIR, f"{project_id}.pkl")
        idx.load(path)
        _bm25_indices[project_id] = idx
    return _bm25_indices[project_id]


def _text(chunk) -> str:
    if isinstance(chunk, dict):
        return chunk.get("content", "")
    return str(chunk)


def _cosine(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / (denom + 1e-10))


def store_chunks(chunks: list[dict], project_id: str) -> None:
    print(f"\n--- STORING {len(chunks)} CHUNKS (project: {project_id}) ---")

    stored          = 0
    texts_for_bm25  = []

    for i, chunk in enumerate(chunks):
        text = _text(chunk)
        if not text.strip():
            continue

        embedding = get_embedding(text)
        if not embedding:
            print(f"[SKIP] embedding failed for chunk {i}")
            continue

        uid = f"{project_id}_{i}"

        try:
            collection.add(
                documents=[text],
                embeddings=[embedding],
                metadatas=[{
                    "file":     chunk.get("file", ""),
                    "project":  project_id,
                    "chunk_id": i,
                }],
                ids=[uid],
            )
            stored += 1
        except Exception as e:
            print(f"[SKIP] chunk {i} already stored: {e}")

        texts_for_bm25.append({
            "content": text,
            "project": project_id,
            "file":    chunk.get("file", ""),
        })

    print(f"--- STORED {stored}/{len(chunks)} ---")

    idx              = BM25Index()
    path             = os.path.join(BM25_DIR, f"{project_id}.pkl")
    idx.persist_path = path
    idx.build(texts_for_bm25)
    _bm25_indices[project_id] = idx

    from app.services.query_rewriter import index_chunks_for_expansion
    index_chunks_for_expansion(chunks, project_id)


@traceable(name="query_codebase", run_type="retriever")
def query_codebase(
    query:      str,
    project_id: str,
    n_results:  int = 5,
) -> list[str]:
    print("\n--- QUERY START ---")

    rewritten = rewrite_query(query, project_id)
    print(f"[QUERY]     {query}")
    print(f"[REWRITTEN] {rewritten}")

    candidates = []

    # BM25
    idx        = _get_bm25(project_id)
    bm25_hits  = idx.search(rewritten, top_k=8)
    bm25_texts = [_text(d) for d in bm25_hits if _text(d).strip()]
    candidates.extend(bm25_texts)
    print(f"[BM25] {len(bm25_texts)} chunks")

    # Chroma
    query_emb        = get_embedding(rewritten)
    chroma_docs      = []
    chroma_embeddings = []

    if query_emb:
        try:
            chroma_result = collection.query(
                query_embeddings=[query_emb],
                n_results=8,
                where={"project": project_id},
                include=["documents", "embeddings"]
            )
            chroma_docs       = chroma_result.get("documents", [[]])[0]
            chroma_embeddings = chroma_result.get("embeddings", [[]])[0]
            candidates.extend([d for d in chroma_docs if d.strip()])
            print(f"[Chroma] {len(chroma_docs)} chunks")
        except Exception as e:
            print(f"[Chroma Error] {e}")

    # dedup
    seen   = set()
    unique = []
    for t in candidates:
        if t and t.strip() and t not in seen:
            seen.add(t)
            unique.append(t)

    print(f"[CANDIDATES] {len(unique)} after dedup")

    if not unique:
        return []

    # rerank
    try:
        q_vec      = get_embedding(rewritten)
        emb_lookup = dict(zip(chroma_docs, chroma_embeddings))

        scored = []
        for doc in unique:
            if doc in emb_lookup:
                score = _cosine(q_vec, emb_lookup[doc])
            else:
                doc_emb = get_embedding(doc)
                score   = _cosine(q_vec, doc_emb) if doc_emb else 0.0
            scored.append((doc, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        final = [doc for doc, _ in scored[:n_results]]

    except Exception as e:
        print(f"[RERANK ERROR] {e}")
        final = unique[:n_results]

    print(f"[FINAL] {len(final)} chunks returned")
    return final