import uuid
import chromadb
from app.services.embeddings import get_embedding
from langsmith import traceable

client            = chromadb.PersistentClient(path="./chroma_db")
memory_collection = client.get_or_create_collection("experience")


@traceable(name="store_memory", run_type="tool")
def store_memory(mistake: str, fix: str, tags=None):
    if not mistake or not fix:
        return
    if len(mistake) < 10 or len(fix) < 10:
        return

    text = f"Mistake: {mistake}\nFix: {fix}"
    emb  = get_embedding(text)
    if not emb:
        return

    try:
        memory_collection.add(
            ids=[str(uuid.uuid4())],
            documents=[text],
            embeddings=[emb],
            metadatas=[{"mistake": mistake, "fix": fix, "tags": tags or []}]
        )
    except Exception as e:
        print(f"[MEMORY ERROR] {e}")


@traceable(name="retrieve_memory", run_type="tool")
def retrieve_memory(query: str, k: int = 2) -> list[str]:
    try:
        count = memory_collection.count()
    except Exception:
        return []

    if count == 0:
        return []

    emb = get_embedding(query)
    if not emb:
        return []

    try:
        results = memory_collection.query(
            query_embeddings=[emb],
            n_results=min(k, count)
        )
        return results.get("documents", [[]])[0]
    except Exception as e:
        print(f"[MEMORY RETRIEVE ERROR] {e}")
        return []


def format_memory(memories: list[str]) -> str:
    if not memories:
        return "None"
    return "\n".join([f"- {m}" for m in memories])