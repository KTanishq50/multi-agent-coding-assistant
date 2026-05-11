import requests

# OLLAMA_URL = "http://localhost:11434/api/embeddings"
OLLAMA_URL = "http://host.docker.internal:11434/api/embeddings"#for docker

MODEL = "nomic-embed-text"


def get_embedding(text: str):
    if not text or not text.strip():
        print("[EMBED ERROR] empty text passed")
        return None

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": text  # Ollama uses "prompt", NOT "input"
            },
            timeout=60
        )
        response.raise_for_status()
        data = response.json()

        embedding = data.get("embedding")

        if not embedding:
            print("[EMBED ERROR] empty embedding returned:", data)
            return None

        return embedding

    except Exception as e:
        print("[EMBED ERROR]", str(e))
        return None