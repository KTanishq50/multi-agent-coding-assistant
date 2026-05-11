import re
import math
import pickle
import os
from collections import Counter


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75, persist_path: str = None):
        self.k1 = k1
        self.b = b
        self.persist_path = persist_path  # None until build() is called
        self.docs: list[dict] = []
        self.tokenized: list[list[str]] = []
        self.df: Counter = Counter()
        self.avgdl: float = 0.0
        self.N: int = 0

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"[a-zA-Z_]\w*|\d+", text.lower())

    def load(self, persist_path: str):
        self.persist_path = persist_path
        if os.path.exists(persist_path):
            try:
                with open(persist_path, "rb") as f:
                    data = pickle.load(f)
                    self.docs = data["docs"]
                    self.tokenized = data["tokenized"]
                    self.df = data["df"]
                    self.avgdl = data["avgdl"]
                    self.N = data["N"]
                    print(f"[BM25] Loaded {self.N} docs from {persist_path}")
            except Exception as e:
                print(f"[BM25] Load failed: {e}")

    def build(self, chunks: list[dict]) -> None:
        print(f"[BM25] Building index with {len(chunks)} chunks...")
        self.docs = chunks
        self.tokenized = []
        self.df = Counter()

        for chunk in chunks:
            text = chunk.get("content", "") if isinstance(chunk, dict) else str(chunk)
            tokens = self._tokenize(text)
            self.tokenized.append(tokens)
            for token in set(tokens):
                self.df[token] += 1

        self.N = len(self.tokenized)
        self.avgdl = sum(len(t) for t in self.tokenized) / self.N if self.N else 1.0

        if self.persist_path:
            try:
                with open(self.persist_path, "wb") as f:
                    pickle.dump({
                        "docs": self.docs,
                        "tokenized": self.tokenized,
                        "df": self.df,
                        "avgdl": self.avgdl,
                        "N": self.N
                    }, f)
                print(f"[BM25] Saved to {self.persist_path}")
            except Exception as e:
                print(f"[BM25] Save failed: {e}")

        print(f"[BM25] Built: {self.N} documents, {len(self.df)} terms")

    def search(self, query: str, top_k: int = 10) -> list[dict]:
        if self.N == 0:
            print("[BM25] Index is empty")
            return []

        q_tokens = self._tokenize(query)
        scores = []

        for tokens in self.tokenized:
            tf_map = Counter(tokens)
            dl = len(tokens)
            score = 0.0

            for term in q_tokens:
                if term not in self.df:
                    continue
                tf = tf_map.get(term, 0)
                idf = math.log((self.N - self.df[term] + 0.5) / (self.df[term] + 0.5) + 1)
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (num / den if den > 0 else 0)

            scores.append(score)

        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        results = [self.docs[i] for i, s in ranked[:top_k] if s > 0.001]

        print(f"[BM25] Found {len(results)} chunks")
        return results