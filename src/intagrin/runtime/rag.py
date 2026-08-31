import asyncio
import json
import math
from pathlib import Path
from typing import Any

import litellm


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    dot = sum(a * b for a, b in zip(v1, v2))
    norm_a = math.sqrt(sum(a * a for a in v1))
    norm_b = math.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _rank_by_embedding(
    query_embedding: list[float], chunks: list[dict[str, Any]], top_k: int
) -> list[tuple[float, dict[str, Any]]]:
    """Ranks chunks by cosine similarity to query_embedding, returning the top_k highest-scoring
    (score, chunk) pairs. A brute-force, pure-Python O(n·d) scan over every chunk — fine for a
    knowledge base of a few hundred/thousand chunks, the ceiling for a project without a real
    vector index (out of scope here — see the module docstring). Computes the query vector's own
    norm once, up front, rather than recomputing it on every chunk the way calling
    cosine_similarity(query_embedding, chunk_embedding) in a loop would — cosine_similarity itself
    stays a simple single-pair function for other single-pair callers (e.g.
    episodic_memory.py's semantic_search_episodes, capped at 200 candidates)."""
    scored: list[tuple[float, dict[str, Any]]] = []
    query_norm = math.sqrt(sum(q * q for q in query_embedding))
    if query_norm == 0:
        return scored
    for c in chunks:
        embedding = c.get("embedding")
        if not embedding:
            continue
        dot = sum(a * b for a, b in zip(query_embedding, embedding))
        chunk_norm = math.sqrt(sum(e * e for e in embedding))
        score = dot / (query_norm * chunk_norm) if chunk_norm else 0.0
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


def _rank_by_keywords(
    query: str, chunks: list[dict[str, Any]], top_k: int
) -> list[tuple[float, dict[str, Any]]]:
    """Fallback ranking when no query embedding is available (embedding API failure) — plain
    keyword-overlap count, same O(n) scan shape as _rank_by_embedding."""
    query_words = set(query.lower().split())
    scored = [(sum(1 for w in query_words if w in c["text"].lower()), c) for c in chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


class VectorRAGEngine:
    """
    Lightweight, zero-bloat declarative Vector RAG Engine for IntaGrin.
    Chunks local markdown/text documents in `docs/`, computes embeddings via LiteLLM,
    and performs semantic cosine search without external vector DB dependencies.
    """

    def __init__(
        self,
        docs_dir: Path,
        embedding_model: str = "text-embedding-3-small",
        top_k: int = 4,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        hyde: bool = False,
        cache_dir: Path | None = None,
    ):
        self.docs_dir = docs_dir
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.hyde = hyde
        self.chunks: list[dict[str, Any]] = []
        self._indexed = False
        # On-disk embedding cache, keyed by (file name, mtime, chunk index, model) so unchanged
        # files/chunks reuse their embedding across process restarts instead of re-calling the
        # embedding API every time the engine boots.
        self._cache_path = (cache_dir / "rag_cache.json") if cache_dir else None
        self._disk_cache: dict[str, list[float]] = {}

    def _load_disk_cache(self):
        if not self._cache_path or not self._cache_path.exists():
            return
        try:
            self._disk_cache = json.loads(self._cache_path.read_text())
        except Exception:
            self._disk_cache = {}

    def _save_disk_cache(self):
        if not self._cache_path:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(json.dumps(self._disk_cache))
        except Exception:
            pass

    def _chunk_text(self, text: str, source: str) -> list[dict[str, Any]]:
        chunks = []
        words = text.split()
        if not words:
            return []

        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk_words = words[i : i + self.chunk_size]
            chunk_str = " ".join(chunk_words)
            if chunk_str.strip():
                chunks.append({"text": chunk_str, "source": source, "embedding": None})
        return chunks

    async def index_documents(self):
        if self._indexed or not self.docs_dir.exists():
            return

        self._load_disk_cache()

        all_chunks = []
        for ext in ["*.md", "*.txt", "*.json"]:
            for f in self.docs_dir.glob(ext):
                if f.is_file():
                    try:
                        mtime = f.stat().st_mtime
                        content = f.read_text(encoding="utf-8", errors="ignore")
                        file_chunks = self._chunk_text(content, source=f.name)
                        for idx, c in enumerate(file_chunks):
                            c["_cache_key"] = (
                                f"{f.name}:{mtime}:{idx}:{self.embedding_model}"
                            )
                            c["embedding"] = self._disk_cache.get(c["_cache_key"])
                        all_chunks.extend(file_chunks)
                    except Exception:
                        pass

        if not all_chunks:
            if self._disk_cache:
                self._disk_cache = {}
                self._save_disk_cache()
            self._indexed = True
            return

        # Only call the embedding API for chunks that weren't already in the on-disk cache
        # (new files, or files whose mtime changed since they were last indexed).
        to_embed = [c for c in all_chunks if c["embedding"] is None]
        if to_embed:
            try:
                texts = [c["text"] for c in to_embed]
                batch_size = 100
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i : i + batch_size]
                    resp = await litellm.aembedding(
                        model=self.embedding_model, input=batch_texts
                    )
                    for idx, data in enumerate(resp.data):
                        chunk = to_embed[i + idx]
                        chunk["embedding"] = data["embedding"]
                        self._disk_cache[chunk["_cache_key"]] = chunk["embedding"]
            except Exception:
                # Fallback to simple keyword search for the unembedded chunks below
                pass

        # Drop cache entries for files/chunks that no longer exist so the cache file doesn't
        # grow forever as documents are edited or removed.
        active_keys = {c["_cache_key"] for c in all_chunks}
        if set(self._disk_cache) - active_keys:
            self._disk_cache = {
                k: v for k, v in self._disk_cache.items() if k in active_keys
            }
        self._save_disk_cache()

        embedded = [c for c in all_chunks if c["embedding"] is not None]
        self.chunks = embedded or all_chunks

        self._indexed = True

    async def search(self, query: str) -> str:
        """Search indexed documents semantically and return formatted context."""
        await self.index_documents()
        if not self.chunks:
            return "No knowledge base documents found in the docs directory."

        search_query = query
        if self.hyde:
            try:
                # Generate Hypothetical Document Embeddings for advanced semantic matching
                hyde_prompt = f"Write a factual passage that directly answers or relates to this exact query, to be used for semantic similarity search:\n'{query}'"
                hyde_resp = await litellm.acompletion(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": hyde_prompt}],
                )
                if hyde_resp.choices[0].message.content:
                    search_query = hyde_resp.choices[0].message.content
            except Exception:
                pass

        # Compute query embedding
        query_embedding = None
        try:
            resp = await litellm.aembedding(
                model=self.embedding_model, input=[search_query]
            )
            query_embedding = resp.data[0]["embedding"]
        except Exception:
            pass

        # Both ranking passes are a pure-Python O(n) scan over every indexed chunk — run off the
        # event loop thread so a large knowledge base can't stall every other in-flight request
        # on this process for the duration of one search() call.
        if query_embedding:
            top_results = await asyncio.to_thread(
                _rank_by_embedding, query_embedding, self.chunks, self.top_k
            )
        else:
            top_results = await asyncio.to_thread(
                _rank_by_keywords, query, self.chunks, self.top_k
            )
        if not top_results or top_results[0][0] == 0:
            return f"No relevant context found in documents for query: '{query}'."

        formatted = []
        for rank, (score, chunk) in enumerate(top_results, 1):
            formatted.append(
                f"[{rank}] Source: {chunk['source']} (Relevance: {score:.2f})\n{chunk['text']}"
            )

        return "\n\n---\n\n".join(formatted)
