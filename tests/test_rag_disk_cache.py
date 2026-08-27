import asyncio
import json
from unittest.mock import MagicMock, patch

from intagrin.runtime.rag import VectorRAGEngine


def _fake_embedding_response(vectors):
    resp = MagicMock()
    resp.data = [{"embedding": v} for v in vectors]
    return resp


def test_unchanged_files_reuse_cached_embeddings_across_engine_instances(tmp_path):
    """Two separate VectorRAGEngine instances (simulating two process restarts) pointed at the
    same cache_dir and an unchanged docs_dir must only call the embedding API on the first
    index_documents() — the second should be a pure cache hit with zero API calls."""

    async def run():
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        (docs_dir / "a.md").write_text("alpha beta gamma")
        cache_dir = tmp_path / ".ai"

        embed_calls = []

        async def fake_aembedding(model, input):
            embed_calls.append(list(input))
            return _fake_embedding_response([[1.0, 0.0] for _ in input])

        with patch("intagrin.runtime.rag.litellm.aembedding", side_effect=fake_aembedding):
            engine1 = VectorRAGEngine(docs_dir=docs_dir, cache_dir=cache_dir)
            await engine1.index_documents()
            assert len(embed_calls) == 1, "first index should call the embedding API"

            engine2 = VectorRAGEngine(docs_dir=docs_dir, cache_dir=cache_dir)
            await engine2.index_documents()
            assert len(embed_calls) == 1, "second index of unchanged docs must be a cache hit"

        assert engine2.chunks[0]["embedding"] == [1.0, 0.0]
        assert (cache_dir / "rag_cache.json").exists()

    asyncio.run(run())


def test_editing_a_file_invalidates_only_that_files_cache_entries(tmp_path):
    """Changing one file's mtime/content must re-embed only that file's chunks, while an
    untouched sibling file's chunks stay served from cache."""

    async def run():
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        a = docs_dir / "a.md"
        b = docs_dir / "b.md"
        a.write_text("alpha content")
        b.write_text("bravo content")
        cache_dir = tmp_path / ".ai"

        embed_calls = []

        async def fake_aembedding(model, input):
            embed_calls.append(list(input))
            return _fake_embedding_response([[0.5, 0.5] for _ in input])

        with patch("intagrin.runtime.rag.litellm.aembedding", side_effect=fake_aembedding):
            engine1 = VectorRAGEngine(docs_dir=docs_dir, cache_dir=cache_dir)
            await engine1.index_documents()
            assert len(embed_calls) == 1

            import os

            a.write_text("alpha content CHANGED")
            os.utime(a, (a.stat().st_mtime + 5, a.stat().st_mtime + 5))

            engine2 = VectorRAGEngine(docs_dir=docs_dir, cache_dir=cache_dir)
            await engine2.index_documents()

        assert len(embed_calls) == 2, "only the changed file's chunk should trigger a re-embed"
        assert embed_calls[1] == ["alpha content CHANGED"]

    asyncio.run(run())


def test_cache_survives_and_is_pruned_when_a_file_is_deleted(tmp_path):
    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    a = docs_dir / "a.md"
    a.write_text("alpha content")
    cache_dir = tmp_path / ".ai"

    async def fake_aembedding(model, input):
        return _fake_embedding_response([[0.1, 0.2] for _ in input])

    async def run():
        with patch("intagrin.runtime.rag.litellm.aembedding", side_effect=fake_aembedding):
            engine1 = VectorRAGEngine(docs_dir=docs_dir, cache_dir=cache_dir)
            await engine1.index_documents()

            cache_before = json.loads((cache_dir / "rag_cache.json").read_text())
            assert len(cache_before) == 1

            a.unlink()

            engine2 = VectorRAGEngine(docs_dir=docs_dir, cache_dir=cache_dir)
            await engine2.index_documents()

        cache_after = json.loads((cache_dir / "rag_cache.json").read_text())
        assert cache_after == {}

    asyncio.run(run())
