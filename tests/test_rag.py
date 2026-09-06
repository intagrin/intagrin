import asyncio
import tempfile
from pathlib import Path

import pytest

from intagrin.runtime.rag import VectorRAGEngine, cosine_similarity


def test_cosine_similarity():
    v1 = [1.0, 0.0, 0.0]
    v2 = [1.0, 0.0, 0.0]
    assert cosine_similarity(v1, v2) == pytest.approx(1.0)
    
    v3 = [0.0, 1.0, 0.0]
    assert cosine_similarity(v1, v3) == pytest.approx(0.0)

def test_vector_rag_keyword_fallback():
    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir)
            (docs_dir / "cloud_guide.md").write_text("Kubernetes cluster setup instructions and deployment manifests.")
            (docs_dir / "billing_faq.txt").write_text("Invoices are processed on the first day of every month.")
            
            rag = VectorRAGEngine(docs_dir=docs_dir, top_k=2)
            results = await rag.search("Kubernetes deployment")
            
            assert "Kubernetes cluster setup" in results
            assert "cloud_guide.md" in results
            
    asyncio.run(_run())

def test_hyde_retrieval():
    """Test that if hyde=True, the RAG engine queries the LLM for a hypothetical answer before embedding."""
    from unittest.mock import AsyncMock, patch
    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir)
            (docs_dir / "doc.md").write_text("dummy")
            
            rag = VectorRAGEngine(docs_dir=docs_dir, hyde=True)
            
            mock_resp = type('obj', (object,), {'choices': [type('obj2', (object,), {'message': type('obj3', (object,), {'content': 'Hypothetical Answer'})})]})()
            
            # Mock litellm.acompletion to return a hypothetical answer
            with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
                mock_acompletion.return_value = mock_resp
                with patch("litellm.aembedding", new_callable=AsyncMock) as mock_aembedding:
                    mock_aembedding.return_value = type('obj', (object,), {'data': [{'embedding': [0.1, 0.2]}]})()
                    await rag.search("Complex query")
                    
                    # Assert that acompletion was called to generate the HyDE answer
                    assert mock_acompletion.called
                    # Assert that aembedding was called with the hypothetical answer, not the raw query
                    mock_aembedding.assert_called_with(model=rag.embedding_model, input=["Hypothetical Answer"])

    asyncio.run(_run())


def test_rank_by_embedding_matches_naive_per_pair_cosine_similarity():
    """_rank_by_embedding avoids recomputing the query vector's own norm on every chunk (unlike
    calling cosine_similarity(query, chunk) in a loop) — this proves that optimization produces
    identical scores/ordering to the naive per-pair approach it replaced, not just "doesn't
    crash". Random-ish, non-trivial vectors (not just axis-aligned unit vectors) so a sign or
    normalization mistake in the optimized path would actually show up as a mismatch."""
    from intagrin.runtime.rag import _rank_by_embedding, cosine_similarity

    query = [0.3, -0.5, 0.8, 0.1]
    chunks = [
        {"text": "a", "embedding": [0.9, 0.1, -0.2, 0.4]},
        {"text": "b", "embedding": [0.3, -0.5, 0.8, 0.1]},  # identical to query -> score 1.0
        {"text": "c", "embedding": [-0.3, 0.5, -0.8, -0.1]},  # opposite -> score -1.0
        {"text": "d", "embedding": None},  # must be skipped, not crash
        {"text": "e", "embedding": [0.0, 0.0, 0.0, 0.0]},  # zero vector -> score 0.0, not a crash
    ]

    naive = sorted(
        (
            (cosine_similarity(query, c["embedding"]), c)
            for c in chunks
            if c.get("embedding")
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    optimized = _rank_by_embedding(query, chunks, top_k=10)

    assert len(optimized) == len(naive) == 4  # the None-embedding chunk is excluded from both
    for (naive_score, naive_chunk), (opt_score, opt_chunk) in zip(naive, optimized):
        assert naive_chunk["text"] == opt_chunk["text"]
        assert opt_score == pytest.approx(naive_score, abs=1e-9)

    assert optimized[0][1]["text"] == "b"
    assert optimized[0][0] == pytest.approx(1.0)
    assert optimized[-1][1]["text"] == "c"
    assert optimized[-1][0] == pytest.approx(-1.0)


def test_rank_by_embedding_respects_top_k():
    from intagrin.runtime.rag import _rank_by_embedding

    query = [1.0, 0.0]
    chunks = [{"text": str(i), "embedding": [1.0 - i * 0.01, i * 0.01]} for i in range(20)]
    top = _rank_by_embedding(query, chunks, top_k=3)
    assert len(top) == 3
    assert [c["text"] for _, c in top] == ["0", "1", "2"]


def test_search_offloads_ranking_to_a_thread(monkeypatch):
    """Regression test: the O(n) ranking scan used to run directly on the event loop thread
    inside async def search(). Proves it's now dispatched via asyncio.to_thread by checking the
    ranking helper actually executes on a different thread than the one running the test's event
    loop."""
    import asyncio
    import tempfile
    import threading

    from unittest.mock import AsyncMock, patch

    from intagrin.runtime.rag import VectorRAGEngine

    async def _run():
        with tempfile.TemporaryDirectory() as tmpdir:
            docs_dir = Path(tmpdir)
            (docs_dir / "doc.md").write_text("some content to chunk")

            rag = VectorRAGEngine(docs_dir=docs_dir)
            main_thread = threading.current_thread()
            seen_threads = []

            real_rank = __import__("intagrin.runtime.rag", fromlist=["_rank_by_embedding"])._rank_by_embedding

            def spy(*args, **kwargs):
                seen_threads.append(threading.current_thread())
                return real_rank(*args, **kwargs)

            with patch("intagrin.runtime.rag._rank_by_embedding", side_effect=spy), patch(
                "litellm.aembedding", new_callable=AsyncMock
            ) as mock_aembedding:
                mock_aembedding.return_value = type(
                    "obj", (object,), {"data": [{"embedding": [0.1, 0.2]}]}
                )()
                await rag.search("query")

            assert len(seen_threads) == 1
            assert seen_threads[0] is not main_thread

    asyncio.run(_run())
