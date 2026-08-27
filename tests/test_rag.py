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
