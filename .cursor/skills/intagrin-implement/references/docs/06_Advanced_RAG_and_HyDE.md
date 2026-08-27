# Advanced RAG & HyDE

IntaGrin includes a zero-dependency Vector RAG engine: no external vector database required. It
chunks markdown/text/JSON documents in your `docs/` folder, embeds them via LiteLLM, and performs
cosine similarity search in-process. This is a pure-Python linear scan — fine for a project-sized
knowledge base (hundreds of chunks), not a replacement for a real vector DB at large
document-corpus scale.

### Disk-Persisted Embedding Cache
Computed embeddings are cached to `.ai/rag_cache.json`, keyed by `(file name, file mtime, chunk
index, embedding model)`. On the next run — a fresh `inta serve`/`inta worker` process, or the next
`inta dev` session — unchanged files reuse their cached embeddings instead of re-calling the
embedding API; only new or edited files pay the embedding cost. Deleting a document prunes its
entries from the cache automatically on the next index. This cache is per-project and gitignore-safe
to delete: worst case, the next index just re-embeds everything.

In your `ai.yaml`:

```yaml
rag:
  docs_dir: "docs"
  embedding_model: "text-embedding-3-small"
  top_k: 4
  chunk_size: 500
  hyde: true
```

## Hypothetical Document Embeddings (HyDE)
If you enable `hyde: true`, semantic search changes shape for abstract/complex queries: instead of
embedding the user's question directly, the framework generates a hypothetical answer to it first,
then embeds *that* to search the vector database — this can improve retrieval precision on queries
that phrase things very differently from how the source documents do.

**Caveat:** the HyDE generation call is currently hardcoded to `gpt-4o-mini` regardless of your
project's configured `model.primary`/`fallback`. If your project doesn't otherwise use OpenAI,
enabling `hyde: true` still requires an `OPENAI_API_KEY` in your environment for that one call.

## Semantic Caching
To opt LLM completions into LiteLLM caching, enable:
```yaml
model:
  use_cache: true
```
IntaGrin passes this option to LiteLLM completion calls. Configure a LiteLLM cache backend before relying on cache hits. This setting does not cache document indexing, embeddings, or RAG query results.
