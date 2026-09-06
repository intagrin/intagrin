import json

import litellm

from intagrin.tracing.console import Tracer

_ALWAYS_INCLUDE_TOOLS = {"transfer_agent", "delegate_task", "read_state", "write_state"}

# Minimum tools the embedding fast path will ever keep, even if only one tool clears the
# similarity threshold — a too-generous tool set costs a few extra schema tokens; a too-narrow
# one silently withholds the one tool the model actually needed. The LLM-based path has no
# equivalent floor since it's asked directly and can legitimately answer "none of these", which
# the embedding heuristic isn't trusted to decide on its own.
_EMBEDDING_SELECTION_FLOOR = 3


class ToolRunner:
    """Handles semantic tool retrieval (lazy tool loading) for large tool schemas.

    Tool *execution* — including approval gating, transfer/delegation, and self-healing —
    lives in RuntimeEngine (runtime/engine.py), which is the only code path actually used at
    runtime; this class previously carried a second, unused copy of that dispatch logic.
    """

    @staticmethod
    async def get_active_tools(engine, agent_cfg, schemas: list[dict]):
        if not getattr(agent_cfg, "lazy_load_tools", False) or len(schemas) <= 5:
            return schemas

        recent_msgs = engine.messages[-3:] if len(engine.messages) >= 3 else engine.messages
        trajectory = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in recent_msgs])
        if not trajectory:
            return schemas

        # Debounce: an unchanged (trajectory, schema-set) pair reuses the prior selection
        # instead of re-querying the router model — cheap to check, and the trajectory is
        # frequently identical across back-to-back calls (e.g. self-healing retries).
        cache_key = (trajectory, tuple(s["function"]["name"] for s in schemas))
        cached = engine._tool_selection_cache
        if cached is not None and cached[0] == cache_key:
            filtered_schemas = [s for s in schemas if s["function"]["name"] in cached[1]]
            if filtered_schemas:
                Tracer.log_step(
                    "Optimizer",
                    f"Reused prior tool selection ({len(filtered_schemas)} tools) — trajectory unchanged.",
                )
                return filtered_schemas

        # Fast path first: a cosine-similarity gate over embeddings, no LLM completion call at
        # all. Falls back to the original LLM-based selection (unchanged below) on any failure —
        # missing embedding-provider credentials, an unresolvable embedding model, a network
        # error — so this is purely additive: worst case, behavior is identical to before this
        # existed, never worse.
        selected_names = await ToolRunner._select_tools_by_embedding(engine, schemas, trajectory)
        source = "embedding"
        if selected_names is None:
            selected_names = await ToolRunner._select_tools_by_llm(engine, schemas, trajectory)
            source = "LLM"

        if selected_names is not None:
            selected_names = selected_names | _ALWAYS_INCLUDE_TOOLS
            filtered_schemas = [s for s in schemas if s["function"]["name"] in selected_names]
            if filtered_schemas:
                engine._tool_selection_cache = (cache_key, selected_names)
                Tracer.log_step(
                    "Optimizer",
                    f"Lazy loaded {len(filtered_schemas)} tools (down from {len(schemas)}, via {source})",
                )
                return filtered_schemas

        return schemas

    @staticmethod
    async def _select_tools_by_embedding(engine, schemas: list[dict], trajectory: str) -> set[str] | None:
        """Cosine-similarity gate over each tool's own name+description embedding, reusing the
        same litellm.aembedding wrapper RAG/episodic memory already use (episodic_memory.py's
        embed_text, which never raises — returns None on any provider failure). Returns None
        (never raises) whenever it can't produce a confident selection, so the caller falls back
        to the LLM-based path exactly as before this existed.

        Embedding model preference mirrors whatever the project already told us about its
        embedding provider — rag.embedding_model, then episodic_memory.embedding_model, then the
        same "text-embedding-3-small" default those two features already use — rather than a new
        config field, so a project using neither still gets a sensible default and a project
        using either doesn't end up calling two different embedding providers.

        Tool embeddings are computed once per (embedding_model, tool-name-set) and cached on the
        engine instance (engine._tool_embedding_cache) — tool schemas are static within a
        session, only the trajectory needs a fresh embedding each call.
        """
        try:
            from .episodic_memory import embed_text
            from .rag import cosine_similarity

            embedding_model = (
                (engine.graph.config.rag.embedding_model if engine.graph.config.rag else None)
                or (
                    engine.graph.config.episodic_memory.embedding_model
                    if engine.graph.config.episodic_memory
                    else None
                )
                or "text-embedding-3-small"
            )

            tool_key = tuple(sorted(s["function"]["name"] for s in schemas))
            model_cache = engine._tool_embedding_cache.setdefault(embedding_model, {})
            tool_embeddings: dict[str, list[float]] = model_cache.get(tool_key, {})
            if not tool_embeddings:
                fresh: dict[str, list[float]] = {}
                for s in schemas:
                    text = f"{s['function']['name']}: {s['function'].get('description', '')}"
                    emb = await embed_text(embedding_model, text)
                    if emb is None:
                        return None  # embedding provider unavailable — let the LLM path handle it
                    fresh[s["function"]["name"]] = emb
                tool_embeddings = fresh
                model_cache[tool_key] = tool_embeddings

            trajectory_embedding = await embed_text(embedding_model, trajectory)
            if trajectory_embedding is None:
                return None

            scored = sorted(
                ((name, cosine_similarity(trajectory_embedding, emb)) for name, emb in tool_embeddings.items()),
                key=lambda pair: pair[1],
                reverse=True,
            )
            if not scored:
                return None

            top_score = scored[0][1]
            spread = top_score - scored[-1][1]
            # Threshold relative to this call's own score spread, not a fixed cosine value —
            # "what counts as similar" varies by embedding model/provider. When every tool
            # scores about the same (spread ~0), that's the gate genuinely unable to
            # discriminate, not a signal to exclude everything — the 0.02 floor keeps the
            # threshold from collapsing to "top score only" in that case.
            threshold = top_score - max(spread * 0.5, 0.02)
            selected = {name for name, score in scored if score >= threshold}

            if len(selected) < min(_EMBEDDING_SELECTION_FLOOR, len(scored)):
                selected = {name for name, _ in scored[:_EMBEDDING_SELECTION_FLOOR]}

            return selected
        except Exception as e:
            Tracer.log_error(f"Embedding-based tool selection failed: {e}")
            return None

    @staticmethod
    async def _select_tools_by_llm(engine, schemas: list[dict], trajectory: str) -> set[str] | None:
        """Original LLM-based semantic tool retrieval — the fallback when the embedding fast
        path (_select_tools_by_embedding) can't run. Returns an empty set (not None) when the
        model explicitly decides no tool beyond the always-include set is needed — only a real
        failure (network/parse error) returns None, distinct from a deliberate empty answer."""
        try:
            router_model = engine.graph.config.model.fallback or engine.graph.config.model.primary or "gemini/gemini-2.5-flash"
            tool_names_and_desc = "\n".join([f"- {s['function']['name']}: {s['function']['description']}" for s in schemas])
            routing_prompt = f"Given this recent conversation trajectory:\n{trajectory}\n\nWhich tools from this list are likely needed by the assistant next? Return ONLY a valid JSON object with a single key 'tools' containing a list of strings (tool names).\n\nTools:\n{tool_names_and_desc}"

            route_res = await litellm.acompletion(
                model=router_model,
                messages=[{"role": "user", "content": routing_prompt}],
                response_format={"type": "json_object"},
                temperature=0,
            )
            content = route_res.choices[0].message.content
            parsed = json.loads(content)
            selected = parsed.get("tools", [])
            if isinstance(selected, list):
                return set(selected)
            return None
        except Exception as e:
            Tracer.log_error(f"Semantic Tool Retrieval failed: {e}")
            return None
