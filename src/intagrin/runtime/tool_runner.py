import json

import litellm

from intagrin.tracing.console import Tracer


class ToolRunner:
    """Handles semantic tool retrieval (lazy tool loading) for large tool schemas.

    Tool *execution* — including approval gating, transfer/delegation, and self-healing —
    lives in RuntimeEngine (runtime/engine.py), which is the only code path actually used at
    runtime; this class previously carried a second, unused copy of that dispatch logic.
    """

    @staticmethod
    async def get_active_tools(engine, agent_cfg, schemas: list[dict]):
        if getattr(agent_cfg, "lazy_load_tools", False) and len(schemas) > 5:
            recent_msgs = engine.messages[-3:] if len(engine.messages) >= 3 else engine.messages
            trajectory = "\n".join([f"{m.get('role', 'unknown')}: {m.get('content', '')}" for m in recent_msgs])

            if trajectory:
                # Debounce: an unchanged (trajectory, schema-set) pair reuses the prior selection
                # instead of re-querying the router model — cheap to check, and the trajectory is
                # frequently identical across back-to-back calls (e.g. self-healing retries).
                cache_key = (trajectory, tuple(s["function"]["name"] for s in schemas))
                cached = engine._tool_selection_cache
                if cached is not None and cached[0] == cache_key:
                    filtered_schemas = [
                        s for s in schemas if s["function"]["name"] in cached[1]
                    ]
                    if filtered_schemas:
                        Tracer.log_step(
                            "Optimizer",
                            f"Reused prior tool selection ({len(filtered_schemas)} tools) — trajectory unchanged.",
                        )
                        return filtered_schemas

                try:
                    router_model = engine.graph.config.model.fallback or engine.graph.config.model.primary or "gemini/gemini-2.5-flash"
                    tool_names_and_desc = "\n".join([f"- {s['function']['name']}: {s['function']['description']}" for s in schemas])
                    routing_prompt = f"Given this recent conversation trajectory:\n{trajectory}\n\nWhich tools from this list are likely needed by the assistant next? Return ONLY a valid JSON object with a single key 'tools' containing a list of strings (tool names).\n\nTools:\n{tool_names_and_desc}"

                    route_res = await litellm.acompletion(
                        model=router_model,
                        messages=[{"role": "user", "content": routing_prompt}],
                        response_format={"type": "json_object"},
                        temperature=0
                    )
                    content = route_res.choices[0].message.content
                    parsed = json.loads(content)
                    selected = parsed.get("tools", [])

                    if isinstance(selected, list):
                        always_include = {"transfer_agent", "delegate_task", "read_state", "write_state"}
                        selected_names = set(selected) | always_include
                        filtered_schemas = [s for s in schemas if s["function"]["name"] in selected_names]
                        if filtered_schemas:
                            engine._tool_selection_cache = (cache_key, selected_names)
                            Tracer.log_step("Optimizer", f"Lazy loaded {len(filtered_schemas)} tools (down from {len(schemas)})")
                            return filtered_schemas
                except Exception as e:
                    Tracer.log_error(f"Semantic Tool Retrieval failed: {e}")
        return schemas
