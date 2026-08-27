import ast
import importlib
from typing import Any

import litellm

from intagrin.tracing.console import Tracer


def safe_eval(expr: str, state: dict, functions: dict[str, Any] | None = None):
    """Secure AST-based expression evaluator to prevent RCE in routing conditions.

    `functions` (see config.schema.ConditionFunctionConfig) is a closed, pre-registered
    name -> callable mapping — the *only* thing a call expression can ever resolve to. There is
    no fallback to builtins, imports, or attribute access, so a condition can express "call this
    one specific, project-declared predicate with these already-restricted-grammar values" and
    nothing more; it can never reach arbitrary code. Every argument expression is itself
    recursively evaluated through this exact same restricted grammar before the call happens."""
    tree = ast.parse(expr, mode="eval")
    functions = functions or {}

    def _eval(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Name):
            if node.id in state:
                return state[node.id]
            raise ValueError(f"Unknown variable: {node.id}")
        elif isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Unsupported call target: only bare function names are callable")
            func_name = node.func.id
            if func_name not in functions:
                raise ValueError(
                    f"Unknown condition function: {func_name!r} — declare it in "
                    "condition_functions to make it callable from a condition."
                )
            args = [_eval(a) for a in node.args]
            kwargs = {kw.arg: _eval(kw.value) for kw in node.keywords}
            return functions[func_name](*args, **kwargs)
        elif isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, right_node in zip(node.ops, node.comparators):
                right = _eval(right_node)
                if isinstance(op, ast.Lt) and not left < right:
                    return False
                if isinstance(op, ast.LtE) and not left <= right:
                    return False
                if isinstance(op, ast.Gt) and not left > right:
                    return False
                if isinstance(op, ast.GtE) and not left >= right:
                    return False
                if isinstance(op, ast.Eq) and not left == right:
                    return False
                if isinstance(op, ast.NotEq) and not left != right:
                    return False
                if isinstance(op, ast.In) and left not in right:
                    return False
                if isinstance(op, ast.NotIn) and left in right:
                    return False
                left = right
            return True
        elif isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(_eval(v) for v in node.values)
            if isinstance(node.op, ast.Or):
                return any(_eval(v) for v in node.values)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return not _eval(node.operand)
        raise ValueError(f"Unsupported syntax: {type(node)}")
    return _eval(tree.body)


def validate_condition_syntax(expr: str, known_functions: set[str] | None = None) -> str | None:
    """Structural-only check of a router condition string — walks the same grammar safe_eval
    supports, but checks AST node *types* only, without evaluating against any state (none is
    available at compile-time/verify-time, only at actual routing time). Returns None if every
    node in the expression is something safe_eval can evaluate, otherwise a human-readable reason.

    This exists because a condition using unsupported syntax (most commonly `state.get(...)`,
    which reads naturally but isn't supported — conditions reference state keys as bare names)
    doesn't raise where a user would notice: safe_eval's caller catches the exception, logs it,
    and treats the router as simply not firing. The condition looks configured but is silently
    dead from turn one. Shared by `inta verify` (static check) and `inta compile`'s validate-
    before-write gate, so both catch this before a user ever finds out at runtime.

    `known_functions` (names declared via config_functions — see config.schema.
    ConditionFunctionConfig) is the whitelist a call expression's function name is checked
    against; omit it (or pass an empty set) to reject every call expression, matching the
    grammar's behavior before condition_functions existed.
    """
    known_functions = known_functions or set()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return f"Not valid Python expression syntax: {e}"

    def _check(node) -> str | None:
        if isinstance(node, (ast.Constant, ast.Name)):
            return None
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                return (
                    "Unsupported call target: attribute-based calls (e.g. state.get(...)) are "
                    "not supported — only a bare function name declared under "
                    "condition_functions is callable."
                )
            if node.func.id not in known_functions:
                return (
                    f"Unknown condition function {node.func.id!r} — declare it under "
                    "condition_functions to make it callable from a condition."
                )
            for arg in [*node.args, *(kw.value for kw in node.keywords)]:
                err = _check(arg)
                if err:
                    return err
            return None
        if isinstance(node, ast.Compare):
            err = _check(node.left)
            if err:
                return err
            for comparator in node.comparators:
                err = _check(comparator)
                if err:
                    return err
            supported_ops = (
                ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq, ast.In, ast.NotIn,
            )
            for op in node.ops:
                if not isinstance(op, supported_ops):
                    return f"Unsupported comparison operator: {type(op).__name__}"
            return None
        if isinstance(node, ast.BoolOp):
            if not isinstance(node.op, (ast.And, ast.Or)):
                return f"Unsupported boolean operator: {type(node.op).__name__}"
            for value in node.values:
                err = _check(value)
                if err:
                    return err
            return None
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return _check(node.operand)
        return (
            f"Unsupported syntax: {type(node).__name__} — router conditions only support bare "
            "state-key names, literals, comparisons (<, <=, >, >=, ==, !=, in, not in), boolean "
            "logic (and/or/not), and calls to a name declared under condition_functions. "
            "Attribute access (e.g. state.get('key', default)) is never supported — reference "
            "the key directly (e.g. 'key == \"value\"')."
        )

    return _check(tree.body)


class SwarmRouter:
    """Handles all deterministic and semantic routing for the execution graph."""
    
    @staticmethod
    def evaluate_root_router(graph, active_agent_name: str, state: dict):
        """Returns (is_transferring, target_agent, error_message)"""
        root_router = graph.config.routers.get(active_agent_name)
        if not root_router:
            return False, None, None
            
        try:
            mod = importlib.import_module(root_router.module)
            if hasattr(mod, "route"):
                target = mod.route(state)
                if target in root_router.possible_targets:
                    return True, target, None
                else:
                    return False, None, f"Router returned target '{target}' not in possible_targets: {root_router.possible_targets}"
        except Exception as e:
            return False, None, str(e)
            
        return False, None, None

    @staticmethod
    def evaluate_conditional_routers(agent_cfg, state: dict, functions: dict[str, Any] | None = None):
        """Evaluates agent_cfg's conditional routers in declared order, stopping at the first one
        that fires (matching real routing precedence). Returns (fired, target, evaluations), where
        `evaluations` is the ordered list of (router, fired_bool_or_None, error_or_None) for every
        router actually checked — callers that want per-router tracing
        (RuntimeEngine._resolve_routing) can log each decision, including a genuinely broken
        condition (fired=None, error=str) rather than that router simply vanishing from the trace
        with no visible signal beyond a log line; callers that only care about the outcome (inta
        simulate) just use fired/target.

        `functions` is the project's condition_functions name -> callable mapping (see safe_eval),
        threaded through so a condition can call a declared predicate exactly like it would at any
        other evaluation site.

        A router whose condition raises (most commonly a typo'd state-key name) fails open here —
        skipped, not treated as fired — but is still recorded in `evaluations` with its error, so
        it stays visible to anything inspecting the trace instead of silently and permanently
        going dark from turn one."""
        evaluations: list[tuple[Any, bool | None, str | None]] = []
        if agent_cfg and agent_cfg.routers:
            for router in agent_cfg.routers:
                if router.condition:
                    try:
                        fired = bool(safe_eval(router.condition, state, functions))
                    except Exception as e:
                        Tracer.log_error(f"Router condition '{router.condition}' error: {e}")
                        evaluations.append((router, None, str(e)))
                        continue
                    evaluations.append((router, fired, None))
                    if fired:
                        return True, router.target, evaluations
        return False, None, evaluations

    @staticmethod
    async def evaluate_semantic_routing(agent_cfg, graph, active_agent_name: str, msg_content: str):
        if not agent_cfg or not getattr(agent_cfg, "auto_route", False) or not msg_content:
            return None
            
        available_agents = {name: cfg.description for name, cfg in graph.config.agents.items() if name != active_agent_name}
        if not available_agents:
            return None
            
        router_model = graph.config.model.fallback or graph.config.model.primary or "gemini/gemini-2.5-flash"
        routing_prompt = f"Given this message: '{msg_content}'. Who should respond next? Available agents and descriptions:\n"
        for name, desc in available_agents.items():
            routing_prompt += f"- {name}: {desc}\n"
        routing_prompt += "\nReturn EXACTLY the name of the agent, or 'NONE' if no one needs to respond. Do not include any other text."
        
        try:
            route_res = await litellm.acompletion(
                model=router_model,
                messages=[{"role": "user", "content": routing_prompt}],
                max_tokens=10,
                temperature=0
            )
            target = route_res.choices[0].message.content.strip().replace("'", "").replace('"', "")
            if target in available_agents:
                return target
        except Exception as e:
            Tracer.log_error(f"Semantic routing failed: {e}")
            
        return None

    @staticmethod
    def resolve_model(graph, requested_model: str, user_input: str) -> str:
        """Dynamic Semantic Model & Cost Router"""
        if requested_model.lower() != "auto":
            return requested_model
            
        complex_triggers = ["explain in detail", "write code", "analyze", "refactor", "architect", "compare", "debug", "step by step"]
        is_complex = len(user_input.split()) > 35 or any(trig in user_input.lower() for trig in complex_triggers)
        
        fallback = graph.config.model.fallback or ""
        if "gemini" in fallback:
            return "gemini/gemini-2.5-pro" if is_complex else "gemini/gemini-2.5-flash"
        elif "anthropic" in fallback:
            return "anthropic/claude-3-7-sonnet" if is_complex else "anthropic/claude-3-5-haiku"
        else:
            return "openai/gpt-4o" if is_complex else "openai/gpt-4o-mini"
