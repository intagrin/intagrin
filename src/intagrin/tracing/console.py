import asyncio
import contextvars
import json
import sys
import time
import traceback
from enum import Enum
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


class LogLevel(str, Enum):
    QUIET = "quiet"
    NORMAL = "normal"
    DEBUG = "debug"
    TRACE = "trace"


_LEVEL_ORDER = {LogLevel.QUIET: 0, LogLevel.NORMAL: 1, LogLevel.DEBUG: 2, LogLevel.TRACE: 3}

# Correlates every log line with the session/agent that produced it, without threading
# session_id/agent_name through every one of Tracer's call sites — RuntimeEngine sets this once
# per turn (see runtime/engine.py) and every Tracer method reads it automatically.
_trace_context: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "_trace_context", default={}
)


def set_trace_context(session_id: str = None, agent_name: str = None) -> None:
    ctx = {}
    if session_id:
        ctx["session_id"] = session_id
    if agent_name:
        ctx["agent"] = agent_name
    _trace_context.set(ctx)


def clear_trace_context() -> None:
    _trace_context.set({})


class EventStreamer:
    subscribers: list[asyncio.Queue] = []

    @classmethod
    def subscribe(cls) -> asyncio.Queue:
        q = asyncio.Queue()
        cls.subscribers.append(q)
        return q

    @classmethod
    def unsubscribe(cls, q: asyncio.Queue):
        if q in cls.subscribers:
            cls.subscribers.remove(q)

    @classmethod
    def emit(cls, event_type: str, data: Any):
        payload = {
            "type": event_type,
            "data": data,
            "context": _trace_context.get(),
            "timestamp": time.time(),
        }
        for q in list(cls.subscribers):
            try:
                q.put_nowait(payload)
            except Exception:
                pass


class Tracer:
    """Console + SSE + structured-log sink for everything the runtime does.

    Data collection (`EventStreamer.emit`) always happens regardless of verbosity level — only
    console *printing* is level-gated — so the live monitor dashboard and `--json-logs` output
    never silently miss an event just because the terminal is set to `--log-level quiet`.
    """

    _level: LogLevel = LogLevel.NORMAL
    _json_mode: bool = False

    @classmethod
    def set_level(cls, level) -> None:
        cls._level = level if isinstance(level, LogLevel) else LogLevel(level)

    @classmethod
    def set_json_mode(cls, enabled: bool) -> None:
        cls._json_mode = enabled

    @classmethod
    def _should_print(cls, min_level: LogLevel) -> bool:
        return _LEVEL_ORDER[cls._level] >= _LEVEL_ORDER[min_level]

    @staticmethod
    def _context_tag() -> str:
        ctx = _trace_context.get()
        parts = [ctx[k] for k in ("session_id", "agent") if ctx.get(k)]
        return ":".join(parts)

    @classmethod
    def _print_json(cls, level: str, event_type: str, message: str, **extra):
        record = {
            "timestamp": time.time(),
            "level": level,
            "event": event_type,
            "message": message,
            **_trace_context.get(),
            **extra,
        }
        print(json.dumps(record, default=str))

    @staticmethod
    def log_step(step_name: str, details: str):
        EventStreamer.emit("step", {"name": step_name, "details": details})
        if Tracer._json_mode:
            Tracer._print_json("info", "step", details, step=step_name)
            return
        if not Tracer._should_print(LogLevel.NORMAL):
            return
        tag = Tracer._context_tag()
        prefix = f"[dim]\\[{tag}][/dim] " if tag else ""
        console.print(f"{prefix}[bold cyan]>[/bold cyan] [cyan]{step_name}[/cyan]: {details}")

    @staticmethod
    def log_tool_call(tool_name: str, args: dict):
        EventStreamer.emit("tool_call", {"tool": tool_name, "args": args})
        if Tracer._json_mode:
            Tracer._print_json(
                "info", "tool_call", f"Executing {tool_name}", tool=tool_name, args=args
            )
            return
        if not Tracer._should_print(LogLevel.NORMAL):
            return
        tag = Tracer._context_tag()
        prefix = f"[dim]\\[{tag}][/dim] " if tag else ""
        console.print(
            f"{prefix}[bold magenta]⚙️[/bold magenta] [magenta]Executing Tool:[/magenta] {tool_name} with {args}"
        )

    @staticmethod
    def log_tool_result(result: str):
        EventStreamer.emit("tool_result", {"result": result})
        if Tracer._json_mode:
            Tracer._print_json("info", "tool_result", str(result))
            return
        if not Tracer._should_print(LogLevel.NORMAL):
            return
        tag = Tracer._context_tag()
        prefix = f"[dim]\\[{tag}][/dim] " if tag else ""
        console.print(f"{prefix}[bold green]✓[/bold green] [green]Tool Result:[/green] {result}")

    @staticmethod
    def log_error(error: str, exc_info: bool = None, state: dict = None):
        """Logs an error. Always prints, regardless of --log-level — only the amount of detail
        changes with level/json-mode.

        `exc_info=None` (the default) auto-detects: if this is called from inside an `except`
        block, the real traceback is captured automatically via `sys.exc_info()` — callers no
        longer need to remember to pass `exc_info=True`. Pass `exc_info=False` to force it off, or
        `exc_info=True` to force it on outside an except block (rare).
        """
        capture_tb = exc_info if exc_info is not None else (sys.exc_info()[0] is not None)
        tb_text = traceback.format_exc() if capture_tb and sys.exc_info()[0] is not None else None

        safe_state = None
        if state:
            safe_state = {k: v for k, v in state.items() if not k.startswith("_")}

        EventStreamer.emit(
            "error", {"message": error, "traceback": tb_text, "state": safe_state}
        )

        if Tracer._json_mode:
            Tracer._print_json("error", "error", error, traceback=tb_text, state=safe_state)
            return

        content = error
        if tb_text:
            content += f"\n\n[bold]Traceback:[/bold]\n{tb_text}"
        if safe_state:
            content += f"\n\n[bold]Engine State at Crash:[/bold]\n{json.dumps(safe_state, indent=2, default=str)}"

        tag = Tracer._context_tag()
        title = "[bold red]Critical Engine Error[/bold red]" + (
            f" [dim]\\[{tag}][/dim]" if tag else ""
        )
        console.print(Panel(content, title=title, border_style="red"))

    @staticmethod
    def log_cost(tokens: int, cost: float):
        EventStreamer.emit("cost", {"tokens": tokens, "cost": cost})
        if Tracer._json_mode:
            Tracer._print_json(
                "info", "cost", f"{tokens} tokens, ${cost:.6f}", tokens=tokens, cost=cost
            )
            return
        if not Tracer._should_print(LogLevel.NORMAL):
            return
        table = Table(show_header=True, header_style="bold yellow")
        table.add_column("Metric")
        table.add_column("Value")
        table.add_row("Tokens Used", str(tokens))
        table.add_row("Estimated Cost", f"${cost:.6f}")
        console.print(table)

    @staticmethod
    def log_llm_exchange(model: str, prompt_messages: list, response_content: str):
        """Only active at --log-level trace: logs the full outbound prompt and raw completion for
        prompt-engineering debugging (not just crash debugging). Emitted regardless of level so a
        --json-logs consumer or the dashboard can always capture it; console printing is gated."""
        EventStreamer.emit(
            "llm_exchange",
            {"model": model, "prompt": prompt_messages, "response": response_content},
        )
        if not Tracer._should_print(LogLevel.TRACE):
            return
        if Tracer._json_mode:
            Tracer._print_json(
                "trace",
                "llm_exchange",
                f"LLM call to {model}",
                model=model,
                prompt=prompt_messages,
                response=response_content,
            )
            return
        tag = Tracer._context_tag()
        last_msg = prompt_messages[-1] if prompt_messages else ""
        console.print(
            Panel(
                f"[bold]Model:[/bold] {model}\n\n[bold]Last prompt message:[/bold]\n{last_msg}\n\n[bold]Response:[/bold]\n{response_content}",
                title=f"[dim]LLM Exchange {(chr(91) + tag + chr(93)) if tag else ''}[/dim]",
                border_style="dim",
            )
        )

    @staticmethod
    def log_router_decision(
        router_kind: str,
        description: str,
        state: dict,
        fired: bool,
        target: str = None,
        error: str | None = None,
    ):
        """Logs a routing decision (fired or not) with the state values it was evaluated against
        — answers "why didn't this route to X" without needing print statements in ai.yaml-
        referenced Python. Always emitted; only printed at --log-level debug+ to avoid noise on
        every turn for swarms with many routers that don't fire.

        `error`, when set, means the condition itself raised (most commonly a typo'd state-key
        name) rather than simply evaluating to False — surfaced in the same structured event
        (Monitor's live trace, `inta simulate`'s reconstruction) instead of only a separate log
        line a user has to go looking for, since a broken condition otherwise looks identical to
        one that's just never true yet."""
        safe_state = {k: v for k, v in (state or {}).items() if not k.startswith("_")}
        EventStreamer.emit(
            "router_decision",
            {
                "kind": router_kind,
                "description": description,
                "fired": fired,
                "target": target,
                "state": safe_state,
                "error": error,
            },
        )
        if not Tracer._should_print(LogLevel.DEBUG):
            return
        if Tracer._json_mode:
            Tracer._print_json(
                "debug",
                "router_decision",
                description,
                kind=router_kind,
                fired=fired,
                target=target,
                state=safe_state,
                error=error,
            )
            return
        tag = Tracer._context_tag()
        prefix = f"[dim]\\[{tag}][/dim] " if tag else ""
        if error:
            status = f"[bold red]ERROR: {error}[/bold red]"
        elif fired:
            status = f"[green]FIRED -> {target}[/green]"
        else:
            status = "[dim]did not fire[/dim]"
        console.print(
            f"{prefix}[bold blue]router[/bold blue] ({router_kind}) {description}: {status} "
            f"[dim]state={json.dumps(safe_state, default=str)}[/dim]"
        )
