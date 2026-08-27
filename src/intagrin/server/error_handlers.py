"""Shared FastAPI exception handler for codified IntaGrin errors.

Registered by both `server/api.py` and `server/monitor.py`. Additive to the existing error JSON
shape: `detail` stays exactly what it is today (a plain string almost everywhere, occasionally
FastAPI's own array-of-validation-errors shape) — `code` is a new sibling field, present only for
errors raised as `IntaGrinError`. Frontend code (`server/templates/monitor.html`) that reads
`err.detail` is unaffected.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from ..errors import IntaGrinError


def register_intagrin_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(IntaGrinError)
    async def _handle_intagrin_error(request: Request, exc: IntaGrinError) -> JSONResponse:
        return JSONResponse(
            status_code=exc.http_status or 400,
            content={"detail": exc.message, "code": exc.code},
        )
