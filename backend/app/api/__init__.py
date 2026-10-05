"""REST + WebSocket API (mounted under /api/v1)."""
from fastapi import APIRouter

from . import dataset, diagnostics, field, ingest, optimizer, system, thermal, wells, ws

api_router = APIRouter(prefix="/api/v1")
for module in (system, dataset, field, wells, thermal, diagnostics, optimizer, ingest):
    api_router.include_router(module.router)

ws_router = ws.router

__all__ = ["api_router", "ws_router"]
