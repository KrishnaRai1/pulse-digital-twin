"""WebSocket fan-out hub.

Producers (the simulator tick, job workers) call :meth:`publish` from any thread; the hub
schedules delivery on the server event loop. ``dyno`` messages carry large arrays, so
clients receive them only for wells they subscribed to; every other message type goes to
everybody.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

log = logging.getLogger("pulse.stream")
SEND_TIMEOUT_S = 2.0


class StreamHub:
    def __init__(self) -> None:
        self._clients: dict[WebSocket, set[str]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    @property
    def n_clients(self) -> int:
        return len(self._clients)

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._clients[ws] = set()

    def disconnect(self, ws: WebSocket) -> None:
        self._clients.pop(ws, None)

    def subscribe(self, ws: WebSocket, wells: list[str]) -> None:
        if ws in self._clients:
            self._clients[ws] = {w for w in wells[:50] if isinstance(w, str) and len(w) <= 16}

    async def broadcast(self, message: dict[str, Any]) -> None:
        if not self._clients:
            return
        text = json.dumps(message, separators=(",", ":"), default=float)
        well = message.get("well_id")
        is_dyno = message.get("type") == "dyno"
        dead: list[WebSocket] = []
        for ws, subs in list(self._clients.items()):
            if is_dyno and well not in subs:
                continue
            try:
                await asyncio.wait_for(ws.send_text(text), SEND_TIMEOUT_S)
            except Exception:  # slow or closed client: drop it, never block the producer
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    def publish(self, message: dict[str, Any]) -> None:
        """Thread-safe fire-and-forget."""
        if self._loop is None or not self._clients:
            return
        try:
            asyncio.run_coroutine_threadsafe(self.broadcast(message), self._loop)
        except RuntimeError:  # loop closed during shutdown
            pass
