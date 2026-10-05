"""WebSocket live stream: ``/ws/stream``.

Server -> client messages: ``tick`` (all wells' compact state + KPIs), ``dyno`` (cards, only for
subscribed wells), ``alert`` / ``alert_cleared``, ``advisory``, ``job``, ``ingest``.
Client -> server: ``{"action": "subscribe", "wells": ["BGW-01"]}`` or ``{"action": "ping"}``.
"""
from __future__ import annotations

import json
import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter()
MAX_CLIENT_MSG = 2048
MAX_CLIENTS = 200


def _origin_allowed(ws: WebSocket, allowed: list[str], pattern: str | None = None) -> bool:
    origin = ws.headers.get("origin")
    if not origin:  # non-browser clients (scripts, tests) do not send one
        return True
    if origin in allowed:
        return True
    if pattern and re.fullmatch(pattern, origin):
        return True
    host = ws.headers.get("host", "")
    return origin.split("://", 1)[-1] == host  # same-origin (served behind the nginx proxy)


@router.websocket("/ws/stream")
async def stream(ws: WebSocket) -> None:
    c = ws.app.state.container
    if not _origin_allowed(ws, c.settings.cors_origin_list, c.settings.cors_origin_regex) or c.hub.n_clients >= MAX_CLIENTS:
        await ws.close(code=1008)
        return
    hub = c.hub
    await hub.connect(ws)
    await ws.send_text(json.dumps({"type": "hello", "control_mode": c.settings.control_mode, "sim_ts": int(c.runtime.sim_ts)}))
    try:
        while True:
            raw = await ws.receive_text()
            if len(raw) > MAX_CLIENT_MSG:
                await ws.close(code=1009)
                return
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("action") == "subscribe" and isinstance(msg.get("wells"), list):
                wells = [str(w).upper() for w in msg["wells"] if str(w).upper() in c.twin.fleet]
                hub.subscribe(ws, wells)
                await ws.send_text(json.dumps({"type": "subscribed", "wells": wells}))
                for w in wells:  # send the latest card at once so the panel is never empty
                    card = c.runtime.latest_card(w)
                    if card:
                        await ws.send_text(json.dumps({"type": "dyno", **card}, separators=(",", ":")))
            elif msg.get("action") == "ping":
                await ws.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    finally:
        hub.disconnect(ws)
