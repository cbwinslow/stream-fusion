"""JSON-RPC 2.0 Gateway Route for AI Agents & Remote Subagents (Spec 27)."""

import json
from typing import Any, Dict, List, Union
from fastapi import APIRouter, Request, Response

from stream_fusion.schema.agent_rpc import AgentRpcDispatcher

router = APIRouter(prefix="/api/rpc", tags=["JSON-RPC 2.0"])


@router.post("")
async def json_rpc_gateway(request: Request) -> Response:
    """Universal JSON-RPC 2.0 endpoint delegating to AgentRpcDispatcher."""
    dispatcher = getattr(request.app.state, "rpc_dispatcher", None)
    if not dispatcher:
        dispatcher = AgentRpcDispatcher()
        request.app.state.rpc_dispatcher = dispatcher

    # Wire active daemon if available
    daemon = getattr(request.app.state, "daemon", None)
    if daemon and hasattr(dispatcher, "_active_daemon"):
        dispatcher._active_daemon = daemon

    body = await request.body()
    body_str = body.decode("utf-8")

    try:
        data = json.loads(body_str)
    except json.JSONDecodeError as e:
        err_res = {
            "jsonrpc": "2.0",
            "id": None,
            "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
        }
        return Response(content=json.dumps(err_res), media_type="application/json")

    # Handle batch or single request
    if isinstance(data, list):
        results = [dispatcher.dispatch(item) for item in data]
        return Response(content=json.dumps(results), media_type="application/json")
    else:
        result = dispatcher.dispatch(data)
        return Response(content=json.dumps(result), media_type="application/json")
