"""Agent Communication Protocol & JSON-RPC 2.0 Dispatcher (Spec 16)."""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from stream_fusion.schema.envelope import StreamEventType, StreamFusionEnvelope
from stream_fusion.schema.registry import default_schema_registry, SchemaRegistry
from stream_fusion.schema.adapters import SqliteJsonStore


class AgentRpcDispatcher:
    """JSON-RPC 2.0 Dispatcher for AI Agents (Antigravity subagents, LangGraph, AutoGen)."""

    def __init__(
        self,
        store: Optional[SqliteJsonStore] = None,
        registry: Optional[SchemaRegistry] = None,
    ):
        self.store = store or SqliteJsonStore()
        self.registry = registry or default_schema_registry
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {}
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        self.register_handler("streamfusion.ping", self._handle_ping)
        self.register_handler("streamfusion.listSchemas", self._handle_list_schemas)
        self.register_handler("streamfusion.getJsonSchema", self._handle_get_json_schema)
        self.register_handler("streamfusion.queryEvents", self._handle_query_events)
        self.register_handler("streamfusion.getStreamSummary", self._handle_get_stream_summary)
        self.register_handler("streamfusion.queryClaims", self._handle_query_claims)
        self.register_handler("streamfusion.queryHighlights", self._handle_query_highlights)
        self.register_handler("streamfusion.queryChatters", self._handle_query_chatters)

    def register_handler(
        self, method: str, handler: Callable[[Dict[str, Any]], Any]
    ) -> None:
        """Registers a custom RPC method handler."""
        self._handlers[method] = handler

    def handle_line(self, line: str) -> str:
        """Processes a single JSON string line and returns the JSON-RPC response string."""
        line = line.strip()
        if not line:
            return ""

        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {str(e)}"},
            })

        resp = self.handle_request(req)
        return json.dumps(resp)

    def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Dispatches a single JSON-RPC 2.0 request dict."""
        req_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {}) or {}

        if not isinstance(request, dict) or request.get("jsonrpc") != "2.0" or not method:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32600, "message": "Invalid Request: must be JSON-RPC 2.0"},
            }

        handler = self._handlers.get(method)
        if not handler:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: '{method}'"},
            }

        try:
            result = handler(params)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": result,
            }
        except TypeError as te:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": f"Invalid params: {str(te)}"},
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32603, "message": f"Internal error: {str(e)}"},
            }

    # -------------------------------------------------------------------------
    # Built-in RPC Handlers
    # -------------------------------------------------------------------------
    def _handle_ping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": "pong",
            "server_time_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": "1.0",
        }

    def _handle_list_schemas(self, params: Dict[str, Any]) -> List[Dict[str, str]]:
        return self.registry.list_schemas()

    def _handle_get_json_schema(self, params: Dict[str, Any]) -> Dict[str, Any]:
        schema_name = params.get("schema_name")
        if not schema_name:
            raise ValueError("Parameter 'schema_name' is required.")
        return self.registry.export_json_schema(schema_name)

    def _handle_query_events(self, params: Dict[str, Any]) -> Dict[str, Any]:
        stream_id = params.get("stream_id")
        event_type_str = params.get("event_type")
        event_type = StreamEventType(event_type_str) if event_type_str else None
        filters = params.get("filters", {})
        limit = int(params.get("limit", 100))
        offset = int(params.get("offset", 0))

        events = self.store.query_events(
            stream_id=stream_id,
            event_type=event_type,
            json_filters=filters,
            limit=limit,
            offset=offset,
        )
        total_count = self.store.count_events(stream_id=stream_id, event_type=event_type)

        return {
            "total_count": total_count,
            "returned_count": len(events),
            "events": [e.model_dump() for e in events],
        }

    def _handle_get_stream_summary(self, params: Dict[str, Any]) -> Dict[str, Any]:
        stream_id = params.get("stream_id")
        if not stream_id:
            raise ValueError("Parameter 'stream_id' is required.")

        total_events = self.store.count_events(stream_id=stream_id)
        slices = self.store.count_events(
            stream_id=stream_id, event_type=StreamEventType.FUSION_SLICE
        )
        highlights = self.store.count_events(
            stream_id=stream_id, event_type=StreamEventType.HIGHLIGHT_MOMENT
        )
        claims = self.store.count_events(
            stream_id=stream_id, event_type=StreamEventType.CLAIM_EXTRACTED
        )
        sponsors = self.store.count_events(
            stream_id=stream_id, event_type=StreamEventType.SPONSOR_DETECTED
        )
        safety_alerts = self.store.count_events(
            stream_id=stream_id, event_type=StreamEventType.SAFETY_ALERT
        )

        return {
            "stream_id": stream_id,
            "total_events": total_events,
            "counts": {
                "fusion_slices": slices,
                "highlights": highlights,
                "claims": claims,
                "sponsors": sponsors,
                "safety_alerts": safety_alerts,
            },
        }

    def _handle_query_claims(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        stream_id = params.get("stream_id")
        filters = {}
        if "stance" in params:
            filters["stance"] = params["stance"]
        if "subject_entity" in params:
            filters["subject_entity"] = params["subject_entity"]

        limit = int(params.get("limit", 50))
        events = self.store.query_events(
            stream_id=stream_id,
            event_type=StreamEventType.CLAIM_EXTRACTED,
            json_filters=filters,
            limit=limit,
        )
        return [e.payload for e in events if e.payload]

    def _handle_query_highlights(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        stream_id = params.get("stream_id")
        min_score = float(params.get("min_score", 0.0))
        limit = int(params.get("limit", 50))

        events = self.store.query_events(
            stream_id=stream_id,
            event_type=StreamEventType.HIGHLIGHT_MOMENT,
            limit=limit * 2,
        )
        filtered = []
        for e in events:
            if not e.payload:
                continue
            score = float(e.payload.get("score") or e.payload.get("highlight_score") or 0.0)
            if score >= min_score:
                filtered.append(e.payload)
                if len(filtered) >= limit:
                    break
        return filtered

    def _handle_query_chatters(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        stream_id = params.get("stream_id")
        limit = int(params.get("limit", 50))

        events = self.store.query_events(
            stream_id=stream_id,
            event_type=StreamEventType.SAFETY_ALERT,
            limit=limit,
        )
        return [e.payload for e in events if e.payload]
