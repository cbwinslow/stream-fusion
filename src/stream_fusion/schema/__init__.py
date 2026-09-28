"""StreamFusion Unified JSON Schema & Agent Communication Package (Spec 16)."""

from stream_fusion.schema.envelope import (
    StreamEventType,
    EnvelopeTelemetry,
    StreamFusionEnvelope,
)
from stream_fusion.schema.registry import (
    SchemaRegistry,
    default_schema_registry,
)
from stream_fusion.schema.adapters import (
    JsonlStreamAdapter,
    SqliteJsonStore,
    ParquetJsonBridge,
)
from stream_fusion.schema.agent_rpc import (
    AgentRpcDispatcher,
)

__all__ = [
    "StreamEventType",
    "EnvelopeTelemetry",
    "StreamFusionEnvelope",
    "SchemaRegistry",
    "default_schema_registry",
    "JsonlStreamAdapter",
    "SqliteJsonStore",
    "ParquetJsonBridge",
    "AgentRpcDispatcher",
]
