"""Centralized Schema Registry and JSON-Schema Generation (Spec 16)."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
from pydantic import BaseModel
import yaml

from stream_fusion.models import schemas as model_schemas
from stream_fusion.monitoring.audit import StageTelemetry, RegressionAlert, AuditRunRecord
from stream_fusion.schema.envelope import StreamFusionEnvelope, EnvelopeTelemetry


class SchemaRegistry:
    """Registry maintaining all StreamFusion Pydantic contracts and JSON Schemas."""

    def __init__(self):
        self._registry: Dict[str, Type[BaseModel]] = {}
        self._register_default_models()

    def _register_default_models(self) -> None:
        """Discovers and registers all pipeline data models."""
        # Core & Multimodal Schemas
        for name in [
            "ChatMessage",
            "ChatEmote",
            "WordTiming",
            "AudioSegment",
            "BoundingBox",
            "VisualKeyframe",
            "FusionSlice",
            "StreamAnalysisResult",
            "ChatIntentDistribution",
            "MemeBurstEvent",
            "ChatterProfile",
            "BrandProfile",
            "SponsorSegment",
            "SponsorImpactReport",
            "ChunkManifest",
            "VoiceprintProfile",
            "SpeakerMatchResult",
            "CoStreamInteraction",
            "ChatterDetailedProfile",
            "ChatterSafetyVerdict",
            "BrigadeCluster",
            "StreamerClaim",
            "EntityStanceRecord",
            "SocialPostCard",
            "ScreenWebContext",
            "ScreenGameContext",
            "ReadAlongSegment",
            "SlangCandidate",
            "AdaptiveTermEntry",
            "WorkerTaskInput",
            "ShortCandidate",
            "EditorialCutPlan",
            "ContentAuditReport",
            "PlatformCopyBundle",
            "ViralityScoreCard",
            "ShortProductionPackage",
            "PublishResult",
            "LiveStreamConfig",
            "LiveTailHealthMetrics",
            "LiveStreamStatus",
            "LiveClientSubscription",
            "WebGroundingCitation",
            "GroundedClaimResult",
            "StanceShiftRecord",
            "EntityOpinionSynthesis",
        ]:
            if hasattr(model_schemas, name):
                cls = getattr(model_schemas, name)
                if isinstance(cls, type) and issubclass(cls, BaseModel):
                    self.register(name, cls)

        # Monitoring & Auditing Schemas
        self.register("StageTelemetry", StageTelemetry)
        self.register("RegressionAlert", RegressionAlert)
        self.register("AuditRunRecord", AuditRunRecord)

        # Message Envelope Schemas
        self.register("EnvelopeTelemetry", EnvelopeTelemetry)
        self.register("StreamFusionEnvelope", StreamFusionEnvelope)

    def register(self, name: str, model_cls: Type[BaseModel]) -> None:
        """Registers a Pydantic model under a unique schema identifier."""
        self._registry[name] = model_cls

    def get_model(self, name: str) -> Optional[Type[BaseModel]]:
        """Retrieves registered model class by name."""
        return self._registry.get(name)

    def list_schemas(self) -> List[Dict[str, str]]:
        """Returns metadata list of all registered schemas."""
        result = []
        for name, cls in sorted(self._registry.items()):
            doc = (cls.__doc__ or "").strip().split("\n")[0]
            result.append({
                "schema_name": name,
                "module": cls.__module__,
                "description": doc,
            })
        return result

    def export_json_schema(self, name: str) -> Dict[str, Any]:
        """Generates a Draft-07 / 2020-12 JSON Schema for the specified model."""
        cls = self.get_model(name)
        if not cls:
            raise KeyError(f"Schema '{name}' not found in registry.")
        return cls.model_json_schema()

    def export_all_schemas(
        self, output_dir: Path, fmt: str = "json"
    ) -> Dict[str, Path]:
        """Exports JSON-Schema definitions for all models into the target directory."""
        output_dir.mkdir(parents=True, exist_ok=True)
        exported: Dict[str, Path] = {}

        for name, cls in self._registry.items():
            schema_data = cls.model_json_schema()
            ext = "yaml" if fmt.lower() in ("yaml", "yml") else "json"
            out_file = output_dir / f"{name}.schema.{ext}"

            if ext == "yaml":
                with open(out_file, "w", encoding="utf-8") as f:
                    yaml.dump(schema_data, f, sort_keys=False)
            else:
                with open(out_file, "w", encoding="utf-8") as f:
                    json.dump(schema_data, f, indent=2)

            exported[name] = out_file

        return exported

    def validate_data(
        self, schema_name: str, data: Dict[str, Any]
    ) -> Tuple[bool, Optional[str], Optional[BaseModel]]:
        """Validates raw dict data against a registered schema model."""
        cls = self.get_model(schema_name)
        if not cls:
            return False, f"Unknown schema: {schema_name}", None

        try:
            instance = cls.model_validate(data)
            return True, None, instance
        except Exception as e:
            return False, str(e), None


# Default global instance
default_schema_registry = SchemaRegistry()
