"""Target Streamer Roster Ingestion and Serialization (Spec 25).

Supports loading and saving streamer targets in YAML, JSON, and CSV formats.
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Union
import yaml

from stream_fusion.models.schemas import StreamerTargetRecord


class RosterLoader:
    """Parses and serializes StreamerTargetRecord collections from/to files."""

    @classmethod
    def load(cls, path: Union[str, Path]) -> List[StreamerTargetRecord]:
        """Loads records from a file, automatically detecting format by suffix."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Roster file not found: {p}")

        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            return cls.load_from_yaml(p)
        elif suffix == ".json":
            return cls.load_from_json(p)
        elif suffix == ".csv":
            return cls.load_from_csv(p)
        else:
            raise ValueError(f"Unsupported roster file format: '{suffix}'. Expected .yaml, .json, or .csv")

    @classmethod
    def save(cls, path: Union[str, Path], records: List[StreamerTargetRecord]) -> None:
        """Saves records to a file, automatically detecting format by suffix."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        suffix = p.suffix.lower()
        if suffix in (".yaml", ".yml"):
            cls.save_to_yaml(p, records)
        elif suffix == ".json":
            cls.save_to_json(p, records)
        elif suffix == ".csv":
            cls.save_to_csv(p, records)
        else:
            raise ValueError(f"Unsupported roster file format: '{suffix}'. Expected .yaml, .json, or .csv")

    @classmethod
    def load_from_yaml(cls, path: Union[str, Path]) -> List[StreamerTargetRecord]:
        """Loads records from a YAML file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not data:
            return []

        # Support root list or dict with 'streamers' / 'targets'
        raw_items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = data.get("streamers") or data.get("targets") or data.get("roster") or [data]
        else:
            raise ValueError(f"Invalid YAML structure in {p}: expected list or mapping.")

        return [cls._dict_to_record(item) for item in raw_items]

    @classmethod
    def save_to_yaml(cls, path: Union[str, Path], records: List[StreamerTargetRecord]) -> None:
        """Saves records to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        dump_data = [rec.model_dump(mode="json", exclude_none=True) for rec in records]
        with open(p, "w", encoding="utf-8") as f:
            yaml.safe_dump(dump_data, f, sort_keys=False, indent=2)

    @classmethod
    def load_from_json(cls, path: Union[str, Path]) -> List[StreamerTargetRecord]:
        """Loads records from a JSON file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            return []

        raw_items: List[Dict[str, Any]] = []
        if isinstance(data, list):
            raw_items = data
        elif isinstance(data, dict):
            raw_items = data.get("streamers") or data.get("targets") or data.get("roster") or [data]
        else:
            raise ValueError(f"Invalid JSON structure in {p}: expected list or dict.")

        return [cls._dict_to_record(item) for item in raw_items]

    @classmethod
    def save_to_json(cls, path: Union[str, Path], records: List[StreamerTargetRecord]) -> None:
        """Saves records to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        dump_data = [rec.model_dump(mode="json", exclude_none=True) for rec in records]
        with open(p, "w", encoding="utf-8") as f:
            json.dump(dump_data, f, indent=2)

    @classmethod
    def load_from_csv(cls, path: Union[str, Path]) -> List[StreamerTargetRecord]:
        """Loads records from a CSV file."""
        p = Path(path)
        records: List[StreamerTargetRecord] = []
        with open(p, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                parsed = cls._parse_csv_row(row)
                records.append(cls._dict_to_record(parsed))
        return records

    @classmethod
    def save_to_csv(cls, path: Union[str, Path], records: List[StreamerTargetRecord]) -> None:
        """Saves records to a CSV file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "streamer_id",
            "display_name",
            "channel_urls",
            "primary_platform",
            "quality_preset",
            "include_chat",
            "max_recent_vods",
            "lookback_days",
            "download_priority",
            "destination_override",
            "tags",
            "enabled",
        ]
        with open(p, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in records:
                row = {
                    "streamer_id": r.streamer_id,
                    "display_name": r.display_name,
                    "channel_urls": ";".join(r.channel_urls),
                    "primary_platform": r.primary_platform,
                    "quality_preset": r.quality_preset,
                    "include_chat": "true" if r.include_chat else "false",
                    "max_recent_vods": r.max_recent_vods,
                    "lookback_days": r.lookback_days,
                    "download_priority": r.download_priority,
                    "destination_override": r.destination_override or "",
                    "tags": ";".join(r.tags),
                    "enabled": "true" if r.enabled else "false",
                }
                writer.writerow(row)

    @staticmethod
    def _parse_csv_row(row: Dict[str, str]) -> Dict[str, Any]:
        """Converts raw string fields from CSV to typed dictionary."""
        out: Dict[str, Any] = {}
        for k, v in row.items():
            if v is None:
                continue
            v_str = str(v).strip()
            if not v_str:
                continue

            if k == "channel_urls":
                if v_str.startswith("["):
                    out[k] = json.loads(v_str)
                elif ";" in v_str:
                    out[k] = [u.strip() for u in v_str.split(";") if u.strip()]
                elif "," in v_str:
                    out[k] = [u.strip() for u in v_str.split(",") if u.strip()]
                else:
                    out[k] = [v_str]
            elif k == "tags":
                if v_str.startswith("["):
                    out[k] = json.loads(v_str)
                elif ";" in v_str:
                    out[k] = [t.strip() for t in v_str.split(";") if t.strip()]
                elif "," in v_str:
                    out[k] = [t.strip() for t in v_str.split(",") if t.strip()]
                else:
                    out[k] = [v_str]
            elif k in ("include_chat", "enabled"):
                out[k] = v_str.lower() in ("true", "1", "yes", "t")
            elif k in ("max_recent_vods", "lookback_days", "download_priority"):
                out[k] = int(v_str)
            else:
                out[k] = v_str
        return out

    @staticmethod
    def _dict_to_record(data: Dict[str, Any]) -> StreamerTargetRecord:
        """Parses and validates a dictionary into StreamerTargetRecord."""
        # Normalize channel_urls if given as single string
        if "channel_urls" in data and isinstance(data["channel_urls"], str):
            data["channel_urls"] = [data["channel_urls"]]
        return StreamerTargetRecord(**data)
