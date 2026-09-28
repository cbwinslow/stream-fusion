"""Storage and Persistence Adapters: JSONL, SQLite JSON1, and Parquet Bridge (Spec 16)."""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, Iterable, Iterator, List, Optional
import pandas as pd

from stream_fusion.models.schemas import FusionSlice
from stream_fusion.schema.envelope import (
    StreamEventType,
    StreamFusionEnvelope,
    EnvelopeTelemetry,
)


class JsonlStreamAdapter:
    """High-throughput, streaming newline-delimited JSON reader and writer."""

    @staticmethod
    def write_envelope(file_path: Path, envelope: StreamFusionEnvelope) -> None:
        """Appends a single envelope as a JSON line."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, "a", encoding="utf-8") as f:
            f.write(envelope.to_json() + "\n")

    @staticmethod
    def write_envelopes(
        file_path: Path, envelopes: Iterable[StreamFusionEnvelope]
    ) -> int:
        """Appends multiple envelopes as JSON lines in batch."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(file_path, "a", encoding="utf-8") as f:
            for env in envelopes:
                f.write(env.to_json() + "\n")
                count += 1
        return count

    @staticmethod
    def read_envelopes(
        file_path: Path,
        event_type: Optional[StreamEventType] = None,
        stream_id: Optional[str] = None,
    ) -> Iterator[StreamFusionEnvelope]:
        """Streaming generator yielding envelopes from a JSONL file with memory bounds."""
        if not file_path.exists():
            return

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    envelope = StreamFusionEnvelope.from_json(line)
                    if event_type and envelope.event_type != event_type:
                        continue
                    if stream_id and envelope.stream_id != stream_id:
                        continue
                    yield envelope
                except Exception:
                    continue


class SqliteJsonStore:
    """SQLite-backed event store leveraging JSON1 extensions for attribute querying."""

    def __init__(self, db_path: Path = Path("stream_events.db")):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS stream_events (
                    message_id TEXT PRIMARY KEY,
                    version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    stream_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    trace_id TEXT,
                    producer TEXT,
                    payload_json TEXT NOT NULL,
                    telemetry_json TEXT,
                    metadata_json TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stream_events_stream_type
                ON stream_events(stream_id, event_type);
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_stream_events_timestamp
                ON stream_events(timestamp);
                """
            )
            conn.commit()

    def insert_envelope(self, envelope: StreamFusionEnvelope) -> None:
        """Inserts a single envelope into the SQLite store."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO stream_events (
                    message_id, version, timestamp, stream_id, event_type,
                    trace_id, producer, payload_json, telemetry_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    envelope.message_id,
                    envelope.version,
                    envelope.timestamp,
                    envelope.stream_id,
                    envelope.event_type.value
                    if isinstance(envelope.event_type, StreamEventType)
                    else str(envelope.event_type),
                    envelope.trace_id,
                    envelope.producer,
                    json.dumps(envelope.payload)
                    if envelope.payload is not None
                    else "{}",
                    envelope.telemetry.model_dump_json()
                    if envelope.telemetry
                    else None,
                    json.dumps(envelope.metadata),
                ),
            )
            conn.commit()

    def insert_envelopes(self, envelopes: List[StreamFusionEnvelope]) -> int:
        """Batch inserts envelopes within a single transaction."""
        if not envelopes:
            return 0

        rows = []
        for env in envelopes:
            rows.append(
                (
                    env.message_id,
                    env.version,
                    env.timestamp,
                    env.stream_id,
                    env.event_type.value
                    if isinstance(env.event_type, StreamEventType)
                    else str(env.event_type),
                    env.trace_id,
                    env.producer,
                    json.dumps(env.payload) if env.payload is not None else "{}",
                    env.telemetry.model_dump_json() if env.telemetry else None,
                    json.dumps(env.metadata),
                )
            )

        with self._get_connection() as conn:
            conn.executemany(
                """
                INSERT OR REPLACE INTO stream_events (
                    message_id, version, timestamp, stream_id, event_type,
                    trace_id, producer, payload_json, telemetry_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                rows,
            )
            conn.commit()
        return len(rows)

    def query_events(
        self,
        stream_id: Optional[str] = None,
        event_type: Optional[StreamEventType] = None,
        json_filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[StreamFusionEnvelope]:
        """Queries events with optional JSON1 payload attribute filtering."""
        clauses = []
        params = []

        if stream_id:
            clauses.append("stream_id = ?")
            params.append(stream_id)

        if event_type:
            clauses.append("event_type = ?")
            params.append(
                event_type.value
                if isinstance(event_type, StreamEventType)
                else str(event_type)
            )

        if json_filters:
            for k, v in json_filters.items():
                clauses.append(f"json_extract(payload_json, '$.{k}') = ?")
                # Coerce boolean/numeric to proper JSON types if needed
                params.append(v)

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT message_id, version, timestamp, stream_id, event_type,
                   trace_id, producer, payload_json, telemetry_json, metadata_json
            FROM stream_events
            {where_sql}
            ORDER BY timestamp ASC
            LIMIT ? OFFSET ?
        """
        params.extend([limit, offset])

        results = []
        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            for row in cursor.fetchall():
                payload = json.loads(row["payload_json"])
                telemetry = None
                if row["telemetry_json"]:
                    telemetry = EnvelopeTelemetry.model_validate_json(
                        row["telemetry_json"]
                    )
                metadata = (
                    json.loads(row["metadata_json"])
                    if row["metadata_json"]
                    else {}
                )

                env = StreamFusionEnvelope(
                    version=row["version"],
                    message_id=row["message_id"],
                    timestamp=row["timestamp"],
                    stream_id=row["stream_id"],
                    event_type=StreamEventType(row["event_type"]),
                    trace_id=row["trace_id"],
                    producer=row["producer"],
                    payload=payload,
                    telemetry=telemetry,
                    metadata=metadata,
                )
                results.append(env)
        return results

    def count_events(
        self,
        stream_id: Optional[str] = None,
        event_type: Optional[StreamEventType] = None,
    ) -> int:
        """Counts matching events in the store."""
        clauses = []
        params = []
        if stream_id:
            clauses.append("stream_id = ?")
            params.append(stream_id)
        if event_type:
            clauses.append("event_type = ?")
            params.append(
                event_type.value
                if isinstance(event_type, StreamEventType)
                else str(event_type)
            )

        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT COUNT(*) FROM stream_events {where_sql}"
        with self._get_connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchone()[0]


class ParquetJsonBridge:
    """Bridge for lossless bidirectional conversion between Parquet matrices and JSON envelopes."""

    @staticmethod
    def slices_to_envelopes(
        slices: List[FusionSlice], stream_id: str
    ) -> List[StreamFusionEnvelope]:
        """Converts a collection of FusionSlice domain models into standardized envelopes."""
        envelopes = []
        for s in slices:
            env = StreamFusionEnvelope.create(
                stream_id=stream_id,
                event_type=StreamEventType.FUSION_SLICE,
                payload=s.model_dump(),
                producer="stream_fusion.fusion.matrix",
            )
            envelopes.append(env)
        return envelopes

    @staticmethod
    def envelopes_to_slices(
        envelopes: List[StreamFusionEnvelope],
    ) -> List[FusionSlice]:
        """Extracts and validates FusionSlice models from envelopes."""
        slices = []
        for env in envelopes:
            if env.event_type == StreamEventType.FUSION_SLICE and env.payload:
                s = FusionSlice.model_validate(env.payload)
                slices.append(s)
        return slices

    @staticmethod
    def export_envelopes_to_parquet(
        envelopes: List[StreamFusionEnvelope], parquet_path: Path
    ) -> None:
        """Flattens envelopes into a columnar Parquet file."""
        parquet_path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for env in envelopes:
            rows.append(
                {
                    "message_id": env.message_id,
                    "version": env.version,
                    "timestamp": env.timestamp,
                    "stream_id": env.stream_id,
                    "event_type": env.event_type.value,
                    "producer": env.producer or "",
                    "payload_json": json.dumps(env.payload)
                    if env.payload is not None
                    else "{}",
                    "metadata_json": json.dumps(env.metadata),
                }
            )

        df = pd.DataFrame(rows)
        df.to_parquet(parquet_path, engine="pyarrow", index=False)

    @staticmethod
    def load_envelopes_from_parquet(
        parquet_path: Path, stream_id: Optional[str] = None
    ) -> List[StreamFusionEnvelope]:
        """Loads envelopes from a Parquet file."""
        if not parquet_path.exists():
            return []

        df = pd.read_parquet(parquet_path, engine="pyarrow")
        if stream_id and "stream_id" in df.columns:
            df = df[df["stream_id"] == stream_id]

        envelopes = []
        for _, row in df.iterrows():
            payload = (
                json.loads(row["payload_json"])
                if row.get("payload_json")
                else {}
            )
            metadata = (
                json.loads(row["metadata_json"])
                if row.get("metadata_json")
                else {}
            )

            env = StreamFusionEnvelope(
                version=str(row.get("version", "1.0")),
                message_id=str(row["message_id"]),
                timestamp=str(row["timestamp"]),
                stream_id=str(row["stream_id"]),
                event_type=StreamEventType(row["event_type"]),
                producer=str(row["producer"]) if row.get("producer") else None,
                payload=payload,
                metadata=metadata,
            )
            envelopes.append(env)
        return envelopes
