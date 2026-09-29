"""Dual-Backend Catalog Database (Spec 25).

Supports high-throughput PostgreSQL for production homelab environments and
zero-configuration SQLite for local development and self-contained testing.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import threading
from typing import Any, Dict, Iterator, List, Optional, Union

from stream_fusion.models.schemas import (
    HarvestedVodRecord,
    HarvestStatus,
    StreamerTargetRecord,
)


class HarvestCatalog:
    """Central VOD and Streamer Target Catalog supporting SQLite and PostgreSQL."""

    def __init__(self, database_url: Union[str, Path] = "sqlite:///catalog.db"):
        self.raw_url = str(database_url)
        self._lock = threading.RLock()
        self._is_postgres = (
            self.raw_url.startswith("postgresql://")
            or self.raw_url.startswith("postgres://")
        )

        if self._is_postgres:
            self._init_postgres()
        else:
            self._init_sqlite()

    def _init_sqlite(self) -> None:
        """Initializes SQLite backend."""
        sqlite_path = self.raw_url
        if sqlite_path.startswith("sqlite:///"):
            sqlite_path = sqlite_path[len("sqlite:///") :]
        elif sqlite_path.startswith("sqlite://"):
            sqlite_path = sqlite_path[len("sqlite://") :]

        self.db_path = sqlite_path
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        # For :memory:, keep a persistent connection so tables persist across operations
        self._mem_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row

        self._create_sqlite_tables()

    def _init_postgres(self) -> None:
        """Initializes PostgreSQL connection pool / connection."""
        try:
            import psycopg
            self._pg_module = "psycopg"
        except ImportError:
            try:
                import psycopg2
                self._pg_module = "psycopg2"
            except ImportError:
                raise ImportError(
                    "PostgreSQL URL specified, but neither 'psycopg' nor 'psycopg2' is installed. "
                    "Install with: pip install psycopg[binary] or use sqlite:///catalog.db"
                )

        self._create_postgres_tables()

    @contextmanager
    def _get_connection(self) -> Iterator[Any]:
        """Provides a thread-safe database connection."""
        with self._lock:
            if self._is_postgres:
                if self._pg_module == "psycopg":
                    import psycopg
                    conn = psycopg.connect(self.raw_url)
                else:
                    import psycopg2
                    conn = psycopg2.connect(self.raw_url)
                try:
                    yield conn
                    conn.commit()
                except Exception:
                    conn.rollback()
                    raise
                finally:
                    conn.close()
            else:
                if self._mem_conn is not None:
                    try:
                        yield self._mem_conn
                        self._mem_conn.commit()
                    except Exception:
                        self._mem_conn.rollback()
                        raise
                else:
                    conn = sqlite3.connect(self.db_path, check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    conn.execute("PRAGMA journal_mode=WAL;")
                    conn.execute("PRAGMA foreign_keys=ON;")
                    try:
                        yield conn
                        conn.commit()
                    except Exception:
                        conn.rollback()
                        raise
                    finally:
                        conn.close()

    def _create_sqlite_tables(self) -> None:
        """Creates SQLite tables and indices."""
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS streamer_targets (
                    streamer_id TEXT PRIMARY KEY,
                    display_name TEXT NOT NULL,
                    channel_urls TEXT NOT NULL,
                    primary_platform TEXT NOT NULL,
                    quality_preset TEXT DEFAULT 'best',
                    include_chat INTEGER DEFAULT 1,
                    max_recent_vods INTEGER DEFAULT 5,
                    lookback_days INTEGER DEFAULT 14,
                    download_priority INTEGER DEFAULT 5,
                    destination_override TEXT,
                    tags TEXT,
                    enabled INTEGER DEFAULT 1,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    last_synced_at TEXT,
                    voiceprint_embedding TEXT
                );
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS harvested_vods (
                    vod_id TEXT PRIMARY KEY,
                    streamer_id TEXT NOT NULL REFERENCES streamer_targets(streamer_id),
                    platform TEXT NOT NULL,
                    title TEXT NOT NULL,
                    published_at TEXT,
                    duration_sec REAL DEFAULT 0.0,
                    status TEXT NOT NULL DEFAULT 'DISCOVERED',
                    video_path TEXT,
                    chat_path TEXT,
                    metadata_path TEXT,
                    thumbnail_path TEXT,
                    file_size_bytes INTEGER DEFAULT 0,
                    download_speed_mbps REAL DEFAULT 0.0,
                    retry_count INTEGER DEFAULT 0,
                    error_message TEXT,
                    harvested_at TEXT,
                    analyzed_at TEXT,
                    raw_metadata TEXT
                );
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_harvested_vods_status ON harvested_vods(status);"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_harvested_vods_streamer ON harvested_vods(streamer_id, published_at);"
            )

    def _create_postgres_tables(self) -> None:
        """Creates PostgreSQL tables and indices."""
        with self._get_connection() as conn:
            with conn.cursor() as cur:
                # Optionally enable pg_trgm for fuzzy search if available
                try:
                    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")
                except Exception:
                    pass

                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS streamer_targets (
                        streamer_id TEXT PRIMARY KEY,
                        display_name TEXT NOT NULL,
                        channel_urls TEXT NOT NULL,
                        primary_platform TEXT NOT NULL,
                        quality_preset TEXT DEFAULT 'best',
                        include_chat BOOLEAN DEFAULT TRUE,
                        max_recent_vods INTEGER DEFAULT 5,
                        lookback_days INTEGER DEFAULT 14,
                        download_priority INTEGER DEFAULT 5,
                        destination_override TEXT,
                        tags TEXT,
                        enabled BOOLEAN DEFAULT TRUE,
                        created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                        last_synced_at TIMESTAMP WITH TIME ZONE,
                        voiceprint_embedding TEXT
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS harvested_vods (
                        vod_id TEXT PRIMARY KEY,
                        streamer_id TEXT NOT NULL REFERENCES streamer_targets(streamer_id),
                        platform TEXT NOT NULL,
                        title TEXT NOT NULL,
                        published_at TIMESTAMP WITH TIME ZONE,
                        duration_sec REAL DEFAULT 0.0,
                        status TEXT NOT NULL DEFAULT 'DISCOVERED',
                        video_path TEXT,
                        chat_path TEXT,
                        metadata_path TEXT,
                        thumbnail_path TEXT,
                        file_size_bytes BIGINT DEFAULT 0,
                        download_speed_mbps REAL DEFAULT 0.0,
                        retry_count INTEGER DEFAULT 0,
                        error_message TEXT,
                        harvested_at TIMESTAMP WITH TIME ZONE,
                        analyzed_at TIMESTAMP WITH TIME ZONE,
                        raw_metadata TEXT
                    );
                    """
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_harvested_vods_status ON harvested_vods(status);"
                )
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_harvested_vods_streamer ON harvested_vods(streamer_id, published_at DESC);"
                )

    # --- Streamer Target Operations ---

    def add_target(self, target: StreamerTargetRecord) -> None:
        """Upserts a streamer target into the catalog."""
        urls_json = json.dumps(target.channel_urls)
        tags_json = json.dumps(target.tags)
        vp_json = json.dumps(target.voiceprint_embedding) if target.voiceprint_embedding else None
        created_at_str = target.created_at.isoformat() if target.created_at else datetime.now(timezone.utc).isoformat()
        last_synced_str = target.last_synced_at.isoformat() if target.last_synced_at else None

        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO streamer_targets (
                            streamer_id, display_name, channel_urls, primary_platform,
                            quality_preset, include_chat, max_recent_vods, lookback_days,
                            download_priority, destination_override, tags, enabled,
                            created_at, last_synced_at, voiceprint_embedding
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (streamer_id) DO UPDATE SET
                            display_name = EXCLUDED.display_name,
                            channel_urls = EXCLUDED.channel_urls,
                            primary_platform = EXCLUDED.primary_platform,
                            quality_preset = EXCLUDED.quality_preset,
                            include_chat = EXCLUDED.include_chat,
                            max_recent_vods = EXCLUDED.max_recent_vods,
                            lookback_days = EXCLUDED.lookback_days,
                            download_priority = EXCLUDED.download_priority,
                            destination_override = EXCLUDED.destination_override,
                            tags = EXCLUDED.tags,
                            enabled = EXCLUDED.enabled,
                            voiceprint_embedding = EXCLUDED.voiceprint_embedding;
                        """,
                        (
                            target.streamer_id,
                            target.display_name,
                            urls_json,
                            target.primary_platform,
                            target.quality_preset,
                            target.include_chat,
                            target.max_recent_vods,
                            target.lookback_days,
                            target.download_priority,
                            target.destination_override,
                            tags_json,
                            target.enabled,
                            created_at_str,
                            last_synced_str,
                            vp_json,
                        ),
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO streamer_targets (
                        streamer_id, display_name, channel_urls, primary_platform,
                        quality_preset, include_chat, max_recent_vods, lookback_days,
                        download_priority, destination_override, tags, enabled,
                        created_at, last_synced_at, voiceprint_embedding
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(streamer_id) DO UPDATE SET
                        display_name = excluded.display_name,
                        channel_urls = excluded.channel_urls,
                        primary_platform = excluded.primary_platform,
                        quality_preset = excluded.quality_preset,
                        include_chat = excluded.include_chat,
                        max_recent_vods = excluded.max_recent_vods,
                        lookback_days = excluded.lookback_days,
                        download_priority = excluded.download_priority,
                        destination_override = excluded.destination_override,
                        tags = excluded.tags,
                        enabled = excluded.enabled,
                        voiceprint_embedding = excluded.voiceprint_embedding;
                    """,
                    (
                        target.streamer_id,
                        target.display_name,
                        urls_json,
                        target.primary_platform,
                        target.quality_preset,
                        1 if target.include_chat else 0,
                        target.max_recent_vods,
                        target.lookback_days,
                        target.download_priority,
                        target.destination_override,
                        tags_json,
                        1 if target.enabled else 0,
                        created_at_str,
                        last_synced_str,
                        vp_json,
                    ),
                )

    def get_target(self, streamer_id: str) -> Optional[StreamerTargetRecord]:
        """Retrieves a streamer target by ID."""
        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM streamer_targets WHERE streamer_id = %s;",
                        (streamer_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    cols = [desc[0] for desc in cur.description]
                    data = dict(zip(cols, row))
            else:
                cursor = conn.execute(
                    "SELECT * FROM streamer_targets WHERE streamer_id = ?;",
                    (streamer_id,),
                )
                row = cursor.fetchone()
                if not row:
                    return None
                data = dict(row)

        return self._row_to_target(data)

    def list_targets(self, enabled_only: bool = False) -> List[StreamerTargetRecord]:
        """Lists all registered streamer targets, optionally filtering by enabled status."""
        query = "SELECT * FROM streamer_targets"
        if enabled_only:
            query += " WHERE enabled = 1" if not self._is_postgres else " WHERE enabled = TRUE"
        query += " ORDER BY download_priority DESC, streamer_id ASC;"

        records: List[StreamerTargetRecord] = []
        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
                    cols = [desc[0] for desc in cur.description]
                    for r in rows:
                        records.append(self._row_to_target(dict(zip(cols, r))))
            else:
                cursor = conn.execute(query)
                for r in cursor.fetchall():
                    records.append(self._row_to_target(dict(r)))

        return records

    def update_target_sync_time(self, streamer_id: str, synced_at: Optional[datetime] = None) -> None:
        """Updates last_synced_at timestamp for a target."""
        ts_str = (synced_at or datetime.now(timezone.utc)).isoformat()
        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE streamer_targets SET last_synced_at = %s WHERE streamer_id = %s;",
                        (ts_str, streamer_id),
                    )
            else:
                conn.execute(
                    "UPDATE streamer_targets SET last_synced_at = ? WHERE streamer_id = ?;",
                    (ts_str, streamer_id),
                )

    def delete_target(self, streamer_id: str) -> bool:
        """Deletes a streamer target by ID."""
        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute("DELETE FROM streamer_targets WHERE streamer_id = %s;", (streamer_id,))
                    return bool(cur.rowcount > 0)
            else:
                cur = conn.execute("DELETE FROM streamer_targets WHERE streamer_id = ?;", (streamer_id,))
                return bool(cur.rowcount > 0)

    # --- Harvested VOD Operations ---

    def add_vod(self, vod: HarvestedVodRecord) -> None:
        """Upserts a harvested VOD entry."""
        pub_str = vod.published_at.isoformat() if vod.published_at else None
        harv_str = vod.harvested_at.isoformat() if vod.harvested_at else None
        analy_str = vod.analyzed_at.isoformat() if vod.analyzed_at else None
        raw_meta = json.dumps(vod.raw_metadata) if vod.raw_metadata else None
        status_val = vod.status.value if isinstance(vod.status, HarvestStatus) else str(vod.status)

        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        INSERT INTO harvested_vods (
                            vod_id, streamer_id, platform, title, published_at,
                            duration_sec, status, video_path, chat_path, metadata_path,
                            thumbnail_path, file_size_bytes, download_speed_mbps,
                            retry_count, error_message, harvested_at, analyzed_at, raw_metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (vod_id) DO UPDATE SET
                            title = EXCLUDED.title,
                            duration_sec = EXCLUDED.duration_sec,
                            status = EXCLUDED.status,
                            video_path = EXCLUDED.video_path,
                            chat_path = EXCLUDED.chat_path,
                            metadata_path = EXCLUDED.metadata_path,
                            thumbnail_path = EXCLUDED.thumbnail_path,
                            file_size_bytes = EXCLUDED.file_size_bytes,
                            download_speed_mbps = EXCLUDED.download_speed_mbps,
                            retry_count = EXCLUDED.retry_count,
                            error_message = EXCLUDED.error_message,
                            harvested_at = EXCLUDED.harvested_at,
                            analyzed_at = EXCLUDED.analyzed_at,
                            raw_metadata = EXCLUDED.raw_metadata;
                        """,
                        (
                            vod.vod_id,
                            vod.streamer_id,
                            vod.platform,
                            vod.title,
                            pub_str,
                            vod.duration_sec,
                            status_val,
                            vod.video_path,
                            vod.chat_path,
                            vod.metadata_path,
                            vod.thumbnail_path,
                            vod.file_size_bytes,
                            vod.download_speed_mbps,
                            vod.retry_count,
                            vod.error_message,
                            harv_str,
                            analy_str,
                            raw_meta,
                        ),
                    )
            else:
                conn.execute(
                    """
                    INSERT INTO harvested_vods (
                        vod_id, streamer_id, platform, title, published_at,
                        duration_sec, status, video_path, chat_path, metadata_path,
                        thumbnail_path, file_size_bytes, download_speed_mbps,
                        retry_count, error_message, harvested_at, analyzed_at, raw_metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(vod_id) DO UPDATE SET
                        title = excluded.title,
                        duration_sec = excluded.duration_sec,
                        status = excluded.status,
                        video_path = excluded.video_path,
                        chat_path = excluded.chat_path,
                        metadata_path = excluded.metadata_path,
                        thumbnail_path = excluded.thumbnail_path,
                        file_size_bytes = excluded.file_size_bytes,
                        download_speed_mbps = excluded.download_speed_mbps,
                        retry_count = excluded.retry_count,
                        error_message = excluded.error_message,
                        harvested_at = excluded.harvested_at,
                        analyzed_at = excluded.analyzed_at,
                        raw_metadata = excluded.raw_metadata;
                    """,
                    (
                        vod.vod_id,
                        vod.streamer_id,
                        vod.platform,
                        vod.title,
                        pub_str,
                        vod.duration_sec,
                        status_val,
                        vod.video_path,
                        vod.chat_path,
                        vod.metadata_path,
                        vod.thumbnail_path,
                        vod.file_size_bytes,
                        vod.download_speed_mbps,
                        vod.retry_count,
                        vod.error_message,
                        harv_str,
                        analy_str,
                        raw_meta,
                    ),
                )

    def get_vod(self, vod_id: str) -> Optional[HarvestedVodRecord]:
        """Retrieves a harvested VOD record by ID."""
        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT * FROM harvested_vods WHERE vod_id = %s;",
                        (vod_id,),
                    )
                    row = cur.fetchone()
                    if not row:
                        return None
                    cols = [desc[0] for desc in cur.description]
                    data = dict(zip(cols, row))
            else:
                cur = conn.execute(
                    "SELECT * FROM harvested_vods WHERE vod_id = ?;",
                    (vod_id,),
                )
                row = cur.fetchone()
                if not row:
                    return None
                data = dict(row)

        return self._row_to_vod(data)

    def list_vods(
        self,
        streamer_id: Optional[str] = None,
        status: Optional[Union[HarvestStatus, str]] = None,
        limit: Optional[int] = None,
    ) -> List[HarvestedVodRecord]:
        """Lists harvested VODs with optional filtering by streamer and status."""
        conditions: List[str] = []
        params: List[Any] = []

        if streamer_id:
            conditions.append("streamer_id = %s" if self._is_postgres else "streamer_id = ?")
            params.append(streamer_id)
        if status:
            status_val = status.value if isinstance(status, HarvestStatus) else str(status)
            conditions.append("status = %s" if self._is_postgres else "status = ?")
            params.append(status_val)

        where_clause = ("WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"SELECT * FROM harvested_vods {where_clause} ORDER BY published_at DESC, vod_id DESC"
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"

        records: List[HarvestedVodRecord] = []
        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(query, tuple(params))
                    rows = cur.fetchall()
                    cols = [desc[0] for desc in cur.description]
                    for r in rows:
                        records.append(self._row_to_vod(dict(zip(cols, r))))
            else:
                cur = conn.execute(query, tuple(params))
                for r in cur.fetchall():
                    records.append(self._row_to_vod(dict(r)))

        return records

    def get_queued_vods(self, limit: Optional[int] = None) -> List[HarvestedVodRecord]:
        """Returns all VODs in QUEUED status ordered by streamer priority and publish date."""
        query = """
            SELECT hv.* FROM harvested_vods hv
            JOIN streamer_targets st ON hv.streamer_id = st.streamer_id
            WHERE hv.status = 'QUEUED'
            ORDER BY st.download_priority DESC, hv.published_at ASC
        """
        if limit and limit > 0:
            query += f" LIMIT {int(limit)}"

        records: List[HarvestedVodRecord] = []
        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(query)
                    rows = cur.fetchall()
                    cols = [desc[0] for desc in cur.description]
                    for r in rows:
                        records.append(self._row_to_vod(dict(zip(cols, r))))
            else:
                cur = conn.execute(query)
                for r in cur.fetchall():
                    records.append(self._row_to_vod(dict(r)))

        return records

    def update_vod_status(
        self,
        vod_id: str,
        status: Union[HarvestStatus, str],
        **kwargs: Any,
    ) -> Optional[HarvestedVodRecord]:
        """Updates status and additional fields of a VOD record."""
        status_val = status.value if isinstance(status, HarvestStatus) else str(status)
        updates = ["status = %s" if self._is_postgres else "status = ?"]
        params: List[Any] = [status_val]

        for k, v in kwargs.items():
            if v is not None:
                if isinstance(v, datetime):
                    v = v.isoformat()
                elif isinstance(v, dict):
                    v = json.dumps(v)
            updates.append(f"{k} = %s" if self._is_postgres else f"{k} = ?")
            params.append(v)

        params.append(vod_id)
        update_sql = f"UPDATE harvested_vods SET {', '.join(updates)} WHERE vod_id = {'%s' if self._is_postgres else '?'};"

        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(update_sql, tuple(params))
            else:
                conn.execute(update_sql, tuple(params))

        return self.get_vod(vod_id)

    def count_vods_by_status(self) -> Dict[str, int]:
        """Returns counts of VODs partitioned by status."""
        query = "SELECT status, COUNT(*) as cnt FROM harvested_vods GROUP BY status;"
        counts: Dict[str, int] = {}
        with self._get_connection() as conn:
            if self._is_postgres:
                with conn.cursor() as cur:
                    cur.execute(query)
                    for row in cur.fetchall():
                        counts[str(row[0])] = int(row[1])
            else:
                cur = conn.execute(query)
                for row in cur.fetchall():
                    counts[str(row[0])] = int(row[1])
        return counts

    def close(self) -> None:
        """Closes any persistent connections."""
        with self._lock:
            if self._mem_conn is not None:
                try:
                    self._mem_conn.close()
                except Exception:
                    pass
                self._mem_conn = None

    # --- Internal Row Helpers ---

    @staticmethod
    def _row_to_target(data: Dict[str, Any]) -> StreamerTargetRecord:
        """Deserializes a database row dictionary to StreamerTargetRecord."""
        urls = data.get("channel_urls")
        if isinstance(urls, str):
            try:
                data["channel_urls"] = json.loads(urls)
            except Exception:
                data["channel_urls"] = [urls]

        tags = data.get("tags")
        if isinstance(tags, str):
            try:
                data["tags"] = json.loads(tags)
            except Exception:
                data["tags"] = [tags] if tags else []

        vp = data.get("voiceprint_embedding")
        if isinstance(vp, str) and vp:
            try:
                data["voiceprint_embedding"] = json.loads(vp)
            except Exception:
                data["voiceprint_embedding"] = None

        if "include_chat" in data:
            data["include_chat"] = bool(data["include_chat"])
        if "enabled" in data:
            data["enabled"] = bool(data["enabled"])

        # Parse datetime strings
        for dt_field in ("created_at", "last_synced_at"):
            val = data.get(dt_field)
            if isinstance(val, str) and val:
                try:
                    data[dt_field] = datetime.fromisoformat(val.replace("Z", "+00:00"))
                except Exception:
                    data[dt_field] = None

        return StreamerTargetRecord(**data)

    @staticmethod
    def _row_to_vod(data: Dict[str, Any]) -> HarvestedVodRecord:
        """Deserializes a database row dictionary to HarvestedVodRecord."""
        raw_meta = data.get("raw_metadata")
        if isinstance(raw_meta, str) and raw_meta:
            try:
                data["raw_metadata"] = json.loads(raw_meta)
            except Exception:
                data["raw_metadata"] = {}
        elif not isinstance(raw_meta, dict):
            data["raw_metadata"] = {}

        # Parse status enum
        if "status" in data and isinstance(data["status"], str):
            try:
                data["status"] = HarvestStatus(data["status"])
            except Exception:
                data["status"] = HarvestStatus.DISCOVERED

        # Parse datetime fields
        for dt_field in ("published_at", "harvested_at", "analyzed_at"):
            val = data.get(dt_field)
            if isinstance(val, str) and val:
                try:
                    data[dt_field] = datetime.fromisoformat(val.replace("Z", "+00:00"))
                except Exception:
                    data[dt_field] = None

        return HarvestedVodRecord(**data)
