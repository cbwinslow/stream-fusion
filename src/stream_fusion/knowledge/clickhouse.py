"""ClickHouse Columnar Storage for Raw Chat Events and ASOF Cross-Section Alignment (Spec 29).

Provides:
- High-throughput ingestion of raw chat streams with LowCardinality dictionaries.
- Storage of speech_segments for temporal transcript correlation.
- Native ASOF JOIN query execution across speech timelines and chatter reactions.
- Transparent zero-dependency mock fallback for local environments without ClickHouse daemons.
"""

import json
import logging
import threading
from typing import Any, Dict, List, Optional, Union
import urllib.error
import urllib.parse
import urllib.request

logger = logging.getLogger(__name__)


CHAT_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS chat_events (
    streamer_id LowCardinality(String),
    vod_id String,
    timestamp_ms UInt64,
    timestamp_sec Float32,
    chatter_username LowCardinality(String),
    message_text String,
    emotes Array(LowCardinality(String)),
    sentiment_score Float32,
    is_subscriber UInt8,
    burst_flag UInt8
) ENGINE = MergeTree()
ORDER BY (streamer_id, vod_id, timestamp_ms)
SETTINGS index_granularity = 8192;
"""

SPEECH_SEGMENTS_DDL = """
CREATE TABLE IF NOT EXISTS speech_segments (
    streamer_id LowCardinality(String),
    vod_id String,
    start_ms UInt64,
    end_ms UInt64,
    speaker_id LowCardinality(String),
    transcript String
) ENGINE = MergeTree()
ORDER BY (streamer_id, vod_id, start_ms);
"""


class ClickHouseChatStorage:
    """High-performance ClickHouse storage backend for chat logs and ASOF cross-sections."""

    def __init__(
        self,
        url: str = "http://localhost:8123",
        database: str = "default",
        user: str = "default",
        password: str = "",
        timeout: float = 2.0,
        force_mock: bool = False,
    ):
        self.url = str(url).strip()
        self.database = database
        self.user = user
        self.password = password
        self.timeout = timeout
        self.force_mock = force_mock
        self._lock = threading.RLock()

        # Parse base URL
        clean_url = self.url
        if clean_url.startswith("clickhouse://"):
            host_part = clean_url[len("clickhouse://"):]
            self.base_url = f"http://{host_part}"
        elif clean_url.startswith("http://") or clean_url.startswith("https://"):
            self.base_url = clean_url.rstrip("/")
        else:
            self.base_url = f"http://{clean_url}"

        # In-memory mock storage structures
        self._mock_chat_events: List[Dict[str, Any]] = []
        self._mock_speech_segments: List[Dict[str, Any]] = []
        self.is_mock = force_mock

        if not self.force_mock:
            self._try_init_remote()
        else:
            logger.debug("ClickHouseChatStorage initialized in forced mock mode.")

    def _execute_http(
        self,
        query: str,
        post_data: Optional[bytes] = None,
    ) -> Optional[str]:
        if self.is_mock:
            return None
        params = {
            "database": self.database,
            "user": self.user,
        }
        if self.password:
            params["password"] = self.password
        if query:
            params["query"] = query

        full_url = f"{self.base_url}/?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            full_url,
            data=post_data,
            headers={"Content-Type": "application/octet-stream" if post_data else "text/plain"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                return raw
        except Exception as e:
            logger.debug(f"ClickHouse HTTP request failed: {e}")
            raise

    def _try_init_remote(self) -> None:
        try:
            # Ping ClickHouse server
            with urllib.request.urlopen(f"{self.base_url}/ping", timeout=self.timeout) as resp:
                if resp.read().decode("utf-8").strip() != "Ok.":
                    raise ConnectionError("ClickHouse ping failed")
            self.init_schema()
        except Exception as e:
            logger.info(f"ClickHouse server unreachable ({e}); operating in zero-dependency in-memory mode.")
            self.is_mock = True

    def init_schema(self) -> None:
        """Initializes ClickHouse tables if not present."""
        with self._lock:
            if not self.is_mock:
                try:
                    self._execute_http(CHAT_EVENTS_DDL)
                    self._execute_http(SPEECH_SEGMENTS_DDL)
                    return
                except Exception as e:
                    logger.warning(f"Failed to initialize ClickHouse tables remotely ({e}); using mock storage.")
                    self.is_mock = True

    @staticmethod
    def _normalize_chat_event(event: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Normalizes a chat event object or dictionary into the chat_events schema."""
        if hasattr(event, "model_dump"):
            d = event.model_dump()
        elif hasattr(event, "__dict__"):
            d = dict(event.__dict__)
        else:
            d = dict(event)

        streamer_id = str(d.get("streamer_id") or "")
        vod_id = str(d.get("vod_id") or "")

        ts_sec = float(d.get("timestamp_sec") or d.get("timestamp") or 0.0)
        ts_ms = d.get("timestamp_ms")
        if ts_ms is None:
            ts_ms = int(ts_sec * 1000)
        else:
            ts_ms = int(ts_ms)

        chatter = str(d.get("chatter_username") or d.get("username") or d.get("author") or "")
        text = str(d.get("message_text") or d.get("message") or d.get("text") or "")

        emotes_val = d.get("emotes") or []
        if isinstance(emotes_val, str):
            emotes = [e.strip() for e in emotes_val.split(",") if e.strip()]
        elif isinstance(emotes_val, (list, tuple)):
            emotes = [str(e) for e in emotes_val]
        else:
            emotes = []

        sentiment = float(d.get("sentiment_score") or 0.0)
        is_sub = 1 if d.get("is_subscriber") else 0
        burst = 1 if d.get("burst_flag") else 0

        return {
            "streamer_id": streamer_id,
            "vod_id": vod_id,
            "timestamp_ms": ts_ms,
            "timestamp_sec": float(ts_sec),
            "chatter_username": chatter,
            "message_text": text,
            "emotes": emotes,
            "sentiment_score": sentiment,
            "is_subscriber": is_sub,
            "burst_flag": burst,
        }

    @staticmethod
    def _normalize_speech_segment(segment: Union[Dict[str, Any], Any]) -> Dict[str, Any]:
        """Normalizes a speech segment object or dictionary into the speech_segments schema."""
        if hasattr(segment, "model_dump"):
            d = segment.model_dump()
        elif hasattr(segment, "__dict__"):
            d = dict(segment.__dict__)
        else:
            d = dict(segment)

        streamer_id = str(d.get("streamer_id") or "")
        vod_id = str(d.get("vod_id") or "")

        start_ms = d.get("start_ms")
        if start_ms is None:
            start_sec = float(d.get("start_sec") or d.get("start_time") or 0.0)
            start_ms = int(start_sec * 1000)
        else:
            start_ms = int(start_ms)

        end_ms = d.get("end_ms")
        if end_ms is None:
            end_sec = float(d.get("end_sec") or d.get("end_time") or 0.0)
            end_ms = int(end_sec * 1000)
        else:
            end_ms = int(end_ms)

        speaker_id = str(d.get("speaker_id") or "SPEAKER_00")
        transcript = str(d.get("transcript") or d.get("text") or "")

        return {
            "streamer_id": streamer_id,
            "vod_id": vod_id,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "speaker_id": speaker_id,
            "transcript": transcript,
        }

    def insert_chat_events_batch(self, events: List[Union[Dict[str, Any], Any]]) -> int:
        """Batch inserts raw chat events into ClickHouse."""
        if not events:
            return 0

        normalized = [self._normalize_chat_event(e) for e in events]

        with self._lock:
            if not self.is_mock:
                try:
                    payload_lines = "\n".join(json.dumps(row) for row in normalized).encode("utf-8")
                    self._execute_http("INSERT INTO chat_events FORMAT JSONEachRow", post_data=payload_lines)
                    return len(normalized)
                except Exception as e:
                    logger.warning(f"ClickHouse batch insert failed ({e}); switching to mock mode.")
                    self.is_mock = True

            # In mock mode
            self._mock_chat_events.extend(normalized)
            return len(normalized)

    def insert_speech_segments_batch(self, segments: List[Union[Dict[str, Any], Any]]) -> int:
        """Batch inserts speech segments into ClickHouse."""
        if not segments:
            return 0

        normalized = [self._normalize_speech_segment(s) for s in segments]

        with self._lock:
            if not self.is_mock:
                try:
                    payload_lines = "\n".join(json.dumps(row) for row in normalized).encode("utf-8")
                    self._execute_http("INSERT INTO speech_segments FORMAT JSONEachRow", post_data=payload_lines)
                    return len(normalized)
                except Exception as e:
                    logger.warning(f"ClickHouse speech segments insert failed ({e}); switching to mock mode.")
                    self.is_mock = True

            # In mock mode
            self._mock_speech_segments.extend(normalized)
            return len(normalized)

    def generate_asof_query(
        self,
        vod_id: str,
        window_sec: float = 8.0,
        streamer_id: Optional[str] = None,
    ) -> str:
        """Generates standard ASOF JOIN query aligning speech segments with chat reactions."""
        window_ms = int(window_sec * 1000)
        streamer_filter = f" AND s.streamer_id = '{streamer_id}'" if streamer_id else ""
        return (
            "SELECT \n"
            "    s.vod_id, \n"
            "    s.start_ms, \n"
            "    s.transcript, \n"
            "    c.chatter_username, \n"
            "    c.message_text, \n"
            "    c.emotes\n"
            "FROM speech_segments s\n"
            "ASOF LEFT JOIN chat_events c\n"
            "  ON s.vod_id = c.vod_id \n"
            " AND s.streamer_id = c.streamer_id\n"
            " AND s.start_ms <= c.timestamp_ms\n"
            f"WHERE s.vod_id = '{vod_id}'{streamer_filter}\n"
            f"  AND c.timestamp_ms <= s.start_ms + {window_ms}"
        )

    def asof_align_chat_reactions(
        self,
        vod_id: str,
        window_sec: float = 8.0,
        streamer_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Executes ASOF JOIN between speech segments and subsequent chat messages within window_sec."""
        with self._lock:
            if not self.is_mock:
                query = self.generate_asof_query(vod_id, window_sec, streamer_id) + " FORMAT JSONEachRow"
                try:
                    res = self._execute_http(query)
                    if res:
                        return [json.loads(line) for line in res.strip().split("\n") if line.strip()]
                    return []
                except Exception:
                    self.is_mock = True

            # Mock in-memory ASOF JOIN evaluation
            window_ms = int(window_sec * 1000)
            segments = [
                s for s in self._mock_speech_segments
                if s["vod_id"] == vod_id and (streamer_id is None or s["streamer_id"] == streamer_id)
            ]
            segments.sort(key=lambda x: x["start_ms"])

            chats = [
                c for c in self._mock_chat_events
                if c["vod_id"] == vod_id and (streamer_id is None or c["streamer_id"] == streamer_id)
            ]
            chats.sort(key=lambda x: x["timestamp_ms"])

            aligned: List[Dict[str, Any]] = []
            for seg in segments:
                s_ms = seg["start_ms"]
                # Find closest subsequent chat event where s.start_ms <= c.timestamp_ms <= s.start_ms + window_ms
                matched_chat = None
                for c in chats:
                    if c["streamer_id"] == seg["streamer_id"] and s_ms <= c["timestamp_ms"] <= s_ms + window_ms:
                        matched_chat = c
                        break

                aligned.append({
                    "vod_id": seg["vod_id"],
                    "start_ms": seg["start_ms"],
                    "transcript": seg["transcript"],
                    "chatter_username": matched_chat["chatter_username"] if matched_chat else None,
                    "message_text": matched_chat["message_text"] if matched_chat else None,
                    "emotes": matched_chat["emotes"] if matched_chat else [],
                })
            return aligned

    def get_chat_events_count(self, vod_id: Optional[str] = None) -> int:
        """Returns total count of chat events stored."""
        with self._lock:
            if not self.is_mock:
                where_clause = f" WHERE vod_id = '{vod_id}'" if vod_id else ""
                try:
                    res = self._execute_http(f"SELECT count() FROM chat_events{where_clause}")
                    if res:
                        return int(res.strip())
                except Exception:
                    self.is_mock = True

            if vod_id:
                return sum(1 for e in self._mock_chat_events if e["vod_id"] == vod_id)
            return len(self._mock_chat_events)

    def get_speech_segments_count(self, vod_id: Optional[str] = None) -> int:
        """Returns total count of speech segments stored."""
        with self._lock:
            if not self.is_mock:
                where_clause = f" WHERE vod_id = '{vod_id}'" if vod_id else ""
                try:
                    res = self._execute_http(f"SELECT count() FROM speech_segments{where_clause}")
                    if res:
                        return int(res.strip())
                except Exception:
                    self.is_mock = True

            if vod_id:
                return sum(1 for s in self._mock_speech_segments if s["vod_id"] == vod_id)
            return len(self._mock_speech_segments)

    def close(self) -> None:
        """Closes any active sessions."""
        pass
