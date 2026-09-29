"""Chatter Profiling, Banter vs. Griefing Classification & Community Safety (Spec 12).

Maintains persistent historical chatter activity, distinguishes playful roasting and
gaming banter from bad-faith griefing and brigading, and flags suspicious accounts.
"""

from datetime import datetime, timezone
import math
import re
import sqlite3
from typing import Dict, List, Optional, Set, Tuple

from stream_fusion.chat.nlp import ChatNLPAnalyzer, jaccard_similarity, tokenize_for_similarity
from stream_fusion.models.schemas import (
    BrigadeCluster,
    ChatterDetailedProfile,
    ChatterSafetyVerdict,
    ChatMessage,
)


WHITELISTED_BOTS = {
    "nightbot",
    "streamelements",
    "moobot",
    "fossabot",
    "soundalerts",
    "wizebot",
}

HOSTILE_TOKENS = {
    "loser", "kill", "die", "ugly", "fake", "dox", "doxx", "hate",
    "unsub", "unsubbed", "nobody cares", "irrelevant", "stfu",
}

BANTER_TOKENS = {
    "trash", "washed", "throw", "threw", "cringe", "bald", "greasy", "omegalul",
    "kekw", "lul", "icant", "lmao", "bad",
}


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-50.0, min(50.0, x))))


class BanterClassifier:
    """Disambiguates good-natured gaming banter/roasting from malicious harassment and bad-faith griefing."""

    def __init__(self):
        self.nlp = ChatNLPAnalyzer()

    def is_bot_message(self, message: ChatMessage) -> bool:
        """Determines if a message originates from an automated channel bot."""
        author_lower = message.author_name.lower().replace(" ", "")
        if any(bot in author_lower for bot in WHITELISTED_BOTS):
            return True
        content_stripped = message.content.strip()
        if content_stripped.startswith("!"):
            return True
        if re.search(r"^(welcome to the stream|follow the channel|join the discord|check out)", content_stripped, re.IGNORECASE):
            return True
        return False

    def classify_message(
        self,
        message: ChatMessage,
        is_streamer_failure_moment: bool = False,
        is_amusement_spike: bool = False,
        streamer_topic_is_serious: bool = False,
    ) -> ChatterSafetyVerdict:
        """Assigns contextual safety verdict to a chat message."""
        content_lower = message.content.lower()
        tokens = tokenize_for_similarity(content_lower)

        # 1. Malicious Harassment Check (Extreme doxxing / threat tokens)
        if any(threat in content_lower for threat in ["doxx you", "kill yourself", "kys"]):
            return ChatterSafetyVerdict(
                message_id=message.message_id,
                user_id=message.user_id,
                verdict="MALICIOUS_HARASSMENT",
                confidence=0.99,
                reason="Severe personal threat or explicit harassment violation",
            )

        # 2. Contextual Banter vs Griefing Check
        has_banter_tokens = bool(tokens & BANTER_TOKENS)
        has_hostile_tokens = bool(tokens & HOSTILE_TOKENS)

        # If streamer just died or failed in a game, or chat is laughing:
        if is_streamer_failure_moment or is_amusement_spike:
            if has_banter_tokens or "l" in content_lower.split():
                return ChatterSafetyVerdict(
                    message_id=message.message_id,
                    user_id=message.user_id,
                    verdict="GOOD_NATURED_BANTER",
                    confidence=0.90,
                    reason="Playful roasting synchronized with streamer misplay or amusement spike",
                )

        # Community roasts on appearance / bald / greasy when not a serious topic
        if not streamer_topic_is_serious and any(k in content_lower for k in ["bald", "greasy", "hairline", "washed"]):
            return ChatterSafetyVerdict(
                message_id=message.message_id,
                user_id=message.user_id,
                verdict="COMMUNITY_ROAST",
                confidence=0.85,
                reason="Traditional streamer-chat running joke / community roast",
            )

        # If topic is serious or unprovoked persistent hostility:
        if streamer_topic_is_serious and (has_hostile_tokens or has_banter_tokens):
            return ChatterSafetyVerdict(
                message_id=message.message_id,
                user_id=message.user_id,
                verdict="BAD_FAITH_GRIEFING",
                confidence=0.88,
                reason="Hostile interruption during serious streamer monologue",
            )

        if has_hostile_tokens:
            return ChatterSafetyVerdict(
                message_id=message.message_id,
                user_id=message.user_id,
                verdict="BAD_FAITH_GRIEFING",
                confidence=0.80,
                reason="Unprovoked hostility and negative contrarian stance",
            )

        return ChatterSafetyVerdict(
            message_id=message.message_id,
            user_id=message.user_id,
            verdict="GOOD_NATURED_BANTER" if has_banter_tokens else "BENIGN_CHAT",
            confidence=0.75,
            reason="Standard community interaction",
        )


class ChatterProfileStore:
    """Manages historical chatter profiles and aggregates behavioral metrics in SQLite."""

    def __init__(self, db_path: str = ":memory:"):
        self.conn = sqlite3.connect(db_path)
        self._init_db()
        self.classifier = BanterClassifier()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chatters (
                user_id TEXT PRIMARY KEY,
                username TEXT NOT NULL,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                total_messages INT DEFAULT 0,
                contrarian_index REAL DEFAULT 0.0,
                hostility_index REAL DEFAULT 0.0,
                banter_reciprocity REAL DEFAULT 0.0,
                griefer_score REAL DEFAULT 0.0,
                flagged_status TEXT DEFAULT 'CLEAN',
                is_automated_bot INT DEFAULT 0
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS chatter_messages (
                message_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                vod_id TEXT NOT NULL,
                timestamp_offset REAL NOT NULL,
                content TEXT NOT NULL,
                verdict TEXT NOT NULL,
                is_bot INT DEFAULT 0
            )
        """)
        self.conn.commit()

    def ingest_message(
        self,
        message: ChatMessage,
        vod_id: str = "unknown_vod",
        is_streamer_failure_moment: bool = False,
        is_amusement_spike: bool = False,
        streamer_topic_is_serious: bool = False,
    ) -> ChatterSafetyVerdict:
        """Processes and records a chat message, updating chatter behavioral scores."""
        is_bot = self.classifier.is_bot_message(message)
        verdict = self.classifier.classify_message(
            message,
            is_streamer_failure_moment=is_streamer_failure_moment,
            is_amusement_spike=is_amusement_spike,
            streamer_topic_is_serious=streamer_topic_is_serious,
        )

        cur = self.conn.cursor()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Insert message
        cur.execute("""
            INSERT OR REPLACE INTO chatter_messages (message_id, user_id, vod_id, timestamp_offset, content, verdict, is_bot)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (message.message_id, message.user_id, vod_id, message.timestamp_offset, message.content, verdict.verdict, int(is_bot)))

        # Upsert chatter
        cur.execute("SELECT total_messages, first_seen_at FROM chatters WHERE user_id = ?", (message.user_id,))
        row = cur.fetchone()
        if row:
            total_msgs = row[0] + 1
            first_seen = row[1]
        else:
            total_msgs = 1
            first_seen = now_iso

        # Compute updated metrics
        cur.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN verdict IN ('BAD_FAITH_GRIEFING', 'MALICIOUS_HARASSMENT') THEN 1 ELSE 0 END) as hostile,
                SUM(CASE WHEN verdict IN ('GOOD_NATURED_BANTER', 'COMMUNITY_ROAST') THEN 1 ELSE 0 END) as banter
            FROM chatter_messages WHERE user_id = ?
        """, (message.user_id,))
        stats = cur.fetchone()
        t_count = stats[0] or 1
        h_count = stats[1] or 0
        b_count = stats[2] or 0

        hostility_idx = float(h_count) / float(t_count)
        banter_reciprocity = float(b_count) / float(t_count)
        contrarian_idx = hostility_idx  # In basic form, hostile messages oppose the streamer

        # Griefer score formula:
        # P_griefer = sigmoid(3.0 * contrarian + 2.0 * hostility - 2.5 * banter - 1.5)
        raw_score = 3.0 * contrarian_idx + 2.0 * hostility_idx - 2.5 * banter_reciprocity - 1.5
        griefer_score = round(sigmoid(raw_score), 3)

        flagged_status = "CLEAN"
        if griefer_score >= 0.75:
            flagged_status = "GRIEFER"
        elif griefer_score >= 0.50:
            flagged_status = "WATCHLIST"

        cur.execute("""
            INSERT OR REPLACE INTO chatters (
                user_id, username, first_seen_at, last_seen_at, total_messages,
                contrarian_index, hostility_index, banter_reciprocity, griefer_score,
                flagged_status, is_automated_bot
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            message.user_id,
            message.author_name,
            first_seen,
            now_iso,
            total_msgs,
            round(contrarian_idx, 3),
            round(hostility_idx, 3),
            round(banter_reciprocity, 3),
            griefer_score,
            flagged_status,
            int(is_bot),
        ))

        self.conn.commit()
        return verdict

    def get_chatter_profile(self, user_id: str) -> Optional[ChatterDetailedProfile]:
        """Retrieves profile and scores for a specific user."""
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM chatters WHERE user_id = ?", (user_id,))
        row = cur.fetchone()
        if not row:
            return None
        return ChatterDetailedProfile(
            user_id=row[0],
            username=row[1],
            first_seen_at=row[2],
            last_seen_at=row[3],
            total_messages=row[4],
            contrarian_index=row[5],
            hostility_index=row[6],
            banter_reciprocity=row[7],
            griefer_score=row[8],
            flagged_status=row[9],
            is_automated_bot=bool(row[10]),
        )


class BrigadeDetector:
    """Detects synchronized coordinated raids and political brigading clusters."""

    def __init__(self, window_sec: float = 120.0, similarity_threshold: float = 0.70, min_accounts: int = 5):
        self.window_sec = window_sec
        self.similarity_threshold = similarity_threshold
        self.min_accounts = min_accounts

    def detect_brigades(self, messages: List[ChatMessage]) -> List[BrigadeCluster]:
        """Scans for clusters of new accounts firing similar hostile messages simultaneously."""
        if len(messages) < self.min_accounts:
            return []

        sorted_msgs = sorted(messages, key=lambda m: m.timestamp_offset)
        token_sets = [tokenize_for_similarity(m.content) for m in sorted_msgs]
        clusters: List[BrigadeCluster] = []

        visited: Set[int] = set()
        cluster_idx = 0

        for i in range(len(sorted_msgs)):
            if i in visited:
                continue
            t_i = token_sets[i]
            if len(t_i) < 2:
                continue

            current_group = [i]
            start_t = sorted_msgs[i].timestamp_offset

            for j in range(i + 1, len(sorted_msgs)):
                if j in visited:
                    continue
                if (sorted_msgs[j].timestamp_offset - start_t) > self.window_sec:
                    break

                t_j = token_sets[j]
                sim = jaccard_similarity(t_i, t_j)
                if sim >= self.similarity_threshold:
                    current_group.append(j)

            unique_users = {sorted_msgs[idx].user_id for idx in current_group}
            if len(unique_users) >= self.min_accounts:
                visited.update(current_group)
                end_t = sorted_msgs[current_group[-1]].timestamp_offset
                rep_text = sorted_msgs[current_group[0]].content

                clusters.append(
                    BrigadeCluster(
                        cluster_id=f"brigade_{cluster_idx}",
                        window_start_sec=round(start_t, 2),
                        window_end_sec=round(end_t, 2),
                        participant_user_ids=list(unique_users),
                        similarity_score=self.similarity_threshold,
                        flagged_phrase=rep_text,
                    )
                )
                cluster_idx += 1

        return clusters


# Backward-compatibility alias
ChatterProfiler = ChatterProfileStore

