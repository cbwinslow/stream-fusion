"""Universal Message Normalizer converting Twitch, Kick, and YouTube payloads to ChatMessage (Spec 22)."""

from datetime import datetime, timezone
import json
import logging
import re
import time
from typing import Any, Dict, List, Optional
import uuid

from stream_fusion.models.schemas import ChatEmote, ChatMessage

logger = logging.getLogger(__name__)

# Kick emote syntax: [emote:2506823:azzzjh]
KICK_EMOTE_REGEX = re.compile(r"\[emote:(\d+):([a-zA-Z0-9_]+)\]")


class MessageNormalizer:
    """Standardizes heterogeneous streaming chat wire formats into canonical StreamFusion ChatMessage models."""

    @staticmethod
    def parse_kick_emotes(content: str) -> List[ChatEmote]:
        """Extracts Kick inline emotes matching [emote:ID:name]."""
        emotes: List[ChatEmote] = []
        counts: Dict[str, int] = {}
        names: Dict[str, str] = {}

        for match in KICK_EMOTE_REGEX.finditer(content):
            emote_id = match.group(1)
            emote_name = match.group(2)
            counts[emote_id] = counts.get(emote_id, 0) + 1
            names[emote_id] = emote_name

        for emote_id, count in counts.items():
            emotes.append(ChatEmote(id=emote_id, name=names[emote_id], count=count))
        return emotes

    @classmethod
    def from_kick_pusher(
        cls,
        payload: Any,
        stream_start_ms: Optional[float] = None,
    ) -> Optional[ChatMessage]:
        """Normalizes a Kick Pusher ChatMessageEvent payload into a ChatMessage."""
        if isinstance(payload, str):
            try:
                data = json.loads(payload)
            except json.JSONDecodeError:
                return None
        elif isinstance(payload, dict):
            data = payload
        else:
            return None

        # Kick ChatMessageEvent schema
        msg_id = str(data.get("id") or uuid.uuid4())
        content = data.get("content", "")
        if not content and "message" in data:
            content = data.get("message", "")

        # Sender identity
        sender = data.get("sender") or {}
        user_id = str(sender.get("id") or data.get("user_id") or "kick_anon")
        author_name = str(sender.get("username") or sender.get("slug") or "Anonymous")

        # Badges
        badges: List[str] = []
        identity = sender.get("identity") or {}
        raw_badges = identity.get("badges") or []
        for b in raw_badges:
            if isinstance(b, dict):
                b_type = b.get("type", "")
                b_count = b.get("count")
                if b_count is not None:
                    badges.append(f"{b_type}/{b_count}")
                elif b_type:
                    badges.append(b_type)
            elif isinstance(b, str):
                badges.append(b)

        # Emotes
        emotes = cls.parse_kick_emotes(content)

        # Timestamp offset
        created_at = data.get("created_at")
        timestamp_offset = 0.0
        now = time.time()
        start_sec = (stream_start_ms / 1000.0) if stream_start_ms else now

        if isinstance(created_at, (int, float)):
            # epoch seconds or ms
            ts = float(created_at)
            if ts > 1e11:  # ms
                ts /= 1000.0
            timestamp_offset = max(0.0, ts - start_sec)
        elif isinstance(created_at, str):
            try:
                # ISO formats: 2026-09-28T16:00:00Z or 2026-09-28 16:00:00
                cleaned = created_at.replace("Z", "+00:00")
                dt = datetime.fromisoformat(cleaned)
                ts = dt.timestamp()
                timestamp_offset = max(0.0, ts - start_sec)
            except Exception:
                timestamp_offset = max(0.0, now - start_sec)
        else:
            timestamp_offset = max(0.0, now - start_sec)

        metadata: Dict[str, Any] = {
            "platform": "KICK",
            "chatroom_id": data.get("chatroom_id"),
        }
        if "type" in data:
            metadata["kick_message_type"] = data["type"]

        return ChatMessage(
            message_id=msg_id,
            timestamp_offset=timestamp_offset,
            user_id=user_id,
            author_name=author_name,
            content=content,
            emotes=emotes,
            badges=badges,
            metadata=metadata,
        )

    @classmethod
    def from_youtube_item(
        cls,
        item: Dict[str, Any],
        stream_start_ms: Optional[float] = None,
    ) -> Optional[ChatMessage]:
        """Normalizes a YouTube live chat item (from chat-downloader or InnerTube) into a ChatMessage."""
        if not isinstance(item, dict):
            return None

        # Check for InnerTube raw renderers
        renderer = item.get("liveChatTextMessageRenderer") or item.get("liveChatPaidMessageRenderer")
        if renderer:
            return cls._from_innertube_renderer(renderer, is_paid=("liveChatPaidMessageRenderer" in item), stream_start_ms=stream_start_ms)

        # Standard chat-downloader normalized dictionary
        msg_id = str(item.get("message_id") or item.get("id") or uuid.uuid4())
        
        # Author details
        author_info = item.get("author") or {}
        if isinstance(author_info, str):
            author_name = author_info
            user_id = author_info.lower()
            badges = []
        else:
            author_name = str(author_info.get("name") or author_info.get("display_name") or "Anonymous")
            user_id = str(author_info.get("id") or author_info.get("channel_id") or author_name.lower())
            raw_badges = author_info.get("badges") or []
            badges = []
            for b in raw_badges:
                if isinstance(b, dict):
                    b_title = b.get("title") or b.get("label") or b.get("type", "")
                    if b_title:
                        badges.append(b_title)
                elif isinstance(b, str):
                    badges.append(b)

        # Message text & emotes
        raw_msg = item.get("message")
        emotes: List[ChatEmote] = []
        content = ""

        if isinstance(raw_msg, str):
            content = raw_msg
        elif isinstance(raw_msg, list):
            # runs list
            parts = []
            for run in raw_msg:
                if isinstance(run, dict):
                    if "text" in run:
                        parts.append(run["text"])
                    elif "emoji" in run:
                        em_data = run["emoji"]
                        em_id = em_data.get("emoji_id") or em_data.get("id", "yt_emoji")
                        em_name = em_data.get("shortcuts", [":emoji:"])[0] if em_data.get("shortcuts") else ":emoji:"
                        parts.append(em_name)
                        emotes.append(ChatEmote(id=str(em_id), name=em_name, count=1))
                elif isinstance(run, str):
                    parts.append(run)
            content = "".join(parts)
        elif isinstance(raw_msg, dict):
            content = str(raw_msg.get("text") or "")

        # Also collect emotes array if provided directly by chat-downloader
        if "emotes" in item and isinstance(item["emotes"], list):
            for em in item["emotes"]:
                if isinstance(em, dict):
                    eid = str(em.get("id") or em.get("emoticon_id") or "")
                    ename = str(em.get("name") or em.get("text") or eid)
                    if eid:
                        emotes.append(ChatEmote(id=eid, name=ename, count=1))

        # Timestamp offset
        now = time.time()
        start_sec = (stream_start_ms / 1000.0) if stream_start_ms else now
        time_in_sec = item.get("time_in_seconds")
        if time_in_sec is not None:
            timestamp_offset = max(0.0, float(time_in_sec))
        else:
            ts = item.get("timestamp")
            if isinstance(ts, (int, float)):
                if ts > 1e11:  # ms or usec
                    ts = ts / 1e6 if ts > 1e14 else ts / 1000.0
                timestamp_offset = max(0.0, float(ts) - start_sec)
            else:
                timestamp_offset = max(0.0, now - start_sec)

        metadata: Dict[str, Any] = {
            "platform": "YOUTUBE_LIVE",
        }

        # Super Chat / Monetization detection
        money = item.get("money") or item.get("paid") or {}
        if money:
            metadata["monetization"] = {
                "type": "SUPER_CHAT",
                "amount": money.get("amount", 0.0),
                "currency": money.get("currency", "USD"),
                "text": money.get("text", ""),
            }

        return ChatMessage(
            message_id=msg_id,
            timestamp_offset=timestamp_offset,
            user_id=user_id,
            author_name=author_name,
            content=content,
            emotes=emotes,
            badges=badges,
            metadata=metadata,
        )

    @classmethod
    def _from_innertube_renderer(
        cls,
        renderer: Dict[str, Any],
        is_paid: bool,
        stream_start_ms: Optional[float] = None,
    ) -> Optional[ChatMessage]:
        """Normalizes raw InnerTube liveChatTextMessageRenderer or liveChatPaidMessageRenderer."""
        msg_id = renderer.get("id") or str(uuid.uuid4())
        author_name = (
            renderer.get("authorName", {}).get("simpleText")
            or "Anonymous"
        )
        user_id = str(renderer.get("authorExternalChannelId") or author_name.lower())

        # Author Badges
        badges = []
        raw_badges = renderer.get("authorBadges", [])
        for b in raw_badges:
            badge_renderer = b.get("liveChatAuthorBadgeRenderer", {})
            tooltip = badge_renderer.get("tooltip")
            if tooltip:
                badges.append(tooltip)

        # Message Runs
        message_obj = renderer.get("message", {})
        runs = message_obj.get("runs", [])
        content_parts = []
        emotes: List[ChatEmote] = []

        for run in runs:
            if "text" in run:
                content_parts.append(run["text"])
            elif "emoji" in run:
                emoji = run["emoji"]
                eid = emoji.get("emojiId", "emoji")
                shortcuts = emoji.get("shortcuts", [":emoji:"])
                name = shortcuts[0] if shortcuts else ":emoji:"
                content_parts.append(name)
                emotes.append(ChatEmote(id=str(eid), name=name, count=1))

        content = "".join(content_parts)

        # Timestamp
        ts_usec = renderer.get("timestampUsec")
        now = time.time()
        start_sec = (stream_start_ms / 1000.0) if stream_start_ms else now
        if ts_usec:
            try:
                timestamp_offset = max(0.0, (float(ts_usec) / 1_000_000.0) - start_sec)
            except Exception:
                timestamp_offset = 0.0
        else:
            timestamp_offset = max(0.0, now - start_sec)

        metadata: Dict[str, Any] = {
            "platform": "YOUTUBE_LIVE",
        }

        if is_paid:
            amount_text = renderer.get("purchaseAmountText", {}).get("simpleText", "")
            metadata["monetization"] = {
                "type": "SUPER_CHAT",
                "text": amount_text,
                "header_color": renderer.get("headerBackgroundColor"),
                "body_color": renderer.get("bodyBackgroundColor"),
            }

        return ChatMessage(
            message_id=msg_id,
            timestamp_offset=timestamp_offset,
            user_id=user_id,
            author_name=author_name,
            content=content,
            emotes=emotes,
            badges=badges,
            metadata=metadata,
        )

    @classmethod
    def from_twitch_irc(
        cls,
        tags: Dict[str, str],
        prefix: Optional[str],
        content: str,
        stream_start_ms: Optional[float] = None,
    ) -> ChatMessage:
        """Normalizes parsed Twitch IRC message tags into canonical ChatMessage."""
        msg_id = tags.get("id") or str(uuid.uuid4())

        # Author Name
        author_name = tags.get("display-name")
        if not author_name and prefix:
            author_name = prefix.split("!")[0]
        if not author_name:
            author_name = "anonymous"

        # User ID
        user_id = tags.get("user-id") or author_name.lower()

        # Timestamp offset
        tmi_sent_ts = tags.get("tmi-sent-ts")
        now = time.time()
        if tmi_sent_ts:
            try:
                sent_ms = float(tmi_sent_ts)
                if stream_start_ms:
                    timestamp_offset = max(0.0, (sent_ms - stream_start_ms) / 1000.0)
                else:
                    timestamp_offset = sent_ms / 1000.0
            except ValueError:
                timestamp_offset = now
        else:
            timestamp_offset = now

        # Badges
        badges: List[str] = []
        raw_badges = tags.get("badges")
        if raw_badges:
            badges = [b.strip() for b in raw_badges.split(",") if b.strip()]

        # Emotes
        emotes: List[ChatEmote] = []
        raw_emotes = tags.get("emotes")
        if raw_emotes:
            for emote_group in raw_emotes.split("/"):
                if ":" in emote_group:
                    emote_id, spans = emote_group.split(":", 1)
                    span_list = spans.split(",")
                    name = "emote"
                    if span_list and "-" in span_list[0]:
                        try:
                            s, e = map(int, span_list[0].split("-", 1))
                            name = content[s:e + 1]
                        except Exception:
                            name = emote_id
                    emotes.append(ChatEmote(id=emote_id, name=name, count=len(span_list)))

        metadata: Dict[str, Any] = {
            "platform": "TWITCH",
        }
        if "bits" in tags:
            try:
                metadata["monetization"] = {
                    "type": "BITS",
                    "amount": float(tags["bits"]),
                    "currency": "BITS",
                }
            except ValueError:
                pass

        if "color" in tags:
            metadata["user_color"] = tags["color"]

        return ChatMessage(
            message_id=msg_id,
            timestamp_offset=timestamp_offset,
            user_id=user_id,
            author_name=author_name,
            content=content,
            emotes=emotes,
            badges=badges,
            metadata=metadata,
        )
