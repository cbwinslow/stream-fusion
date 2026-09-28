"""Kick Official Developer Webhook Payload Parser (Spec 22)."""

import hashlib
import hmac
import json
import logging
from typing import Any, Dict, Optional

from stream_fusion.connectors.normalizer import MessageNormalizer
from stream_fusion.models.schemas import ChatMessage

logger = logging.getLogger(__name__)


class KickWebhookReceiver:
    """Parses and validates official Kick Developer Portal webhook events.
    
    Reference: developers.kick.com / events:subscribe / chat.message.sent
    """

    def __init__(self, webhook_secret: Optional[str] = None):
        self.webhook_secret = webhook_secret

    def verify_signature(self, raw_body: bytes, signature_header: str) -> bool:
        """Verifies HMAC SHA-256 webhook signature if secret is configured."""
        if not self.webhook_secret:
            return True
        if not signature_header:
            return False

        computed = hmac.new(
            self.webhook_secret.encode("utf-8"),
            raw_body,
            hashlib.sha256,
        ).hexdigest()

        return hmac.compare_digest(computed, signature_header.strip())

    def parse_webhook_payload(
        self,
        raw_body: bytes,
        signature_header: Optional[str] = None,
        stream_start_ms: Optional[float] = None,
    ) -> Optional[ChatMessage]:
        """Parses webhook JSON payload and returns normalized ChatMessage if event is chat.message.sent."""
        if signature_header and not self.verify_signature(raw_body, signature_header):
            logger.warning("[KickWebhook] Invalid webhook HMAC signature")
            return None

        try:
            data = json.loads(raw_body.decode("utf-8"))
        except Exception as e:
            logger.warning(f"[KickWebhook] Malformed JSON: {e}")
            return None

        event_type = data.get("event") or data.get("subscription", {}).get("type")
        if event_type not in ("chat.message.sent", "App\\Events\\ChatMessageEvent"):
            # Not a chat message event (e.g. livestream.status.updated)
            return None

        message_data = data.get("data") or data.get("event_data") or data
        return MessageNormalizer.from_kick_pusher(message_data, stream_start_ms=stream_start_ms)
