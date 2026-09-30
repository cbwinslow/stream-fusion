"""Brand Safety & Accidental Screen Leak Scanner (Spec 33).

Detects sensitive personal information (credit cards, emails, IP addresses, auth tokens)
and brand safety risks on screen.
"""

import logging
import re
from typing import List

from stream_fusion.models.schemas import BrandSafetyAlert, VisualKeyframe

logger = logging.getLogger(__name__)

# Regular expression patterns for sensitive data
CREDIT_CARD_PATTERN = re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b")
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
IPV4_PATTERN = re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b")
AUTH_TOKEN_PATTERN = re.compile(r"(?:Bearer\s+[A-Za-z0-9_\-\.]{20,}|token=[A-Za-z0-9_\-\.]{16,})", re.IGNORECASE)


class BrandSafetyScanner:
    """Scans visual OCR text and telemetry for accidental leaks and TOS risks."""

    def scan_ocr_for_leaks(self, keyframes: List[VisualKeyframe]) -> List[BrandSafetyAlert]:
        """Scans on-screen OCR text across all keyframes for sensitive information leaks."""
        alerts: List[BrandSafetyAlert] = []

        for kf in keyframes:
            ts = float(getattr(kf, "timestamp_sec", 0.0))
            blocks = getattr(kf, "ocr_text_blocks", []) or []
            if isinstance(blocks, list):
                screen_text = " ".join(blocks)
            else:
                screen_text = str(blocks)
            if not screen_text:
                screen_text = getattr(kf, "screen_summary", "") or getattr(kf, "screen_text", "") or ""
            if not screen_text:
                continue

            # 1. Credit Card Check
            if cc_match := CREDIT_CARD_PATTERN.search(screen_text):
                matched_val = cc_match.group(0)
                # Filter out obvious false positives like 0000-0000-0000-0000
                if len(set(matched_val.replace(" ", "").replace("-", ""))) > 1:
                    alerts.append(
                        BrandSafetyAlert(
                            timestamp_sec=ts,
                            risk_type="ocr_leak",
                            description=f"Potential credit card pattern visible on screen: {matched_val[:4]} **** **** {matched_val[-4:]}",
                            severity="critical",
                            redaction_recommended=True,
                        )
                    )

            # 2. Email Address Check
            if email_match := EMAIL_PATTERN.search(screen_text):
                alerts.append(
                    BrandSafetyAlert(
                        timestamp_sec=ts,
                        risk_type="ocr_leak",
                        description=f"Private email address visible on screen: {email_match.group(0)}",
                        severity="warning",
                        redaction_recommended=True,
                    )
                )

            # 3. Auth Token Check
            if AUTH_TOKEN_PATTERN.search(screen_text):
                alerts.append(
                    BrandSafetyAlert(
                        timestamp_sec=ts,
                        risk_type="ocr_leak",
                        description="Private API token or Bearer authorization string visible on screen",
                        severity="critical",
                        redaction_recommended=True,
                    )
                )

            # 4. Public IPv4 Address Check
            if ip_match := IPV4_PATTERN.search(screen_text):
                ip_str = ip_match.group(0)
                # Ignore loopback 127.0.0.1 or standard 0.0.0.0
                if not (ip_str.startswith("127.") or ip_str == "0.0.0.0"):
                    alerts.append(
                        BrandSafetyAlert(
                            timestamp_sec=ts,
                            risk_type="ocr_leak",
                            description=f"IP address visible on screen: {ip_str}",
                            severity="warning",
                            redaction_recommended=True,
                        )
                    )

        logger.info("Brand Safety OCR scan completed: %d alerts detected", len(alerts))
        return alerts
