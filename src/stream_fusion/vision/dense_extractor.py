"""Dense Frame Extraction, Screen OCR & Web/Social Post Grounding (Spec 14).

Extracts Twitter/X post cards, browser URLs/domains, gaming HUD states,
and synchronizes spoken words with on-screen text bounding boxes (Read-Along Alignment).
"""

import re
from typing import Dict, List, Optional, Set, Tuple

from stream_fusion.chat.nlp import tokenize_for_similarity
from stream_fusion.models.schemas import (
    ReadAlongSegment,
    ScreenGameContext,
    ScreenWebContext,
    SocialPostCard,
    WordTiming,
)


class SocialCardParser:
    """Parses on-screen social media post cards (X.com / Twitter, Reddit, etc.) from OCR blocks."""

    # Matches Twitter/X handle pattern: @username
    HANDLE_PATTERN = re.compile(r"(@[A-Za-z0-9_]{1,25})")
    URL_PATTERN = re.compile(r"(https?://[^\s]+|(?:www\.)?[a-zA-Z0-9-]+\.[a-z]{2,}(?:/[^\s]*)?)")

    def parse_social_cards(self, ocr_blocks: List[str]) -> List[SocialPostCard]:
        """Detects Twitter/X post cards from OCR text blocks."""
        cards: List[SocialPostCard] = []

        for block in ocr_blocks:
            lines = [line.strip() for line in block.split("\n") if line.strip()]
            for i, line in enumerate(lines):
                handle_match = self.HANDLE_PATTERN.search(line)
                if handle_match:
                    handle = handle_match.group(1)
                    # Extract author name if on the same line or preceding
                    name_cand = line.replace(handle, "").strip()
                    if not name_cand and i > 0:
                        name_cand = lines[i - 1]
                    display_name = name_cand if name_cand else handle.lstrip("@")

                    # Remaining lines form the post body text
                    body_lines = [l for j, l in enumerate(lines) if j != i and l != name_cand]
                    post_text = " ".join(body_lines).strip()
                    if not post_text:
                        post_text = line

                    cards.append(
                        SocialPostCard(
                            platform="X_TWITTER",
                            author_handle=handle,
                            author_name=display_name,
                            post_text=post_text,
                            bounding_box=[0.2, 0.2, 0.7, 0.8],  # default normalized ROI
                            has_embedded_media="pic.twitter.com" in post_text or "t.co" in post_text,
                        )
                    )

        return cards

    def extract_browser_url(self, ocr_blocks: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """Extracts active URL and domain from browser address bar OCR.

        Returns (url, domain).
        """
        for block in ocr_blocks:
            matches = self.URL_PATTERN.findall(block)
            for m in matches:
                m_clean = m.lower().strip()
                if any(domain in m_clean for domain in ["x.com", "twitter.com", "reddit.com", "youtube.com", "ign.com", "twitch.tv"]):
                    domain_match = re.search(r"([a-z0-9-]+\.[a-z]{2,})", m_clean)
                    domain = domain_match.group(1) if domain_match else None
                    return m, domain

        return None, None


class GameHUDParser:
    """Parses gaming HUD elements: health/mana bars, minimap, and combat killfeed."""

    KILLFEED_TRIGGERS = {"defeated", "killed", "slain", "headshot", "eliminated", "downed"}

    def parse_game_context(self, ocr_blocks: List[str], timestamp_sec: float) -> ScreenGameContext:
        """Extracts game HUD information from OCR text and bounding elements."""
        killfeed: List[str] = []
        for block in ocr_blocks:
            lower = block.lower()
            if any(trig in lower for trig in self.KILLFEED_TRIGGERS):
                killfeed.append(block.strip())

        return ScreenGameContext(
            timestamp_sec=timestamp_sec,
            game_title_hint=None,
            minimap_box=[0.02, 0.80, 0.25, 0.98],
            health_percentage=None,
            mana_percentage=None,
            killfeed_entries=killfeed,
            inventory_open=any("inventory" in b.lower() or "equipment" in b.lower() for b in ocr_blocks),
        )


class ReadAlongAligner:
    """Calculates synchronization score between spoken speech and on-screen text blocks."""

    def __init__(self, min_alignment_threshold: float = 0.55):
        self.min_alignment_threshold = min_alignment_threshold

    def compute_alignment(
        self,
        spoken_words: List[WordTiming],
        on_screen_text: str,
        start_sec: float,
        end_sec: float,
        source_handle: Optional[str] = None,
    ) -> Optional[ReadAlongSegment]:
        """Aligns spoken words with screen text."""
        if not spoken_words or not on_screen_text.strip():
            return None

        speech_text = " ".join(w.word for w in spoken_words)
        speech_tokens = tokenize_for_similarity(speech_text)
        screen_tokens = tokenize_for_similarity(on_screen_text)

        if not speech_tokens:
            return None

        intersection = len(speech_tokens & screen_tokens)
        score = float(intersection) / float(len(speech_tokens))

        if score >= self.min_alignment_threshold:
            duration_min = max(0.01, (end_sec - start_sec) / 60.0)
            wpm = len(spoken_words) / duration_min

            return ReadAlongSegment(
                start_sec=round(start_sec, 2),
                end_sec=round(end_sec, 2),
                spoken_text=speech_text,
                matched_screen_text=on_screen_text.strip(),
                alignment_score=round(score, 3),
                reading_wpm=round(wpm, 1),
                source_handle=source_handle,
            )

        return None


class DenseFrameExtractor:
    """Coordinates deep visual feature extraction for every 1.0s video keyframe."""

    def __init__(self):
        self.social_parser = SocialCardParser()
        self.hud_parser = GameHUDParser()
        self.read_along_aligner = ReadAlongAligner()

    def process_dense_frame(
        self,
        ocr_blocks: List[str],
        timestamp_sec: float,
        spoken_words_in_window: Optional[List[WordTiming]] = None,
    ) -> Tuple[ScreenWebContext, ScreenGameContext, Optional[ReadAlongSegment]]:
        """Performs continuous deep parsing of a frame's visual and reading context."""
        # 1. Social post cards
        cards = self.social_parser.parse_social_cards(ocr_blocks)

        # 2. Browser URL & domain
        url, domain = self.social_parser.extract_browser_url(ocr_blocks)
        browser_detected = bool(url or domain or cards)

        web_context = ScreenWebContext(
            timestamp_sec=timestamp_sec,
            browser_detected=browser_detected,
            detected_url=url,
            domain=domain,
            social_post_cards=cards,
            article_headlines=[],
        )

        # 3. Game HUD
        game_context = self.hud_parser.parse_game_context(ocr_blocks, timestamp_sec=timestamp_sec)

        # 4. Read-Along Alignment
        read_along = None
        if spoken_words_in_window and cards:
            best_card = cards[0]
            read_along = self.read_along_aligner.compute_alignment(
                spoken_words=spoken_words_in_window,
                on_screen_text=best_card.post_text,
                start_sec=timestamp_sec,
                end_sec=timestamp_sec + 2.0,
                source_handle=best_card.author_handle,
            )

        return web_context, game_context, read_along
