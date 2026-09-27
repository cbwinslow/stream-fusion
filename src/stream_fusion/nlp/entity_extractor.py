"""Named entity recognition and topic extraction for livestream commentary and chat."""

from collections import Counter
import re
from typing import Dict, List, Set
from pydantic import BaseModel, Field


class StreamEntity(BaseModel):
    name: str
    category: str  # GAME, CREATOR, COMPANY, TOPIC
    mention_count: int
    source_distribution: Dict[str, int] = Field(default_factory=dict)  # CHAT vs AUDIO vs OCR


# Curated knowledge base for livestream gaming and creator culture
KNOWN_ENTITIES = {
    "GAME": {
        "World of Warcraft": [r"\bwow\b", r"\bworld of warcraft\b", r"\bgrim batol\b", r"\bazeroth\b", r"\bwar within\b"],
        "Elden Ring": [r"\belden ring\b", r"\berdtree\b", r"\bmalenia\b", r"\bshadow of the erdtree\b"],
        "Black Myth Wukong": [r"\bwukong\b", r"\bblack myth\b"],
        "Grand Theft Auto VI": [r"\bgta\s?6\b", r"\bgta\s?vi\b", r"\bgrand theft auto\b"],
        "Path of Exile": [r"\bpoe\b", r"\bpath of exile\b", r"\bpoe\s?2\b"],
        "Final Fantasy XIV": [r"\bffxiv\b", r"\bfinal fantasy\b", r"\bdawntrail\b"],
        "League of Legends": [r"\bleague\b", r"\blol\b", r"\bleague of legends\b"],
        "Diablo IV": [r"\bdiablo\b", r"\bdiablo\s?4\b", r"\bvessel of hatred\b"],
    },
    "CREATOR": {
        "Asmongold": [r"\basmon\b", r"\basmongold\b", r"\bzack\b", r"\bzackrawrr\b"],
        "xQc": [r"\bxqc\b", r"\bthe juicer\b"],
        "HasanAbi": [r"\bhasan\b", r"\bhasanabi\b"],
        "Kai Cenat": [r"\bkai\b", r"\bkai cenat\b"],
        "Shroud": [r"\bshroud\b"],
        "PewDiePie": [r"\bpewdiepie\b"],
    },
    "COMPANY": {
        "Blizzard": [r"\bblizzard\b", r"\bactivision\b"],
        "Valve": [r"\bvalve\b", r"\bsteam\b", r"\bgabe\b", r"\bgaben\b"],
        "Twitch": [r"\btwitch\b", r"\btwitchtv\b"],
        "YouTube": [r"\byoutube\b", r"\byt\b"],
        "Sony": [r"\bsony\b", r"\bplaystation\b", r"\bps5\b"],
        "Microsoft": [r"\bmicrosoft\b", r"\bxbox\b"],
        "Ubisoft": [r"\bubisoft\b"],
        "EA": [r"\bea\b", r"\belectronic arts\b"],
    },
    "TOPIC": {
        "Account Ban / Moderation": [r"\bban\b", r"\bbanned\b", r"\bunbanned\b", r"\bsuspension\b", r"\btos\b"],
        "Microtransactions / P2W": [r"\bcash shop\b", r"\bmicrotransaction\b", r"\bp2w\b", r"\bpay to win\b", r"\bbattle pass\b"],
        "Game Balance / Patch": [r"\bpatch\b", r"\bupdate\b", r"\bnerf\b", r"\bbuff\b", r"\bhotfix\b"],
        "Game Release / Beta": [r"\bbeta\b", r"\brelease\b", r"\btrailer\b", r"\bdemo\b", r"\blaunch\b"],
        "Industry Drama": [r"\bdrama\b", r"\blawsuit\b", r"\blayoffs\b", r"\bcontroversy\b"],
    }
}


class StreamEntityExtractor:
    """Extracts named entities and hot topics from multimodal text streams."""

    def __init__(self):
        self._compiled_patterns = {}
        for category, entities in KNOWN_ENTITIES.items():
            self._compiled_patterns[category] = {}
            for entity_name, patterns in entities.items():
                self._compiled_patterns[category][entity_name] = [
                    re.compile(p, re.IGNORECASE) for p in patterns
                ]

    def extract_from_text(self, text: str, source: str = "CHAT") -> List[Dict[str, str]]:
        """Extracts entities mentioned in a single string."""
        if not text:
            return []

        found = []
        for category, entities in self._compiled_patterns.items():
            for entity_name, regex_list in entities.items():
                for rx in regex_list:
                    if rx.search(text):
                        found.append(
                            {
                                "name": entity_name,
                                "category": category,
                                "source": source,
                            }
                        )
                        break
        return found

    def extract_corpus_entities(
        self,
        audio_texts: List[str],
        chat_texts: List[str],
        ocr_texts: List[str],
    ) -> List[StreamEntity]:
        """Extracts and ranks dominant entities across all multimodal text sources."""
        counts = Counter()
        source_breakdown = {}

        # 1. Process Audio
        for txt in audio_texts:
            for item in self.extract_from_text(txt, source="AUDIO"):
                key = (item["name"], item["category"])
                counts[key] += 2  # Streamer speech receives 2x weight
                source_breakdown.setdefault(key, Counter())["AUDIO"] += 1

        # 2. Process Screen OCR
        for txt in ocr_texts:
            for item in self.extract_from_text(txt, source="OCR"):
                key = (item["name"], item["category"])
                counts[key] += 3  # Screen OCR (titles) receives 3x weight
                source_breakdown.setdefault(key, Counter())["OCR"] += 1

        # 3. Process Chat
        for txt in chat_texts:
            for item in self.extract_from_text(txt, source="CHAT"):
                key = (item["name"], item["category"])
                counts[key] += 1
                source_breakdown.setdefault(key, Counter())["CHAT"] += 1

        results = []
        for (name, category), total in counts.most_common():
            dist = dict(source_breakdown.get((name, category), {}))
            results.append(
                StreamEntity(
                    name=name,
                    category=category,
                    mention_count=total,
                    source_distribution=dist,
                )
            )

        return results
