"""Tests for StreamEntityExtractor."""

from stream_fusion.nlp.entity_extractor import StreamEntityExtractor


def test_extract_from_single_text():
    extractor = StreamEntityExtractor()
    text = "Asmongold is playing World of Warcraft and talking about Blizzard cash shop drama"
    entities = extractor.extract_from_text(text)

    names = {e["name"] for e in entities}
    assert "Asmongold" in names
    assert "World of Warcraft" in names
    assert "Blizzard" in names
    assert "Microtransactions / P2W" in names


def test_extract_corpus_entities():
    extractor = StreamEntityExtractor()
    audio = ["I cannot believe Blizzard added this to World of Warcraft"]
    chat = ["LULW Blizzard", "wow is cooked", "OMEGALUL", "Elden Ring better"]
    ocr = ["World of Warcraft: Grim Batol"]

    entities = extractor.extract_corpus_entities(audio, chat, ocr)

    entity_names = [e.name for e in entities]
    assert "World of Warcraft" in entity_names
    assert "Blizzard" in entity_names

    # Verify OCR and Audio weighted heavily
    wow_entity = next(e for e in entities if e.name == "World of Warcraft")
    assert wow_entity.mention_count >= 5
