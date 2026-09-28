"""Unit and Integration Tests for Spec 20: Web Grounding & Live Knowledge Graph Expansion."""

from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile
import pytest
from typer.testing import CliRunner

from stream_fusion.cli import app
from stream_fusion.knowledge.temporal_stance import (
    CrossStreamOpinionSynthesizer,
    TemporalStanceShiftTracker,
)
from stream_fusion.knowledge.web_grounding import LiveWebGroundingEngine
from stream_fusion.models.schemas import (
    FactCheckVerdict,
    GroundedClaimResult,
    ScreenWebContext,
    SocialPostCard,
    StancePolarity,
    StreamerClaim,
    WebGroundingCitation,
)
from stream_fusion.schema.agent_rpc import AgentRpcDispatcher


@pytest.fixture
def cli_runner():
    return CliRunner()


# --- 1. Web Grounding Engine Tests ---

def test_query_formulation():
    claim = StreamerClaim(
        claim_id="c1",
        subject="World of Warcraft",
        predicate="is shutting down servers",
        object="servers",
        statement="World of Warcraft is shutting down servers permanently.",
    )
    query = LiveWebGroundingEngine.formulate_search_query(claim)
    assert "World of Warcraft" in query
    assert "shutting down servers" in query


def test_ground_claim_verified_true():
    engine = LiveWebGroundingEngine()
    claim = StreamerClaim(
        claim_id="c2",
        subject="Starforge Systems",
        predicate="builds custom gaming PCs",
        statement="Starforge Systems builds high end gaming PCs.",
    )
    res = engine.ground_claim(claim)
    assert res.verdict == FactCheckVerdict.VERIFIED_TRUE
    assert len(res.citations) > 0
    assert "starforgesystems.com" in res.citations[0].domain


def test_ground_claim_contradicted():
    engine = LiveWebGroundingEngine()
    claim = StreamerClaim(
        claim_id="c3",
        subject="Blizzard servers",
        predicate="shutting down permanently",
        statement="Blizzard servers are fake and shutting down permanently.",
    )
    res = engine.ground_claim(claim)
    assert res.verdict == FactCheckVerdict.CONTRADICTED
    assert "contradicted" in res.explanation.lower()


def test_ground_claim_with_onscreen_social_card():
    engine = LiveWebGroundingEngine()
    claim = StreamerClaim(
        claim_id="c4",
        subject="Asmongold",
        predicate="announces new raid stream",
        statement="Asmongold announced a new raid stream on Twitter.",
    )
    web_ctx = ScreenWebContext(
        timestamp_sec=25.0,
        browser_detected=True,
        detected_url="https://x.com/Zackrawrr/status/999",
        domain="x.com",
        social_post_cards=[
            SocialPostCard(
                platform="X_TWITTER",
                author_handle="@Zackrawrr",
                author_name="Zack",
                post_text="New raid stream coming tomorrow night. Be ready.",
                bounding_box=[0.1, 0.2, 0.5, 0.8],
            )
        ],
    )
    res = engine.ground_claim(claim, web_context=web_ctx)
    assert len(res.citations) >= 2
    assert any("x.com" in c.domain for c in res.citations)


# --- 2. Temporal Stance Shift Tracker Tests ---

def test_temporal_stance_tracker_shifts_and_reversals():
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "test_stance.json"
        tracker = TemporalStanceShiftTracker(storage_path=db_file)

        # Stream 1: Favorable towards Blizzard
        tracker.record_observation(
            entity_name="Blizzard",
            stance="POSITIVE",
            stream_id="stream_001",
            quote="Blizzard really cooked with this new expansion, it is incredible.",
            timestamp_sec=120.0,
        )

        # Stream 2: Critical / Negative stance
        tracker.record_observation(
            entity_name="Blizzard",
            stance="NEGATIVE",
            stream_id="stream_002",
            quote="Blizzard completely ruined the game with the latest microtransactions.",
            timestamp_sec=350.0,
        )

        timeline = tracker.get_timeline("Blizzard")
        assert len(timeline) == 2

        shifts = tracker.detect_shifts("Blizzard")
        assert len(shifts) == 1
        shift = shifts[0]
        assert shift.is_reversal is True
        assert shift.shift_delta == -2.0
        assert shift.previous_stance == "POSITIVE"
        assert shift.new_stance == "NEGATIVE"
        assert "ruined" in shift.evidence_quote_after

        # Volatility index
        volatility = tracker.compute_volatility("Blizzard")
        assert volatility == 1.0  # Max swing +1 to -1

        # Reload from disk
        tracker2 = TemporalStanceShiftTracker(storage_path=db_file)
        assert len(tracker2.get_timeline("Blizzard")) == 2


# --- 3. Cross-Stream Opinion Synthesizer Tests ---

def test_opinion_synthesis():
    tracker = TemporalStanceShiftTracker()

    tracker.record_observation("Ubisoft", "NEGATIVE", "s1", "Ubisoft games feel repetitive.")
    tracker.record_observation("Ubisoft", "NEGATIVE", "s2", "Another mediocre Ubisoft title.")
    tracker.record_observation("Ubisoft", "NEUTRAL", "s3", "This new indie game from them is okay.")

    synthesizer = CrossStreamOpinionSynthesizer(tracker=tracker)
    syn = synthesizer.synthesize("Ubisoft")

    assert syn.overall_consensus_stance == "NEGATIVE"
    assert syn.total_claims_count == 3
    assert syn.volatility_score > 0.0
    assert "repetitive" in syn.summary or "Ubisoft" in syn.summary


# --- 4. JSON-RPC 2.0 Integration Tests ---

def test_agent_rpc_knowledge_methods():
    dispatcher = AgentRpcDispatcher()

    # 1. groundClaim
    req_ground = {
        "jsonrpc": "2.0",
        "id": 10,
        "method": "streamfusion.groundClaim",
        "params": {
            "claim": {
                "claim_id": "c_rpc",
                "subject": "Starforge Systems",
                "predicate": "makes custom PCs",
                "statement": "Starforge Systems is an official PC brand.",
            }
        },
    }
    resp_ground = dispatcher.handle_request(req_ground)
    assert "result" in resp_ground
    assert resp_ground["result"]["verdict"] == "VERIFIED_TRUE"

    # Pre-populate stance tracker inside dispatcher
    dispatcher._stance_tracker = TemporalStanceShiftTracker()
    dispatcher._stance_tracker.record_observation("Elden Ring", "POSITIVE", "s1", "Masterpiece.")
    dispatcher._stance_tracker.record_observation("Elden Ring", "POSITIVE", "s2", "Still GOTY.")

    # 2. queryStanceShifts
    req_shifts = {
        "jsonrpc": "2.0",
        "id": 11,
        "method": "streamfusion.queryStanceShifts",
        "params": {"entity": "Elden Ring"},
    }
    resp_shifts = dispatcher.handle_request(req_shifts)
    assert "result" in resp_shifts
    assert isinstance(resp_shifts["result"], list)

    # 3. synthesizeEntityOpinions
    req_syn = {
        "jsonrpc": "2.0",
        "id": 12,
        "method": "streamfusion.synthesizeEntityOpinions",
        "params": {"entity": "Elden Ring"},
    }
    resp_syn = dispatcher.handle_request(req_syn)
    assert "result" in resp_syn
    assert resp_syn["result"]["overall_consensus_stance"] == "POSITIVE"


# --- 5. CLI Knowledge Commands Tests ---

def test_cli_knowledge_help(cli_runner):
    result = cli_runner.invoke(app, ["knowledge", "--help"])
    assert result.exit_code == 0
    assert "Web Grounding" in result.output
    assert "ground" in result.output
    assert "shifts" in result.output
    assert "synthesize" in result.output


def test_cli_knowledge_ground(cli_runner):
    with tempfile.TemporaryDirectory() as tmp_dir:
        claims_file = Path(tmp_dir) / "claims.json"
        claims_file.write_text(json.dumps([
            {
                "claim_id": "c_cli_1",
                "subject": "Starforge Systems",
                "predicate": "custom PCs",
                "statement": "Starforge Systems makes high end PCs.",
            }
        ]), encoding="utf-8")

        result = cli_runner.invoke(app, ["knowledge", "ground", str(claims_file)])
        assert result.exit_code == 0
        assert "Starforge Systems" in result.output
        assert "VERIFIED_TRUE" in result.output


def test_cli_knowledge_shifts_and_synthesize(cli_runner):
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_file = Path(tmp_dir) / "stance.json"
        tracker = TemporalStanceShiftTracker(storage_path=db_file)
        tracker.record_observation("Blizzard", "POSITIVE", "s1", "Great update.")
        tracker.record_observation("Blizzard", "NEGATIVE", "s2", "Worst patch ever.")

        # shifts command
        res_shifts = cli_runner.invoke(app, ["knowledge", "shifts", "--entity", "Blizzard", "--db", str(db_file)])
        assert res_shifts.exit_code == 0
        assert "Temporal Stance Shifts" in res_shifts.output
        assert "YES" in res_shifts.output

        # synthesize command
        res_syn = cli_runner.invoke(app, ["knowledge", "synthesize", "--entity", "Blizzard", "--db", str(db_file)])
        assert res_syn.exit_code == 0
        assert "Opinion Synthesis" in res_syn.output
