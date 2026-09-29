"""FullSpectrumPipeline Ingestion Bridge (Spec 25).

Provides a zero-copy handoff from harvested homelab storage into the
unified 9-Phase StreamFusion FullSpectrumPipeline with integrity checks.
"""

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Optional, Tuple

from stream_fusion.harvester.catalog import HarvestCatalog
from stream_fusion.harvester.engine import compute_file_sha256
from stream_fusion.models.schemas import (
    FullSpectrumConfig,
    FullSpectrumManifest,
    HarvestedVodRecord,
    HarvestStatus,
    StreamAnalysisResult,
)
from stream_fusion.orchestration.full_spectrum import FullSpectrumPipeline

logger = logging.getLogger(__name__)


def verify_harvest_checksums(vod_dir: Path) -> bool:
    """Verifies files in vod_dir against checksums.sha256."""
    checksum_file = vod_dir / "checksums.sha256"
    if not checksum_file.exists():
        return True  # No checksum file to verify against

    with open(checksum_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                expected_hash, fname = parts[0], parts[1].strip()
                fpath = vod_dir / fname
                if fpath.exists():
                    actual_hash = compute_file_sha256(fpath)
                    if actual_hash != expected_hash:
                        raise ValueError(
                            f"Integrity check failed for {fpath.name}: "
                            f"expected {expected_hash}, calculated {actual_hash}"
                        )
    return True


def ingest_to_pipeline(
    vod_id: str,
    catalog: HarvestCatalog,
    config: Optional[FullSpectrumConfig] = None,
    output_dir: Optional[Path] = None,
    duration_sec: Optional[float] = None,
    verify_checksums: bool = True,
    run_shorts: bool = True,
    ground_claims: bool = True,
    update_slang: bool = True,
    sponsor_audit: bool = True,
    dry_run_shorts: bool = False,
    short_candidate_count: int = 3,
) -> Tuple[StreamAnalysisResult, FullSpectrumManifest]:
    """Bridges a harvested homelab VOD directly into FullSpectrumPipeline."""
    vod = catalog.get_vod(vod_id)
    if not vod:
        raise ValueError(f"Harvested VOD '{vod_id}' not found in catalog.")

    if not vod.video_path or not Path(vod.video_path).exists():
        raise FileNotFoundError(f"Media video file not found for VOD '{vod_id}': {vod.video_path}")

    media_path = Path(vod.video_path)
    chat_path = Path(vod.chat_path) if vod.chat_path and Path(vod.chat_path).exists() else None
    vod_dir = media_path.parent

    # 1. Integrity check
    if verify_checksums:
        verify_harvest_checksums(vod_dir)

    # 2. Build or adapt configuration
    if config is None:
        config = FullSpectrumConfig(
            enable_short_production=run_shorts,
            enable_web_grounding=ground_claims,
            enable_adaptive_slang=update_slang,
            enable_sponsor_quantifier=sponsor_audit,
            dry_run_shorts=dry_run_shorts,
            short_candidate_count=short_candidate_count,
        )

    # Resolve output directory: default to analysis folder inside VOD directory
    target_out_dir = Path(output_dir) if output_dir else (vod_dir / "analysis")
    target_out_dir.mkdir(parents=True, exist_ok=True)

    # 3. Retrieve creator display name from target
    target = catalog.get_target(vod.streamer_id)
    creator_name = target.display_name if target else vod.streamer_id

    # 4. Execute FullSpectrumPipeline
    logger.info(f"Ingesting harvested VOD {vod_id} into FullSpectrumPipeline...")
    pipeline = FullSpectrumPipeline(config=config)
    analysis_res, manifest = pipeline.run(
        media_input=media_path,
        chat_input=chat_path,
        output_dir=target_out_dir,
        duration_sec=duration_sec,
        creator_name=creator_name,
    )

    # 5. Update catalog status to ANALYZED
    catalog.update_vod_status(
        vod_id=vod_id,
        status=HarvestStatus.ANALYZED,
        analyzed_at=datetime.now(timezone.utc),
    )

    return analysis_res, manifest
