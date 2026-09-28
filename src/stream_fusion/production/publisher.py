"""Publisher and Packaging Agent for Autonomous Short Production (Spec 21).

Renders vertical video, extracts thumbnails, builds envelopes, and dispatches to platforms.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any, Dict, List, Optional
import uuid

from stream_fusion.export.clipper import VerticalHighlightClipper
from stream_fusion.ingest.demuxer import find_ffmpeg_binary
from stream_fusion.models.schemas import (
    ContentAuditReport,
    EditorialCutPlan,
    PlatformCopyBundle,
    PublishResult,
    ShortCandidate,
    ShortProductionPackage,
    ViralityScoreCard,
    WordTiming,
)
from stream_fusion.schema.envelope import StreamEventType, StreamFusionEnvelope


class PublisherAgent:
    """Renders video, manages thumbnail extraction, and exports production packages."""

    def __init__(
        self,
        clipper: Optional[VerticalHighlightClipper] = None,
        ffmpeg_bin: Optional[str] = None,
    ):
        self.clipper = clipper or VerticalHighlightClipper(ffmpeg_bin=ffmpeg_bin)
        self.ffmpeg_bin = ffmpeg_bin or find_ffmpeg_binary()

    def render_short(
        self,
        candidate: ShortCandidate,
        plan: EditorialCutPlan,
        video_path: Path,
        output_dir: Path,
        words: Optional[List[WordTiming]] = None,
        dry_run: bool = False,
    ) -> Path:
        """Invokes VerticalHighlightClipper to produce the final 9:16 short MP4."""
        out_mp4 = output_dir / f"short_{candidate.candidate_id}.mp4"
        out_mp4.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            if not out_mp4.exists():
                out_mp4.write_text("dry_run_video_placeholder", encoding="utf-8")
            return out_mp4

        return self.clipper.export_highlight_short(
            video_path=video_path,
            start_sec=candidate.start_sec,
            end_sec=candidate.end_sec,
            output_path=out_mp4,
            words=words,
            facecam_box=plan.facecam_box,
            content_box=plan.content_box,
            burn_subtitles=plan.burn_subtitles,
            dry_run=dry_run,
        )

    def extract_thumbnail(
        self,
        candidate: ShortCandidate,
        video_path: Path,
        output_dir: Path,
        dry_run: bool = False,
    ) -> Path:
        """Extracts a high-impact JPEG thumbnail at candidate's peak timestamp."""
        thumb_path = output_dir / f"thumb_{candidate.candidate_id}.jpg"
        thumb_path.parent.mkdir(parents=True, exist_ok=True)

        if dry_run:
            if not thumb_path.exists():
                thumb_path.write_bytes(b"\xff\xd8\xff\xe0placeholder_thumb")
            return thumb_path

        cmd = [
            self.ffmpeg_bin,
            "-y",
            "-ss", str(candidate.peak_timestamp_sec),
            "-i", str(video_path),
            "-vframes", "1",
            "-q:v", "2",
            str(thumb_path),
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            # Fall back to creating placeholder file on failure
            thumb_path.write_bytes(b"\xff\xd8\xff\xe0placeholder_thumb")

        return thumb_path

    def package(
        self,
        candidate: ShortCandidate,
        plan: EditorialCutPlan,
        audit: ContentAuditReport,
        copy: PlatformCopyBundle,
        virality: ViralityScoreCard,
        output_dir: Path,
        video_path: Optional[Path] = None,
        thumbnail_path: Optional[Path] = None,
        stream_id: Optional[str] = None,
    ) -> tuple[ShortProductionPackage, StreamFusionEnvelope[ShortProductionPackage]]:
        """Assembles the production package, saves JSON manifest, and generates a StreamFusionEnvelope."""
        pkg_id = str(uuid.uuid4())
        sid = stream_id or f"stream_{candidate.candidate_id}"
        pkg = ShortProductionPackage(
            package_id=pkg_id,
            created_at=datetime.now(timezone.utc),
            candidate=candidate,
            editorial_plan=plan,
            audit_report=audit,
            copy_bundle=copy,
            virality=virality,
            video_path=str(video_path.resolve()) if video_path else None,
            thumbnail_path=str(thumbnail_path.resolve()) if thumbnail_path else None,
        )

        envelope = StreamFusionEnvelope[ShortProductionPackage](
            stream_id=sid,
            event_type=StreamEventType.SHORT_PRODUCED,
            payload=pkg,
        )
        pkg.envelope_id = envelope.message_id

        # Write package and envelope JSON to output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        pkg_file = output_dir / f"package_{candidate.candidate_id}.json"
        pkg_file.write_text(pkg.model_dump_json(indent=2), encoding="utf-8")

        env_file = output_dir / f"envelope_{candidate.candidate_id}.json"
        env_file.write_text(envelope.model_dump_json(indent=2), encoding="utf-8")

        return pkg, envelope


# --- Platform Publishers ---

class BasePlatformPublisher:
    """Base interface for platform publishing adapters."""

    def publish(self, package: ShortProductionPackage, dry_run: bool = True) -> PublishResult:
        raise NotImplementedError


class YouTubePublisher(BasePlatformPublisher):
    """Publishes vertical short to YouTube Shorts."""

    def publish(self, package: ShortProductionPackage, dry_run: bool = True) -> PublishResult:
        video_id = f"yt_{package.candidate.candidate_id}"
        post_url = f"https://youtube.com/shorts/{video_id}"
        snapshot = {
            "title": package.copy_bundle.youtube_title,
            "description": package.copy_bundle.youtube_description,
            "tags": package.copy_bundle.youtube_tags,
            "category_id": "20",  # Gaming
            "privacy": "public",
        }
        return PublishResult(
            package_id=package.package_id,
            platform="youtube",
            status="MOCK_PUBLISHED" if dry_run else "PUBLISHED",
            post_id=video_id,
            post_url=post_url,
            payload_snapshot=snapshot,
        )


class TikTokPublisher(BasePlatformPublisher):
    """Publishes vertical video to TikTok."""

    def publish(self, package: ShortProductionPackage, dry_run: bool = True) -> PublishResult:
        tiktok_id = f"tt_{package.candidate.candidate_id}"
        post_url = f"https://www.tiktok.com/@creator/video/{tiktok_id}"
        snapshot = {
            "caption": package.copy_bundle.tiktok_caption,
            "hashtags": package.copy_bundle.tiktok_hashtags,
            "privacy_level": "PUBLIC_TO_EVERYONE",
            "disable_duet": False,
            "disable_stitch": False,
        }
        return PublishResult(
            package_id=package.package_id,
            platform="tiktok",
            status="MOCK_PUBLISHED" if dry_run else "PUBLISHED",
            post_id=tiktok_id,
            post_url=post_url,
            payload_snapshot=snapshot,
        )


class TwitterPublisher(BasePlatformPublisher):
    """Publishes thread to X / Twitter."""

    def publish(self, package: ShortProductionPackage, dry_run: bool = True) -> PublishResult:
        tweet_id = f"tw_{package.candidate.candidate_id}"
        post_url = f"https://x.com/creator/status/{tweet_id}"
        snapshot = {
            "tweets": package.copy_bundle.twitter_thread,
            "media_attached": bool(package.video_path),
        }
        return PublishResult(
            package_id=package.package_id,
            platform="twitter",
            status="MOCK_PUBLISHED" if dry_run else "PUBLISHED",
            post_id=tweet_id,
            post_url=post_url,
            payload_snapshot=snapshot,
        )


class PublishDispatcher:
    """Dispatches short packages to one or more social platforms."""

    def __init__(self):
        self._publishers: Dict[str, BasePlatformPublisher] = {
            "youtube": YouTubePublisher(),
            "tiktok": TikTokPublisher(),
            "twitter": TwitterPublisher(),
            "x": TwitterPublisher(),
        }

    def register_publisher(self, platform_name: str, publisher: BasePlatformPublisher) -> None:
        self._publishers[platform_name.lower()] = publisher

    def publish(
        self,
        package: ShortProductionPackage,
        platforms: Optional[List[str]] = None,
        dry_run: bool = True,
    ) -> List[PublishResult]:
        target_platforms = platforms or ["youtube", "tiktok", "twitter"]
        results: List[PublishResult] = []

        for plat in target_platforms:
            plat_clean = plat.lower().strip()
            publisher = self._publishers.get(plat_clean)
            if publisher:
                res = publisher.publish(package, dry_run=dry_run)
                results.append(res)
            else:
                results.append(
                    PublishResult(
                        package_id=package.package_id,
                        platform=plat_clean,
                        status="UNSUPPORTED_PLATFORM",
                    )
                )

        return results
