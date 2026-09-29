"""StreamFusion Command Line Interface."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import typer
from rich.console import Console
from rich.table import Table

from stream_fusion.models.schemas import AudioSegment, VisualKeyframe, ChatMessage
from stream_fusion.chat.analyzer import ChatAnalyzer
from stream_fusion.fusion.matrix import FusionEngine
from stream_fusion.export.html_report import export_html_report
from stream_fusion.export.clipper import VerticalHighlightClipper
from stream_fusion.pipeline import StreamPipeline
from stream_fusion.config import StreamFusionConfig

app = typer.Typer(
    name="streamfusion",
    help="Multimodal Livestream VOD & Chat Grounding and Analysis CLI",
    no_args_is_help=True,
)
console = Console()


@app.command()
def process(
    video: Path = typer.Argument(..., help="Path to local VOD video file (mp4, mkv)"),
    chat: Optional[Path] = typer.Option(None, "--chat", "-c", help="Path to Twitch/YouTube chat replay JSON"),
    output_dir: Path = typer.Option(Path("./output"), "--out", "-o", help="Output directory"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Limit analysis to N seconds"),
    latency_offset: Optional[float] = typer.Option(None, "--latency-offset", "-l", help="Manual broadcast delay offset in seconds"),
    auto_latency: bool = typer.Option(True, "--auto-latency/--no-auto-latency", help="Automatically calibrate broadcast latency via cross-correlation"),
    chunk_duration: Optional[float] = typer.Option(None, "--chunk-duration", help="Chunk duration in seconds for processing long streams in chunks"),
    cache_dir: Optional[Path] = typer.Option(None, "--cache-dir", help="Directory for stateful resumption checkpoints"),
    isolate_workers: bool = typer.Option(False, "--isolate-workers", help="Run Whisper and Florence in isolated subprocesses for zero VRAM leakage"),
    bounded_buffer: bool = typer.Option(False, "--bounded-buffer", help="Process frames in sliding windows and immediately purge temporary image files"),
    window_size: float = typer.Option(30.0, "--window-size", help="Window duration in seconds for bounded keyframe buffering"),
):
    """Run full multimodal grounding on a video VOD and chat replay."""
    console.print(f"[bold purple]StreamFusion Pipeline[/bold purple]: Processing {video.name}")
    config = StreamFusionConfig()
    config.chat.auto_calibrate_latency = auto_latency
    config.execution.isolate_gpu_workers = isolate_workers
    config.execution.bounded_buffering = bounded_buffer
    config.execution.window_size_sec = window_size
    if latency_offset is not None:
        config.chat.latency_offset_sec = latency_offset
    pipeline = StreamPipeline(config=config)

    if chunk_duration is not None and chunk_duration > 0:
        results = pipeline.run_chunked(
            media_input=video,
            chunk_duration_sec=chunk_duration,
            chat_input=chat,
            output_dir=output_dir,
            total_duration_sec=duration,
            latency_offset=latency_offset,
            cache_dir=cache_dir,
        )
        console.print(f"[bold green][OK] Chunked Analysis Complete: {len(results)} chunks processed![/bold green]")
    else:
        result = pipeline.run(
            media_input=video,
            chat_input=chat,
            output_dir=output_dir,
            duration_sec=duration,
            latency_offset=latency_offset,
            auto_calibrate_latency=auto_latency,
            cache_dir=cache_dir,
        )
        console.print(f"[bold green][OK] Analysis Complete for {result.stream_id}![/bold green]")


@app.command()
def clip(
    video: Path = typer.Argument(..., help="Path to input video file"),
    start: float = typer.Option(0.0, "--start", "-s", help="Start time in seconds"),
    end: float = typer.Option(15.0, "--end", "-e", help="End time in seconds"),
    output: Path = typer.Option(Path("./output_short.mp4"), "--out", "-o", help="Output path for 9:16 vertical short"),
):
    """Clip a highlight segment into a 9:16 vertical video for YouTube Shorts / TikTok."""
    console.print(f"[bold purple]StreamFusion Clipper[/bold purple]: Extracting [{start:.1f}s - {end:.1f}s] to 9:16 vertical short")
    clipper = VerticalHighlightClipper()
    res = clipper.export_highlight_short(video, start_sec=start, end_sec=end, output_path=output)
    console.print(f"[bold green][OK] Vertical short exported to:[/bold green] {res.resolve()}")


@app.command()
def demo(
    output_html: Path = typer.Option(
        Path("./demo_report.html"),
        "--out",
        "-o",
        help="Path for generated HTML report",
    )
):
    """Run an end-to-end demonstration using an Asmongold reaction stream scenario."""
    console.print("[bold purple]StreamFusion Demonstration Mode[/bold purple]")
    console.print("Simulating 60-second reaction segment: [bold]Asmongold reacts to game trailer[/bold]...")

    duration = 60.0

    # 1. Simulated Diarized Audio: Streamer vs External Video
    audio_segments = [
        AudioSegment(
            segment_id=1,
            start_sec=2.0,
            end_sec=14.0,
            speaker_label="EXTERNAL_VIDEO",
            transcript="In our new update, crafting materials will now be sold in the premium shop.",
        ),
        AudioSegment(
            segment_id=2,
            start_sec=15.0,
            end_sec=28.0,
            speaker_label="STREAMER",
            transcript="Wait, hold on a second. Did they actually just say that? Bro, are you serious?",
        ),
        AudioSegment(
            segment_id=3,
            start_sec=30.0,
            end_sec=45.0,
            speaker_label="STREAMER",
            transcript="There is literally no way they thought people wouldn't notice. This is completely cooked.",
        ),
    ]

    # 2. Simulated Visual Keyframes: Screen & OCR
    visual_frames = [
        VisualKeyframe(
            frame_index=1,
            timestamp_sec=2.0,
            scene_type="REACT_VIDEO",
            screen_summary="Browser playing YouTube video 'Developer Update 2026'. Streamer facecam in bottom right.",
            ocr_text_blocks=["Developer Update 2026", "Premium Shop Bundles"],
        ),
        VisualKeyframe(
            frame_index=2,
            timestamp_sec=16.0,
            scene_type="REACT_VIDEO",
            screen_summary="Streamer pauses YouTube video, leans closer to microphone with wide eyes.",
            ocr_text_blocks=["Paused at 04:12"],
            streamer_facial_expression="disbelief",
        ),
        VisualKeyframe(
            frame_index=3,
            timestamp_sec=32.0,
            scene_type="FULLSCREEN_CAM",
            screen_summary="Streamer switches to fullscreen camera, shaking head in disbelief.",
            streamer_facial_expression="smirking",
        ),
    ]

    # 3. Simulated Chat Replay with realistic latency
    raw_chat = []
    for s in range(0, 18):
        raw_chat.append(
            ChatMessage(
                message_id=f"c_{s}",
                timestamp_offset=float(s),
                user_id=f"u_{s}",
                author_name=f"Viewer_{s}",
                content="pepeJAM nice music",
            )
        )
    react_emotes = ["OMEGALUL", "ICANT", "COOKED", "L", "Aware", "KEKW", "TRUE"]
    for s in range(19, 45):
        for sub in range(5):
            emote = react_emotes[(s + sub) % len(react_emotes)]
            raw_chat.append(
                ChatMessage(
                    message_id=f"spike_{s}_{sub}",
                    timestamp_offset=float(s) + (sub * 0.18),
                    user_id=f"chatter_{s}_{sub}",
                    author_name=f"Chatter_{s}_{sub}",
                    content=f"{emote} {emote} HE PAUSED {emote}",
                )
            )

    console.print(f"[green][OK][/green] Loaded {len(audio_segments)} audio segments")
    console.print(f"[green][OK][/green] Loaded {len(visual_frames)} visual keyframe contexts")
    console.print(f"[green][OK][/green] Ingested {len(raw_chat)} chat messages")

    # 4. Processing
    analyzer = ChatAnalyzer(latency_offset_sec=4.0)
    chat_buckets = analyzer.aggregate_into_buckets(
        raw_chat, stream_duration_sec=duration, bucket_size_sec=2.0
    )

    fusion = FusionEngine(bucket_size_sec=2.0)
    result = fusion.build_matrix(
        stream_id="zackrawrr_react_sample_01",
        duration_sec=duration,
        audio_segments=audio_segments,
        visual_keyframes=visual_frames,
        chat_buckets=chat_buckets,
    )

    # 5. Output Preview
    console.print(f"[bold green]Matrix Aligned successfully![/bold green] Generated {len(result.slices)} time slices.")

    table = Table(title="Multimodal Fusion Preview (Sample Slices)")
    table.add_column("Time (s)", justify="center", style="cyan")
    table.add_column("Streamer Speech", style="blue")
    table.add_column("Visual State", style="yellow")
    table.add_column("Chat Velocity", justify="right", style="magenta")
    table.add_column("Top Emote", style="green")

    for s in result.slices[6:16]:
        top_emote = list(s.dominant_emotes.keys())[0] if s.dominant_emotes else "-"
        table.add_row(
            f"{s.start_sec:.1f} - {s.end_sec:.1f}",
            s.streamer_transcript or "-",
            s.visual_description[:30] + "...",
            f"{s.chat_velocity_per_sec:.1f} msg/s",
            top_emote,
        )

    console.print(table)

    # 6. Export HTML
    export_html_report(result, output_html)
    console.print(f"[bold green][OK] Interactive HTML report exported to:[/bold green] {output_html.resolve()}")


@app.command(name="chat-nlp")
def chat_nlp(
    chat: Path = typer.Argument(..., help="Path to Twitch chat replay JSON"),
    min_authors: int = typer.Option(5, "--min-authors", "-m", help="Minimum distinct authors for meme burst detection"),
    window_sec: float = typer.Option(10.0, "--window-sec", "-w", help="Sliding window size in seconds"),
):
    """Analyze chat emotional intent, meme bursts, and chatter influence rankings."""
    from stream_fusion.chat.nlp import ChatNLPAnalyzer, MemeBurstTracker, ChatterInfluenceScorer

    analyzer = ChatAnalyzer()
    messages = analyzer.parse_twitch_downloader_json(chat)
    console.print(f"[bold purple]Chat NLP[/bold purple]: Parsed {len(messages)} messages from {chat.name}")

    nlp_analyzer = ChatNLPAnalyzer()
    classified = nlp_analyzer.analyze_message_stream(messages)
    intent_counts = {}
    for _, dist in classified:
        intent_counts[dist.primary_intent] = intent_counts.get(dist.primary_intent, 0) + 1

    table = Table(title="Community Emotional Intent Distribution")
    table.add_column("Intent", style="cyan")
    table.add_column("Message Count", justify="right", style="green")
    table.add_column("Percentage", justify="right", style="yellow")
    for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
        pct = (count / len(messages)) * 100.0 if messages else 0.0
        table.add_row(intent, str(count), f"{pct:.1f}%")
    console.print(table)

    # Meme bursts
    tracker = MemeBurstTracker(window_sec=window_sec, min_distinct_authors=min_authors)
    bursts = tracker.detect_meme_bursts(messages)
    console.print(f"[bold green]Meme Bursts Detected:[/bold green] {len(bursts)}")
    if bursts:
        b_table = Table(title="Emergent Meme Bursts")
        b_table.add_column("Time (s)", style="cyan")
        b_table.add_column("Representative Text", style="white")
        b_table.add_column("Originator", style="magenta")
        b_table.add_column("Velocity", justify="right", style="yellow")
        for b in bursts[:5]:
            b_table.add_row(
                f"{b.burst_start_sec:.1f}s - {b.burst_end_sec:.1f}s",
                b.representative_text[:40],
                b.origin_author_name,
                f"{b.propagation_velocity:.1f} msg/s",
            )
        console.print(b_table)

    # Chatter influence
    scorer = ChatterInfluenceScorer()
    profiles = scorer.score_chatters(messages, bursts)
    if profiles:
        c_table = Table(title="Top Opinion Leader Chatters")
        c_table.add_column("Author", style="cyan")
        c_table.add_column("Total Msgs", justify="right", style="white")
        c_table.add_column("Meme Origins", justify="right", style="green")
        c_table.add_column("Influence Score", justify="right", style="bold yellow")
        for p in profiles[:5]:
            c_table.add_row(
                p.author_name,
                str(p.total_messages),
                str(p.first_meme_origin_count),
                f"{p.influence_score:.2f}",
            )
        console.print(c_table)


@app.command()
def sponsor(
    brand_name: str = typer.Argument(..., help="Brand name, e.g. 'Starforge Systems'"),
    chat: Path = typer.Option(..., "--chat", "-c", help="Path to chat replay JSON"),
    start_sec: float = typer.Option(0.0, "--start", "-s", help="Sponsor segment start second"),
    end_sec: float = typer.Option(60.0, "--end", "-e", help="Sponsor segment end second"),
    alias: Optional[str] = typer.Option(None, "--alias", "-a", help="Comma-separated brand aliases"),
    promo: Optional[str] = typer.Option(None, "--promo", "-p", help="Promo code (e.g. ASMON)"),
):
    """Evaluate sponsor engagement window and compute Brand Attention Score."""
    from stream_fusion.models.schemas import BrandProfile, SponsorSegment
    from stream_fusion.analytics.sponsor_quantifier import SponsorReportGenerator

    analyzer = ChatAnalyzer()
    messages = analyzer.parse_twitch_downloader_json(chat)

    aliases = [a.strip() for a in alias.split(",")] if alias else []
    promos = [p.strip() for p in promo.split(",")] if promo else []

    brand = BrandProfile(
        brand_id=brand_name.lower().replace(" ", "_"),
        brand_name=brand_name,
        aliases=aliases,
        promo_codes=promos,
    )
    segment = SponsorSegment(
        segment_id=1,
        brand_id=brand.brand_id,
        start_sec=start_sec,
        end_sec=end_sec,
    )
    report_gen = SponsorReportGenerator()
    report = report_gen.generate_report(brand, segment, messages)

    console.print(f"[bold purple]Sponsor Impact Analysis[/bold purple]: {brand.brand_name}")
    console.print(f"Segment: [{segment.start_sec:.1f}s - {segment.end_sec:.1f}s]")
    console.print(f"Chat Mentions: [bold cyan]{report.chat_mention_count}[/bold cyan] ({report.mention_velocity:.2f}/s)")
    console.print(f"Sentiment Delta: [bold green]{report.sentiment_delta:+.2f}[/bold green] (window: {report.sentiment_during_sponsor:.2f} vs stream: {report.stream_baseline_sentiment:.2f})")
    console.print(f"Backlash Index: [bold red]{report.backlash_index:.1%}[/bold red]")
    console.print(f"Brand Attention Score: [bold yellow]{report.brand_attention_score:.1f} / 100.0[/bold yellow]")


@app.command(name="voice-enroll")
def voice_enroll(
    creator_id: str = typer.Argument(..., help="Unique creator identifier, e.g. 'theburntpeanut'"),
    display_name: str = typer.Argument(..., help="Display name, e.g. 'TheBurntPeanut'"),
    channel: Optional[str] = typer.Option(None, "--channel", help="Twitch/YouTube channel URL"),
    registry: Path = typer.Option(Path("./voiceprints.json"), "--registry", "-r", help="Path to voiceprints registry"),
):
    """Enroll a creator into the global voiceprint library."""
    from stream_fusion.audio.voiceprint import VoiceprintLibrary, SpeakerEmbeddingExtractor
    import numpy as np

    lib = VoiceprintLibrary(storage_path=registry)
    extractor = SpeakerEmbeddingExtractor(embedding_dim=192)
    synthetic_emb = np.random.uniform(-0.1, 0.1, 192).astype(np.float32)
    synthetic_emb[0] = 1.0
    synthetic_emb = (synthetic_emb / np.linalg.norm(synthetic_emb)).tolist()

    prof = lib.enroll_creator(
        creator_id=creator_id,
        display_name=display_name,
        embedding=synthetic_emb,
        primary_channel=channel,
    )
    console.print(f"[bold green][OK] Enrolled Creator Voiceprint:[/bold green] {prof.display_name} ({prof.creator_id})")
    console.print(f"Centroid registered in: {registry.resolve()}")


@app.command(name="flag-griefers")
def flag_griefers(
    chat: Path = typer.Argument(..., help="Path to chat replay JSON"),
    threshold: float = typer.Option(0.65, "--threshold", "-t", help="Griefer probability cutoff threshold"),
):
    """Scan chat replay to flag bad-faith griefers, toxic contrarians, and coordinated brigades."""
    from stream_fusion.chat.profiler import ChatterProfileStore, BrigadeDetector

    analyzer = ChatAnalyzer()
    messages = analyzer.parse_twitch_downloader_json(chat)
    store = ChatterProfileStore(db_path=":memory:")

    for m in messages:
        store.ingest_message(m)

    cur = store.conn.cursor()
    cur.execute("SELECT user_id, username, total_messages, griefer_score, flagged_status FROM chatters WHERE griefer_score >= ? ORDER BY griefer_score DESC", (threshold,))
    flagged = cur.fetchall()

    console.print(f"[bold purple]Chatter Safety Scan[/bold purple]: Analyzed {len(messages)} messages from {chat.name}")
    table = Table(title="Flagged Bad-Faith Accounts / Griefers")
    table.add_column("Username", style="cyan")
    table.add_column("Total Msgs", justify="right", style="white")
    table.add_column("Griefer Score", justify="right", style="bold red")
    table.add_column("Status", style="yellow")

    for row in flagged[:10]:
        table.add_row(row[1], str(row[2]), f"{row[3]:.2f}", row[4])
    console.print(table)

    detector = BrigadeDetector()
    brigades = detector.detect_brigades(messages)
    if brigades:
        console.print(f"[bold red]WARNING:[/bold red] {len(brigades)} coordinated brigade cluster(s) detected!")


@app.command(name="query-claims")
def query_claims(
    query: str = typer.Argument(..., help="Query string, e.g. 'Godzilla special effects'"),
    creator: Optional[str] = typer.Option(None, "--creator", "-c", help="Filter by creator ID"),
):
    """Query the Streamer Knowledge Graph for semantic opinions, stances, and quotes."""
    from stream_fusion.knowledge.claims import StreamerKnowledgeStore

    store = StreamerKnowledgeStore(db_path=":memory:")
    # Pre-populate sample knowledge demo
    from stream_fusion.models.schemas import AudioSegment
    sample_segments = [
        AudioSegment(
            segment_id=1, start_sec=14.0, end_sec=20.0, speaker_label="STREAMER",
            transcript="The special effects on that Godzilla movie look completely cooked.",
        ),
        AudioSegment(
            segment_id=2, start_sec=120.0, end_sec=126.0, speaker_label="STREAMER",
            transcript="I actually really liked the Godzilla movie overall, great ending.",
        ),
    ]
    store.ingest_audio_segments(sample_segments, creator_id=creator or "asmongold", entity_hint="Godzilla")

    res = store.query_streamer_knowledge(query, creator_id=creator)
    console.print(f"[bold purple]Knowledge Query Results[/bold purple]: '{query}'")

    if res["synthesized_stances"]:
        s_table = Table(title="Synthesized Entity Stances")
        s_table.add_column("Entity", style="cyan")
        s_table.add_column("Overall Stance", style="green")
        s_table.add_column("Polarity", justify="right", style="yellow")
        s_table.add_column("Sub-Attributes", style="white")
        for st in res["synthesized_stances"]:
            attrs = ", ".join(f"{k}: {v:+.2f}" for k, v in st.sub_attributes.items())
            s_table.add_row(st.subject_entity, st.overall_stance, f"{st.aggregate_polarity:+.2f}", attrs)
        console.print(s_table)



# ---------------------------------------------------------------------------
# Continuous Benchmarking, Telemetry & Auditing Subcommands (Spec 15)
# ---------------------------------------------------------------------------
audit_app = typer.Typer(
    name="audit",
    help="Continuous Benchmarking, Telemetry, and Auditing Commands",
    no_args_is_help=True,
)
app.add_typer(audit_app, name="audit")


@audit_app.command(name="benchmark")
def audit_benchmark(
    video: Path = typer.Option(Path("./asmon_sample_60s.mp4"), "--video", "-v", help="Path to sample VOD video file"),
    chat: Optional[Path] = typer.Option(Path("./sample_asmon_chat.json"), "--chat", "-c", help="Path to sample chat replay JSON"),
    output_dir: Path = typer.Option(Path("./benchmark_output"), "--out", "-o", help="Output directory for benchmark artifacts"),
    duration: Optional[float] = typer.Option(60.0, "--duration", "-d", help="Max duration in seconds"),
    set_baseline: bool = typer.Option(False, "--set-baseline", help="Set this run as the active baseline"),
    db_path: Path = typer.Option(Path("./audit.db"), "--db", help="Path to SQLite audit database"),
):
    """Run benchmark suite on standard sample, recording telemetry & regression checks."""
    if not video.exists():
        console.print(f"[bold red]Error:[/bold red] Benchmark video file not found: {video}")
        raise typer.Exit(code=1)

    console.print(f"[bold purple]StreamFusion Continuous Benchmark[/bold purple]: {video.name}")
    from stream_fusion.monitoring.telemetry import TelemetryCollector
    from stream_fusion.monitoring.audit import (
        AuditStore,
        AuditRunRecord,
        RegressionComparator,
        StageTelemetry,
        get_git_info,
    )
    from datetime import datetime, timezone

    telemetry = TelemetryCollector()
    config = StreamFusionConfig()
    pipeline = StreamPipeline(config=config)

    effective_chat = chat if (chat and chat.exists()) else None
    result = pipeline.run(
        media_input=video,
        chat_input=effective_chat,
        output_dir=output_dir,
        duration_sec=duration,
        telemetry=telemetry,
    )

    git_commit, git_dirty = get_git_info()
    effective_media_dur = duration or result.metadata.get("duration_sec", 60.0)
    total_dur = telemetry.get_total_duration()
    overall_rtf = round(total_dur / max(effective_media_dur, 0.001), 3)

    stages = {
        name: StageTelemetry(**data) for name, data in telemetry.stages.items()
    }

    record = AuditRunRecord(
        run_id=telemetry.run_id,
        git_commit=git_commit,
        git_dirty=git_dirty,
        timestamp=datetime.now(timezone.utc).isoformat(),
        stream_id=video.stem,
        media_duration_sec=effective_media_dur,
        total_pipeline_duration_sec=total_dur,
        overall_real_time_factor=overall_rtf,
        stages=stages,
        quality_metrics=telemetry.quality_metrics,
        status="SUCCESS",
    )

    store = AuditStore(db_path=db_path)
    baseline = store.get_baseline()

    if baseline and not set_baseline:
        comparator = RegressionComparator()
        alerts, recs = comparator.compare_runs(record, baseline)
        record.regressions_detected = alerts
        record.recommendations = recs

    store.record_run(record, set_as_baseline=set_baseline)

    # Render summary table
    table = Table(title=f"Benchmark Telemetry Report: {record.run_id}")
    table.add_column("Stage / Metric", style="cyan")
    table.add_column("Duration (s)", justify="right", style="white")
    table.add_column("RAM Peak (MB)", justify="right", style="magenta")
    table.add_column("Status", style="green")

    for s_name, s_data in record.stages.items():
        table.add_row(
            s_name,
            f"{s_data.duration_sec:.2f}",
            f"{s_data.ram_mb_peak:.1f}",
            s_data.status,
        )

    table.add_section()
    table.add_row("Total Pipeline", f"{record.total_pipeline_duration_sec:.2f}", "-", "SUCCESS")
    table.add_row(
        "Real-Time Factor (RTF)",
        f"{record.overall_real_time_factor:.3f}x",
        "-",
        "[bold green]FAST[/bold green]" if record.overall_real_time_factor < 1.0 else "[bold yellow]SLOW[/bold yellow]",
    )
    console.print(table)

    if set_baseline:
        console.print(f"[bold green]Registered {record.run_id} as the new ACTIVE BASELINE.[/bold green]")
    elif baseline:
        if record.regressions_detected:
            console.print(f"[bold red]WARNING: {len(record.regressions_detected)} REGRESSION(S) DETECTED vs baseline {baseline.run_id}:[/bold red]")
            for alert in record.regressions_detected:
                console.print(f"  - [{alert.severity}] {alert.description}")
            if record.recommendations:
                console.print("[bold yellow]Recommendations:[/bold yellow]")
                for rec in record.recommendations:
                    console.print(f"  * {rec}")
        else:
            console.print(f"[bold green]No regressions detected against baseline {baseline.run_id}![/bold green]")


@audit_app.command(name="history")
def audit_history(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of recent runs to display"),
    db_path: Path = typer.Option(Path("./audit.db"), "--db", help="Path to SQLite audit database"),
):
    """List execution history and benchmarks from the audit store."""
    if not db_path.exists():
        console.print(f"[yellow]No audit database found at {db_path}[/yellow]")
        return

    from stream_fusion.monitoring.audit import AuditStore
    store = AuditStore(db_path=db_path)
    runs = store.get_recent_runs(limit=limit)
    baseline = store.get_baseline()
    baseline_id = baseline.run_id if baseline else None

    if not runs:
        console.print("[yellow]No audit runs recorded yet.[/yellow]")
        return

    table = Table(title=f"Audit History (Last {len(runs)} Runs)")
    table.add_column("Run ID", style="cyan")
    table.add_column("Git Commit", style="white")
    table.add_column("Timestamp", style="dim")
    table.add_column("Duration", justify="right", style="white")
    table.add_column("RTF", justify="right", style="magenta")
    table.add_column("Regressions", justify="right")
    table.add_column("Status", style="green")

    for r in runs:
        is_base = " [bold green](BASELINE)[/bold green]" if r.run_id == baseline_id else ""
        n_reg = len(r.regressions_detected)
        reg_style = "[red]" if n_reg > 0 else "[green]"
        table.add_row(
            f"{r.run_id}{is_base}",
            f"{r.git_commit}{'*' if r.git_dirty else ''}",
            r.timestamp[:19].replace("T", " "),
            f"{r.total_pipeline_duration_sec:.1f}s",
            f"{r.overall_real_time_factor:.3f}x",
            f"{reg_style}{n_reg}[/]",
            r.status,
        )

    console.print(table)


@audit_app.command(name="compare")
def audit_compare(
    run_id_a: str = typer.Argument(..., help="Baseline run ID (or 'baseline' to use the active baseline)"),
    run_id_b: Optional[str] = typer.Argument(None, help="Comparison run ID (defaults to latest recorded run)"),
    db_path: Path = typer.Option(Path("./audit.db"), "--db", help="Path to SQLite audit database"),
):
    """Compare performance and quality metrics between two audit runs."""
    if not db_path.exists():
        console.print(f"[yellow]No audit database found at {db_path}[/yellow]")
        return

    from stream_fusion.monitoring.audit import AuditStore, RegressionComparator
    store = AuditStore(db_path=db_path)

    if run_id_a.lower() == "baseline":
        rec_a = store.get_baseline()
        if not rec_a:
            console.print("[bold red]Error:[/bold red] No active baseline found in audit database.")
            raise typer.Exit(code=1)
    else:
        rec_a = store.get_run(run_id_a)
        if not rec_a:
            console.print(f"[bold red]Error:[/bold red] Run '{run_id_a}' not found.")
            raise typer.Exit(code=1)

    if run_id_b is None:
        recent = store.get_recent_runs(limit=1)
        if not recent:
            console.print("[bold red]Error:[/bold red] No runs available for comparison.")
            raise typer.Exit(code=1)
        rec_b = recent[0]
    else:
        rec_b = store.get_run(run_id_b)
        if not rec_b:
            console.print(f"[bold red]Error:[/bold red] Run '{run_id_b}' not found.")
            raise typer.Exit(code=1)

    comparator = RegressionComparator()
    alerts, recs = comparator.compare_runs(current=rec_b, baseline=rec_a)

    console.print(f"[bold purple]Audit Run Comparison[/bold purple]: [cyan]{rec_a.run_id}[/cyan] (Baseline) vs [cyan]{rec_b.run_id}[/cyan] (Current)")

    table = Table(title="Performance & Stage Breakdown Comparison")
    table.add_column("Metric / Stage", style="cyan")
    table.add_column(f"Baseline ({rec_a.run_id[:12]})", justify="right", style="white")
    table.add_column(f"Current ({rec_b.run_id[:12]})", justify="right", style="white")
    table.add_column("Diff", justify="right")

    # Total duration
    diff_tot = (rec_b.total_pipeline_duration_sec - rec_a.total_pipeline_duration_sec) / max(rec_a.total_pipeline_duration_sec, 0.001)
    d_style = "[red]+" if diff_tot > 0.05 else ("[green]" if diff_tot < -0.05 else "[dim]")
    table.add_row(
        "Total Pipeline Duration",
        f"{rec_a.total_pipeline_duration_sec:.2f}s",
        f"{rec_b.total_pipeline_duration_sec:.2f}s",
        f"{d_style}{diff_tot:+.1%}[/]",
    )

    # RTF
    diff_rtf = (rec_b.overall_real_time_factor - rec_a.overall_real_time_factor) / max(rec_a.overall_real_time_factor, 0.001)
    table.add_row(
        "Real-Time Factor",
        f"{rec_a.overall_real_time_factor:.3f}x",
        f"{rec_b.overall_real_time_factor:.3f}x",
        f"{diff_rtf:+.1%}",
    )

    table.add_section()
    all_stages = sorted(set(list(rec_a.stages.keys()) + list(rec_b.stages.keys())))
    for st in all_stages:
        dur_a = rec_a.stages[st].duration_sec if st in rec_a.stages else 0.0
        dur_b = rec_b.stages[st].duration_sec if st in rec_b.stages else 0.0
        if dur_a > 0:
            diff_st = (dur_b - dur_a) / dur_a
            diff_str = f"{diff_st:+.1%}"
            st_color = "[red]" if diff_st > 0.25 else ("[green]" if diff_st < -0.10 else "[dim]")
        else:
            diff_str = "NEW"
            st_color = "[cyan]"
        table.add_row(
            f"Stage: {st}",
            f"{dur_a:.2f}s" if dur_a else "-",
            f"{dur_b:.2f}s" if dur_b else "-",
            f"{st_color}{diff_str}[/]",
        )

    console.print(table)

    if alerts:
        console.print(f"[bold red]WARNING: {len(alerts)} Regression(s) Detected![/bold red]")
        for a in alerts:
            console.print(f"  - [{a.severity}] {a.description}")
        if recs:
            console.print("[bold yellow]Recommendations:[/bold yellow]")
            for r in recs:
                console.print(f"  * {r}")
    else:
        console.print("[bold green]Zero regressions detected. Current run matches or exceeds baseline performance![/bold green]")


# ---------------------------------------------------------------------------
# Unified JSON Schema Registry Subcommands (Spec 16)
# ---------------------------------------------------------------------------
schema_app = typer.Typer(
    name="schema",
    help="Unified JSON-Schema Registry, Validation, and Export Commands",
    no_args_is_help=True,
)
app.add_typer(schema_app, name="schema")


@schema_app.command(name="list")
def schema_list():
    """List all registered StreamFusion data schemas and envelope types."""
    from stream_fusion.schema.registry import default_schema_registry
    schemas = default_schema_registry.list_schemas()

    table = Table(title=f"StreamFusion Schema Registry ({len(schemas)} Schemas)")
    table.add_column("Schema Name", style="bold cyan")
    table.add_column("Module Source", style="dim")
    table.add_column("Description", style="white")

    for s in schemas:
        table.add_row(s["schema_name"], s["module"], s["description"])

    console.print(table)


@schema_app.command(name="export")
def schema_export(
    out_dir: Path = typer.Option(Path("./docs/schemas"), "--out-dir", "-o", help="Directory to save JSON/YAML schemas"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Specific model name to export (default: all)"),
    fmt: str = typer.Option("json", "--format", "-f", help="Output format: 'json' or 'yaml'"),
):
    """Export Draft-07 / 2020-12 JSON-Schema definitions for data models."""
    import json
    import yaml
    from stream_fusion.schema.registry import default_schema_registry

    out_dir.mkdir(parents=True, exist_ok=True)
    if model:
        cls = default_schema_registry.get_model(model)
        if not cls:
            console.print(f"[bold red]Error: Schema '{model}' not found in registry.[/bold red]")
            raise typer.Exit(code=1)
        schema_dict = default_schema_registry.export_json_schema(model)
        ext = "yaml" if fmt.lower() in ("yaml", "yml") else "json"
        out_file = out_dir / f"{model}.schema.{ext}"
        with open(out_file, "w", encoding="utf-8") as f:
            if ext == "yaml":
                yaml.dump(schema_dict, f, sort_keys=False)
            else:
                json.dump(schema_dict, f, indent=2)
        console.print(f"[bold green]Exported schema for {model} to {out_file}[/bold green]")
    else:
        exported = default_schema_registry.export_all_schemas(output_dir=out_dir, fmt=fmt)
        console.print(f"[bold green]Successfully exported {len(exported)} schemas to {out_dir} (format: {fmt})[/bold green]")


@schema_app.command(name="validate")
def schema_validate(
    schema_name: str = typer.Argument(..., help="Registered schema name (e.g. FusionSlice, StreamerClaim)"),
    file: Path = typer.Argument(..., help="Path to JSON file to validate"),
):
    """Validate a JSON file against a registered StreamFusion schema."""
    import json
    from stream_fusion.schema.registry import default_schema_registry

    if not file.exists():
        console.print(f"[bold red]Error: File {file} does not exist.[/bold red]")
        raise typer.Exit(code=1)

    try:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        console.print(f"[bold red]Error decoding JSON: {e}[/bold red]")
        raise typer.Exit(code=1)

    is_valid, err, _ = default_schema_registry.validate_data(schema_name, data)
    if is_valid:
        console.print(f"[bold green][OK] File {file.name} conforms strictly to schema '{schema_name}'[/bold green]")
    else:
        console.print(f"[bold red][FAIL] Validation failed for schema '{schema_name}':\n{err}[/bold red]")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Agent Communication & JSON-RPC Protocol Subcommands (Spec 16)
# ---------------------------------------------------------------------------
agent_app = typer.Typer(
    name="agent",
    help="Agent Communication, JSON-RPC 2.0 Dispatcher, and Stream Query Commands",
    no_args_is_help=True,
)
app.add_typer(agent_app, name="agent")


@agent_app.command(name="query")
def agent_query(
    db: Path = typer.Option(Path("./stream_events.db"), "--db", help="Path to SQLite events store"),
    stream_id: Optional[str] = typer.Option(None, "--stream-id", "-s", help="Filter by stream ID"),
    event_type: Optional[str] = typer.Option(None, "--event-type", "-e", help="Filter by event type"),
    filter_json: Optional[str] = typer.Option(None, "--filter", "-f", help="JSON string of payload attribute filters"),
    limit: int = typer.Option(20, "--limit", "-n", help="Max events to return"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON instead of table"),
):
    """Query stream events using SQLite JSON1 attribute filtering."""
    import json
    from stream_fusion.schema.adapters import SqliteJsonStore
    from stream_fusion.schema.envelope import StreamEventType

    if not db.exists():
        console.print(f"[yellow]Database {db} not found.[/yellow]")
        return

    store = SqliteJsonStore(db_path=db)
    parsed_filters = json.loads(filter_json) if filter_json else None
    ev_type = StreamEventType(event_type) if event_type else None

    events = store.query_events(
        stream_id=stream_id,
        event_type=ev_type,
        json_filters=parsed_filters,
        limit=limit,
    )

    if json_out:
        out_payload = [e.model_dump() for e in events]
        console.print(json.dumps(out_payload, indent=2))
        return

    table = Table(title=f"Stream Events Query Results ({len(events)} returned)")
    table.add_column("Timestamp", style="dim")
    table.add_column("Stream ID", style="cyan")
    table.add_column("Event Type", style="bold green")
    table.add_column("Producer", style="dim")
    table.add_column("Payload Preview", style="white")

    for e in events:
        p_preview = json.dumps(e.payload)[:60] + "..." if e.payload else "{}"
        table.add_row(
            e.timestamp[:19].replace("T", " "),
            e.stream_id,
            e.event_type.value,
            e.producer or "-",
            p_preview,
        )

    console.print(table)


@agent_app.command(name="summary")
def agent_summary(
    stream_id: str = typer.Argument(..., help="Stream ID to summarize"),
    db: Path = typer.Option(Path("./stream_events.db"), "--db", help="Path to SQLite events store"),
    json_out: bool = typer.Option(False, "--json", help="Output raw JSON instead of table"),
):
    """Retrieve an aggregated high-level event summary for a stream."""
    import json
    from stream_fusion.schema.agent_rpc import AgentRpcDispatcher
    from stream_fusion.schema.adapters import SqliteJsonStore

    if not db.exists():
        console.print(f"[yellow]Database {db} not found.[/yellow]")
        return

    store = SqliteJsonStore(db_path=db)
    dispatcher = AgentRpcDispatcher(store=store)
    req = {
        "jsonrpc": "2.0",
        "method": "streamfusion.getStreamSummary",
        "params": {"stream_id": stream_id},
        "id": 1,
    }
    resp = dispatcher.handle_request(req)

    if json_out or "error" in resp:
        console.print(json.dumps(resp, indent=2))
        return

    data = resp.get("result", {})
    table = Table(title=f"Stream Summary: {stream_id}")
    table.add_column("Metric / Category", style="cyan")
    table.add_column("Count", justify="right", style="bold green")

    table.add_row("Total Envelopes", str(data.get("total_events", 0)))
    counts = data.get("counts", {})
    for k, v in counts.items():
        table.add_row(f"Events: {k}", str(v))

    console.print(table)


@agent_app.command(name="rpc")
def agent_rpc(
    db: Path = typer.Option(Path("./stream_events.db"), "--db", help="Path to SQLite events store"),
):
    """Run JSON-RPC 2.0 loop over stdin/stdout for subagent communication."""
    import sys
    from stream_fusion.schema.agent_rpc import AgentRpcDispatcher
    from stream_fusion.schema.adapters import SqliteJsonStore

    store = SqliteJsonStore(db_path=db)
    dispatcher = AgentRpcDispatcher(store=store)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        resp_line = dispatcher.handle_line(line)
        sys.stdout.write(resp_line + "\n")
        sys.stdout.flush()


# ---------------------------------------------------------------------------
# Adaptive Slang & Meme Engine Subcommands (Spec 17)
# ---------------------------------------------------------------------------
slang_app = typer.Typer(
    name="slang",
    help="Self-Expanding Adaptive Slang & Meme Engine Commands",
    no_args_is_help=True,
)
app.add_typer(slang_app, name="slang")


@slang_app.command(name="scan")
def slang_scan(
    chat: Path = typer.Option(..., "--chat", "-c", help="Path to Twitch/YouTube chat replay JSON"),
    lexicon: Path = typer.Option(Path("./adaptive_lexicon.json"), "--lexicon", "-l", help="Path to adaptive lexicon store"),
    window: float = typer.Option(10.0, "--window", "-w", help="Rolling burst window in seconds"),
    z_threshold: float = typer.Option(3.0, "--z-score", "-z", help="Minimum standard deviations for burst detection"),
):
    """Scan chat replay for emerging slang bursts and auto-tag emotional intent."""
    from stream_fusion.chat.analyzer import ChatAnalyzer
    from stream_fusion.nlp.adaptive_slang import (
        AdaptiveLexiconStore,
        RollingBurstDetector,
        ContextualAutoTagger,
        AdaptiveSlangEngine,
    )

    if not chat.exists():
        console.print(f"[bold red]Chat replay file {chat} not found.[/bold red]")
        raise typer.Exit(code=1)

    analyzer = ChatAnalyzer()
    messages = analyzer.parse_twitch_downloader_json(chat)
    console.print(f"[bold purple]Scanning {len(messages)} chat messages for slang bursts (Z >= {z_threshold})...[/bold purple]")

    store = AdaptiveLexiconStore(db_path=lexicon)
    detector = RollingBurstDetector(window_sec=window, z_threshold=z_threshold)
    tagger = ContextualAutoTagger()
    engine = AdaptiveSlangEngine(lexicon_store=store, burst_detector=detector, auto_tagger=tagger)

    candidates, clusters = engine.process_chat_stream(messages, persist=True)

    if not candidates:
        console.print("[yellow]No new emerging slang terms exceeded the burst threshold in this chat stream.[/yellow]")
        return

    table = Table(title=f"Discovered Emerging Slang Candidates ({len(candidates)} Detected)")
    table.add_column("Term", style="bold cyan")
    table.add_column("Z-Score", justify="right", style="magenta")
    table.add_column("Velocity (msg/s)", justify="right", style="white")
    table.add_column("Occurrences", justify="right", style="white")
    table.add_column("Inferred Intent", style="bold green")
    table.add_column("Valence", justify="right", style="yellow")
    table.add_column("Confidence", justify="right", style="cyan")

    for c in candidates:
        table.add_row(
            c.term,
            f"{c.z_score:.1f}z",
            f"{c.burst_velocity:.1f}",
            str(c.total_occurrences),
            c.inferred_intent,
            f"{c.inferred_valence:+.2f}",
            f"{c.confidence:.0%}",
        )

    console.print(table)
    console.print(f"[bold green]Updated adaptive lexicon saved to {lexicon} ({len(store.entries)} total cataloged terms)[/bold green]")


@slang_app.command(name="list")
def slang_list(
    lexicon: Path = typer.Option(Path("./adaptive_lexicon.json"), "--lexicon", "-l", help="Path to adaptive lexicon store"),
    status: Optional[str] = typer.Option(None, "--status", "-s", help="Filter by status (ACTIVE, PROMOTED, DECAYED)"),
):
    """List cataloged slang terms, inferred emotions, and temporal decay status."""
    from stream_fusion.nlp.adaptive_slang import AdaptiveLexiconStore

    if not lexicon.exists():
        console.print(f"[yellow]Adaptive lexicon not found at {lexicon}[/yellow]")
        return

    store = AdaptiveLexiconStore(db_path=lexicon)
    entries = store.entries.values()
    if status:
        entries = [e for e in entries if e.status.upper() == status.upper()]

    if not entries:
        console.print("[yellow]No slang entries found matching criteria.[/yellow]")
        return

    table = Table(title=f"Adaptive Slang Lexicon ({len(entries)} Terms)")
    table.add_column("Term", style="bold cyan")
    table.add_column("Intent", style="bold green")
    table.add_column("Valence", justify="right", style="yellow")
    table.add_column("Occurrences", justify="right", style="white")
    table.add_column("Peak Z", justify="right", style="magenta")
    table.add_column("Decayed Conf.", justify="right", style="cyan")
    table.add_column("Status", style="white")

    for e in sorted(entries, key=lambda x: x.occurrence_count, reverse=True):
        cur_conf = e.compute_decayed_confidence()
        st_style = "[bold green]" if e.status == "PROMOTED" else ("[cyan]" if e.status == "ACTIVE" else "[dim red]")
        table.add_row(
            e.term,
            e.inferred_intent,
            f"{e.valence:+.2f}",
            str(e.occurrence_count),
            f"{e.peak_z_score:.1f}z",
            f"{cur_conf:.0%}",
            f"{st_style}{e.status}[/]",
        )

    console.print(table)


@slang_app.command(name="prune")
def slang_prune(
    lexicon: Path = typer.Option(Path("./adaptive_lexicon.json"), "--lexicon", "-l", help="Path to adaptive lexicon store"),
    threshold: float = typer.Option(0.20, "--threshold", "-t", help="Confidence threshold below which decayed terms are removed"),
):
    """Prune inactive terms whose confidence has decayed below threshold."""
    from stream_fusion.nlp.adaptive_slang import AdaptiveLexiconStore

    if not lexicon.exists():
        console.print(f"[yellow]Adaptive lexicon not found at {lexicon}[/yellow]")
        return

    store = AdaptiveLexiconStore(db_path=lexicon)
    store.apply_temporal_decay()
    pruned = store.prune_decayed(threshold=threshold)
    store.save()
    console.print(f"[bold green]Pruned {pruned} decayed terms. Remaining catalog size: {len(store.entries)} terms.[/bold green]")


# ---------------------------------------------------------------------------
# Autonomous Multi-Agent Short Production & Auto-Publisher (Spec 21)
# ---------------------------------------------------------------------------
shorts_app = typer.Typer(
    name="shorts",
    help="Autonomous Multi-Agent Short Production & Auto-Publisher Commands",
    no_args_is_help=True,
)
app.add_typer(shorts_app, name="shorts")


@shorts_app.command(name="generate")
def shorts_generate(
    analysis_path: Optional[Path] = typer.Option(None, "--analysis", "-a", help="Path to stream analysis JSON or output directory"),
    video_path: Optional[Path] = typer.Option(None, "--video", "-v", help="Path to video file for vertical clipping and thumbnail"),
    output_dir: Path = typer.Option(Path("./output/shorts"), "--output-dir", "-o", help="Directory to save generated short packages"),
    top_k: int = typer.Option(3, "--top-k", "-k", help="Maximum number of vertical shorts to produce"),
    min_highlight: float = typer.Option(0.4, "--min-highlight", help="Minimum highlight score threshold"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Skip real video encoding and use dry-run placeholders"),
):
    """Run multi-agent production committee to generate vertical shorts, copy, and packages."""
    import json
    from stream_fusion.models.schemas import StreamAnalysisResult
    from stream_fusion.production.orchestrator import ShortProductionOrchestrator

    # Locate analysis JSON
    target_json: Optional[Path] = None
    if analysis_path:
        if analysis_path.is_dir():
            cand = analysis_path / "fusion_analysis.json"
            if cand.exists():
                target_json = cand
        elif analysis_path.exists():
            target_json = analysis_path

    if not target_json:
        # Check standard default locations
        for cand in [Path("./output/fusion_analysis.json"), Path("./fusion_analysis.json")]:
            if cand.exists():
                target_json = cand
                break

    if not target_json or not target_json.exists():
        console.print("[bold red]No valid analysis JSON found. Please provide --analysis path/to/fusion_analysis.json[/bold red]")
        raise typer.Exit(code=1)

    console.print(f"[bold purple]Multi-Agent Short Studio[/bold purple]: Loading analysis from {target_json}...")
    try:
        data = json.loads(target_json.read_text(encoding="utf-8"))
        analysis = StreamAnalysisResult.model_validate(data)
    except Exception as e:
        console.print(f"[bold red]Failed to parse stream analysis JSON: {e}[/bold red]")
        raise typer.Exit(code=1)

    orchestrator = ShortProductionOrchestrator()
    console.print(f"[bold cyan]Convening Multi-Agent Production Committee (Director, Editor, Policy, Copywriter, Publisher)...[/bold cyan]")
    packages = orchestrator.produce_shorts(
        analysis=analysis,
        output_dir=output_dir,
        video_path=video_path,
        top_k=top_k,
        min_highlight_score=min_highlight,
        dry_run=dry_run,
    )

    if not packages:
        console.print("[yellow]No candidates met the threshold criteria for short production.[/yellow]")
        return

    table = Table(title=f"🎬 Autonomous Short Production Results ({len(packages)} Produced)")
    table.add_column("Short ID", style="bold cyan")
    table.add_column("Window", justify="center")
    table.add_column("Narrative Arc", style="magenta")
    table.add_column("Virality", justify="right", style="bold green")
    table.add_column("Safety", justify="center")
    table.add_column("YouTube Title", style="white")

    for pkg, env in packages:
        c = pkg.candidate
        status_style = "[bold green]PASS[/]" if pkg.audit_report.audit_status.value == "PASSED" else "[bold yellow]FLAGGED[/]"
        table.add_row(
            c.candidate_id,
            f"{c.start_sec:.1f}s - {c.end_sec:.1f}s ({c.duration_sec:.0f}s)",
            c.narrative_arc.value,
            f"{pkg.virality.overall_virality_score:.0f}/100",
            status_style,
            pkg.copy_bundle.youtube_title[:45] + "...",
        )

    console.print(table)
    console.print(f"[bold green]Successfully saved {len(packages)} packages and envelopes to {output_dir.resolve()}[/bold green]")


@shorts_app.command(name="publish")
def shorts_publish(
    package: Path = typer.Argument(..., help="Path to package_<id>.json file"),
    platform: str = typer.Option("all", "--platform", "-p", help="Target platform (youtube, tiktok, twitter, all)"),
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Dry run mode (mock dispatch) vs live API publish"),
):
    """Publish a generated short package to target social platform(s)."""
    import json
    from stream_fusion.models.schemas import ShortProductionPackage
    from stream_fusion.production.publisher import PublishDispatcher

    if not package.exists():
        console.print(f"[bold red]Package file not found: {package}[/bold red]")
        raise typer.Exit(code=1)

    try:
        data = json.loads(package.read_text(encoding="utf-8"))
        pkg = ShortProductionPackage.model_validate(data)
    except Exception as e:
        console.print(f"[bold red]Failed to load package: {e}[/bold red]")
        raise typer.Exit(code=1)

    dispatcher = PublishDispatcher()
    platforms = ["youtube", "tiktok", "twitter"] if platform.lower() == "all" else [platform.lower()]
    results = dispatcher.publish(package=pkg, platforms=platforms, dry_run=dry_run)

    table = Table(title=f"🚀 Publication Dispatch for Short [{pkg.candidate.candidate_id}]")
    table.add_column("Platform", style="bold cyan")
    table.add_column("Status", style="bold green")
    table.add_column("Post ID", style="magenta")
    table.add_column("Post URL", style="blue")

    for r in results:
        table.add_row(
            r.platform.capitalize(),
            r.status,
            r.post_id or "N/A",
            r.post_url or "N/A",
        )

    console.print(table)


@shorts_app.command(name="inspect")
def shorts_inspect(
    package: Path = typer.Argument(..., help="Path to package_<id>.json file"),
):
    """Inspect viral analytics, platform copy, and editorial cut plan for a short package."""
    import json
    from rich.panel import Panel
    from stream_fusion.models.schemas import ShortProductionPackage

    if not package.exists():
        console.print(f"[bold red]Package file not found: {package}[/bold red]")
        raise typer.Exit(code=1)

    try:
        data = json.loads(package.read_text(encoding="utf-8"))
        pkg = ShortProductionPackage.model_validate(data)
    except Exception as e:
        console.print(f"[bold red]Failed to load package: {e}[/bold red]")
        raise typer.Exit(code=1)

    c = pkg.candidate
    v = pkg.virality
    copy = pkg.copy_bundle

    details = (
        f"[bold cyan]Candidate ID:[/] {c.candidate_id}\n"
        f"[bold cyan]Time Window:[/] {c.start_sec:.1f}s - {c.end_sec:.1f}s ({c.duration_sec:.1f}s)\n"
        f"[bold cyan]Peak Timestamp:[/] {c.peak_timestamp_sec:.1f}s (Highlight Score: {c.highlight_score:.2f})\n"
        f"[bold cyan]Narrative Arc:[/] {c.narrative_arc.value} | [bold cyan]Emotion:[/] {c.primary_emotion}\n"
        f"[bold cyan]Chat Burst Z-Score:[/] +{c.chat_burst_zscore:.1f}σ | [bold cyan]Dominant Slang:[/] {', '.join(c.dominant_slang) or 'None'}\n\n"
        f"[bold green]-- Virality Scorecard --[/]\n"
        f"• Overall Virality: [bold yellow]{v.overall_virality_score:.0f}/100[/]\n"
        f"• Hook Strength: {v.hook_strength:.0f} | Pacing: {v.pacing_score:.0f} | Resonance: {v.chat_resonance:.0f}\n"
        f"• Meme Potential: {v.meme_potential:.0f} | Est. Completion Rate: {v.predicted_completion_rate:.1f}%\n\n"
        f"[bold green]-- Policy & Sponsor Compliance --[/]\n"
        f"• Status: {pkg.audit_report.audit_status.value} | Toxicity: {pkg.audit_report.toxicity_score:.2f}\n"
        f"• FTC Disclosure Required: {pkg.audit_report.ftc_disclosure_required} ({pkg.audit_report.disclosure_tag or 'None'})\n\n"
        f"[bold green]-- Platform Copy --[/]\n"
        f"[bold magenta]YouTube Title:[/] {copy.youtube_title}\n"
        f"[bold magenta]TikTok Caption:[/] {copy.tiktok_caption}\n"
        f"[bold magenta]X / Twitter Hook:[/] {copy.twitter_thread[0] if copy.twitter_thread else 'N/A'}"
    )

    console.print(Panel(details, title=f"🎬 Short Package Inspection: {package.name}", border_style="bright_blue"))


@shorts_app.command(name="list")
def shorts_list(
    output_dir: Path = typer.Option(Path("./output/shorts"), "--output-dir", "-o", help="Directory containing generated short packages"),
):
    """List all generated short packages in an output directory."""
    import json
    from stream_fusion.models.schemas import ShortProductionPackage

    if not output_dir.exists():
        console.print(f"[yellow]Shorts directory not found: {output_dir}[/yellow]")
        return

    packages: list[ShortProductionPackage] = []
    for pkg_path in output_dir.glob("package_*.json"):
        try:
            data = json.loads(pkg_path.read_text(encoding="utf-8"))
            packages.append(ShortProductionPackage.model_validate(data))
        except Exception:
            pass

    if not packages:
        console.print(f"[yellow]No valid short packages found in {output_dir}[/yellow]")
        return

    table = Table(title=f"🎬 Staged Vertical Short Packages ({len(packages)} found)")
    table.add_column("Package ID", style="bold cyan")
    table.add_column("Duration", justify="center")
    table.add_column("Virality", justify="right", style="bold green")
    table.add_column("Safety", justify="center")
    table.add_column("YouTube Title", style="white")

    for pkg in packages:
        st_style = "[bold green]PASS[/]" if pkg.audit_report.audit_status.value == "PASSED" else "[bold yellow]FLAGGED[/]"
        table.add_row(
            pkg.candidate.candidate_id,
            f"{pkg.candidate.duration_sec:.0f}s",
            f"{pkg.virality.overall_virality_score:.0f}/100",
            st_style,
            pkg.copy_bundle.youtube_title[:50] + "...",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Real-Time Live Ingestion & WebSocket Stream Tailing (Spec 19)
# ---------------------------------------------------------------------------
live_app = typer.Typer(
    name="live",
    help="Real-Time Live Stream Ingestion & WebSocket/SSE Tailing Commands",
    no_args_is_help=True,
)
app.add_typer(live_app, name="live")


@live_app.command(name="tail")
def live_tail(
    channel: str = typer.Argument(..., help="Streamer channel name or handle (e.g. asmongold)"),
    platform: str = typer.Option("TWITCH", "--platform", "-p", help="Streaming platform (TWITCH, KICK, YOUTUBE_LIVE)"),
    buffer: float = typer.Option(120.0, "--buffer", "-b", help="Depth of rolling video/audio buffer in seconds"),
    ws_port: int = typer.Option(8765, "--ws-port", help="WebSocket broadcaster port"),
    sse_port: int = typer.Option(8766, "--sse-port", help="Server-Sent Events HTTP port"),
    anonymous: bool = typer.Option(True, "--anonymous/--auth", help="Connect to chat anonymously"),
    chatroom_id: Optional[int] = typer.Option(None, "--chatroom-id", help="Direct Kick chatroom numeric ID"),
    youtube_api_key: Optional[str] = typer.Option(None, "--yt-key", help="Optional YouTube Data API v3 key"),
    stream_url: Optional[str] = typer.Option(None, "--stream-url", help="Direct stream or video URL"),
):
    """Start tailing a live broadcast with real-time rolling buffer and event broadcasting."""
    import asyncio
    from stream_fusion.models.schemas import LiveStreamConfig, LivePlatform
    from stream_fusion.ingest.live_coordinator import LiveStreamCoordinator

    console.print(f"[bold purple]StreamFusion Live Tailer[/bold purple]: Tailing channel [bold cyan]{channel}[/bold cyan] ({platform})")
    plat_enum = LivePlatform(platform.upper()) if platform.upper() in LivePlatform.__members__ else LivePlatform.TWITCH
    cfg = LiveStreamConfig(
        channel_name=channel,
        platform=plat_enum,
        stream_url=stream_url,
        buffer_duration_sec=buffer,
        ws_port=ws_port,
        sse_port=sse_port,
        anonymous_chat=anonymous,
        chatroom_id=chatroom_id,
        youtube_api_key=youtube_api_key,
    )
    coord = LiveStreamCoordinator(config=cfg)

    async def run():
        await coord.start()
        console.print(f"[bold green][OK] Live Tailer running on ws://localhost:{ws_port} and http://localhost:{ws_port}/events[/bold green]")
        console.print("[dim]Press Ctrl+C to terminate session...[/dim]")
        try:
            while coord.state.value == "RUNNING":
                await asyncio.sleep(2.0)
                st = coord.get_status()
                console.print(
                    f"  [cyan]Uptime:[/] {st.uptime_sec:.0f}s | "
                    f"[green]Chat Msgs:[/] {st.total_chat_messages} ({st.health.chat_messages_per_sec:.1f} msg/s) | "
                    f"[yellow]Subscribers:[/] {st.active_subscribers} | "
                    f"[magenta]Memory:[/] {st.health.memory_mb:.1f} MB"
                )
        except (KeyboardInterrupt, asyncio.CancelledError):
            pass
        finally:
            console.print("\n[bold yellow]Stopping Live Stream Coordinator...[/bold yellow]")
            await coord.stop()
            console.print("[bold green][OK] Stopped cleanly.[/bold green]")

    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass


@live_app.command(name="status")
def live_status():
    """Query active live tailing status."""
    console.print("[dim]No local background daemon running. Use 'streamfusion live tail' to launch interactive session.[/dim]")


# ---------------------------------------------------------------------------
# Web Grounding & Live Knowledge Graph Expansion (Spec 20)
# ---------------------------------------------------------------------------
knowledge_app = typer.Typer(
    name="knowledge",
    help="Web Grounding & Live Knowledge Graph Expansion Commands",
    no_args_is_help=True,
)
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command(name="ground")
def knowledge_ground(
    claims_path: Path = typer.Argument(..., help="Path to JSON file containing extracted claims"),
    output: Optional[Path] = typer.Option(None, "--out", "-o", help="Optional output path for grounded claims JSON"),
):
    """Ground and fact-check extracted claims against web search citations."""
    import json
    from stream_fusion.models.schemas import StreamerClaim
    from stream_fusion.knowledge.web_grounding import LiveWebGroundingEngine

    if not claims_path.exists():
        console.print(f"[bold red]Claims file not found: {claims_path}[/bold red]")
        raise typer.Exit(code=1)

    try:
        raw_data = json.loads(claims_path.read_text(encoding="utf-8"))
        if isinstance(raw_data, dict) and "claims" in raw_data:
            claims_list = [StreamerClaim.model_validate(c) for c in raw_data["claims"]]
        elif isinstance(raw_data, list):
            claims_list = [StreamerClaim.model_validate(c) for c in raw_data]
        else:
            claims_list = [StreamerClaim.model_validate(raw_data)]
    except Exception as e:
        console.print(f"[bold red]Failed to parse claims: {e}[/bold red]")
        raise typer.Exit(code=1)

    engine = LiveWebGroundingEngine()
    results = []

    table = Table(title=f"🌐 Web Grounding & Fact-Check Audit ({len(claims_list)} claims)")
    table.add_column("Claim ID", style="bold cyan")
    table.add_column("Subject", style="white")
    table.add_column("Verdict", justify="center")
    table.add_column("Citations", justify="right", style="green")
    table.add_column("Explanation", style="dim")

    for c in claims_list:
        res = engine.ground_claim(c)
        results.append(res)
        color = {
            "VERIFIED_TRUE": "bold green",
            "CONTRADICTED": "bold red",
            "OUTDATED": "bold yellow",
            "UNSUBSTANTIATED": "bold magenta",
        }.get(res.verdict.value, "white")

        table.add_row(
            res.claim_id[:12],
            c.subject,
            f"[{color}]{res.verdict.value}[/]",
            str(len(res.citations)),
            res.explanation[:60] + "...",
        )

    console.print(table)

    if output:
        out_data = [r.model_dump(mode="json") for r in results]
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(out_data, indent=2), encoding="utf-8")
        console.print(f"[bold green][OK] Grounded results saved to: {output}[/bold green]")


@knowledge_app.command(name="shifts")
def knowledge_shifts(
    entity: str = typer.Option(..., "--entity", "-e", help="Target entity name to query (e.g. Blizzard, WoW)"),
    db: Path = typer.Option(Path("./stance_history.json"), "--db", help="Path to stance history JSON"),
):
    """Query temporal stance shifts and reversals across multiple stream broadcasts."""
    from stream_fusion.knowledge.temporal_stance import TemporalStanceShiftTracker

    tracker = TemporalStanceShiftTracker(storage_path=db)
    shifts = tracker.detect_shifts(entity)

    if not shifts:
        console.print(f"[yellow]No stance shifts recorded for entity '{entity}'.[/yellow]")
        return

    table = Table(title=f"🔄 Temporal Stance Shifts for '{entity}' ({len(shifts)} shifts)")
    table.add_column("Shift ID", style="bold cyan")
    table.add_column("Transition", justify="center")
    table.add_column("Delta", justify="right")
    table.add_column("Reversal?", justify="center")
    table.add_column("Earlier Quote -> Later Quote", style="white")

    for s in shifts:
        rev_str = "[bold red]YES[/]" if s.is_reversal else "[dim]NO[/]"
        delta_str = f"[bold green]+{s.shift_delta:.2f}[/]" if s.shift_delta > 0 else f"[bold red]{s.shift_delta:.2f}[/]"
        table.add_row(
            s.shift_id[:8],
            f"{s.previous_stance} -> {s.new_stance}",
            delta_str,
            rev_str,
            f'"{s.evidence_quote_before[:30]}..." -> "{s.evidence_quote_after[:30]}..."',
        )

    console.print(table)


@knowledge_app.command(name="synthesize")
def knowledge_synthesize(
    entity: str = typer.Option(..., "--entity", "-e", help="Target entity name to synthesize"),
    db: Path = typer.Option(Path("./stance_history.json"), "--db", help="Path to stance history JSON"),
):
    """Generate longitudinal opinion synthesis and consensus brief for an entity."""
    from rich.panel import Panel
    from stream_fusion.knowledge.temporal_stance import TemporalStanceShiftTracker, CrossStreamOpinionSynthesizer

    tracker = TemporalStanceShiftTracker(storage_path=db)
    synthesizer = CrossStreamOpinionSynthesizer(tracker=tracker)
    syn = synthesizer.synthesize(entity)

    console.print(Panel(
        syn.summary,
        title=f"📊 Opinion Synthesis: {entity} (Consensus: {syn.overall_consensus_stance})",
        border_style="cyan",
    ))


# --- Spec 23: Co-Stream & Cross-Platform Alignment CLI ---

costream_app = typer.Typer(
    name="costream",
    help="Multi-Stream Co-Stream Synchronization and Cross-Platform Audience Analytics (Spec 23)",
    no_args_is_help=True,
)
app.add_typer(costream_app, name="costream")


@costream_app.command(name="align")
def costream_align(
    streams: List[Path] = typer.Option(..., "--stream", "-s", help="Paths to channel analysis or chat JSON files (provide at least 2)"),
    reference: Optional[str] = typer.Option(None, "--reference", "-r", help="Designated reference channel ID"),
    output: Path = typer.Option(Path("./costream_aligned.json"), "--out", "-o", help="Output path for aligned co-stream session JSON"),
):
    """Align multiple live stream recordings or chat files onto a unified reference clock."""
    import json
    from stream_fusion.costream.sync_engine import CrossStreamSyncEngine
    from stream_fusion.models.schemas import ChatMessage

    if len(streams) < 2:
        console.print("[bold red]Error: Must provide at least 2 stream JSON files to align.[/bold red]")
        raise typer.Exit(1)

    console.print(f"[bold purple]StreamFusion Co-Stream Synchronizer[/bold purple]: Ingesting {len(streams)} streams...")

    channel_msgs: Dict[str, List[ChatMessage]] = {}
    channel_triggers: Dict[str, List[float]] = {}

    for s_path in streams:
        ch_name = s_path.stem.replace("_analysis", "").replace("_chat", "")
        with open(s_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        msgs = []
        raw_msgs = data.get("chat_messages", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        for m in raw_msgs:
            if isinstance(m, dict):
                msgs.append(ChatMessage(**m))
        channel_msgs[ch_name] = msgs

        # Extract timestamps of significant emotes/bursts as triggers
        triggers = [m.timestamp_offset for m in msgs if len(m.emotes) > 0 or len(m.content) > 20]
        channel_triggers[ch_name] = triggers

    engine = CrossStreamSyncEngine(reference_channel_id=reference)
    sync_result = engine.calibrate_from_trigger_events(channel_triggers, reference_channel_id=reference)
    aligned = engine.align_messages(channel_msgs)

    table = Table(title=f"⏱️ Cross-Stream Latency Calibration (Anchor: {sync_result.reference_channel_id})")
    table.add_column("Channel ID", style="bold cyan")
    table.add_column("Latency Offset (s)", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Message Count", justify="right")

    for ch_id, off in sync_result.channel_offsets.items():
        conf = sync_result.confidence_scores.get(ch_id, 0.0)
        table.add_row(
            ch_id,
            f"{off:+.3f}s",
            f"{conf:.2f}",
            str(len(channel_msgs.get(ch_id, []))),
        )

    console.print(table)

    # Save aligned package
    aligned_pkg = {
        "sync_result": sync_result.model_dump(mode="json"),
        "total_aligned_messages": len(aligned),
        "aligned_messages": [
            [round(u_ts, 3), ch_id, m.model_dump(mode="json")]
            for u_ts, ch_id, m in aligned
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(aligned_pkg, f, indent=2)

    console.print(f"[bold green][OK] Aligned co-stream session saved to:[/] {output.resolve()}")


@costream_app.command(name="compare")
def costream_compare(
    aligned_file: Path = typer.Option(..., "--aligned", "-a", help="Path to aligned co-stream JSON file"),
    window_sec: float = typer.Option(2.0, "--window", "-w", help="Bucket window in seconds"),
    divergence_threshold: float = typer.Option(0.75, "--divergence-threshold", "-d", help="Divergence sensitivity threshold"),
):
    """Analyze cross-platform audience reaction and sentiment alignment from aligned stream JSON."""
    import json
    from stream_fusion.costream.audience_comparator import CrossAudienceComparator
    from stream_fusion.models.schemas import ChatMessage

    with open(aligned_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_aligned = data.get("aligned_messages", [])
    if not raw_aligned:
        console.print("[bold red]Error: No aligned messages found in file.[/bold red]")
        raise typer.Exit(1)

    parsed_aligned = []
    for item in raw_aligned:
        u_ts = float(item[0])
        ch_id = str(item[1])
        m = ChatMessage(**item[2])
        parsed_aligned.append((u_ts, ch_id, m))

    comparator = CrossAudienceComparator(
        bucket_window_sec=window_sec,
        divergence_threshold=divergence_threshold,
    )
    timeline = comparator.compute_aligned_sentiment_timeline(parsed_aligned)
    summary = comparator.summarize_session_agreement(timeline)

    table = Table(title=f"📊 Cross-Platform Audience Timeline ({len(timeline)} windows)")
    table.add_column("Unified Ts", justify="right", style="cyan")
    table.add_column("Agreement", justify="center")
    table.add_column("Platform Sentiments", style="white")
    table.add_column("Divergence?", justify="center")

    for p in timeline[:15]:  # Display top 15 windows
        agr_color = "green" if p.cross_platform_agreement >= 0.8 else ("yellow" if p.cross_platform_agreement >= 0.5 else "red")
        agr_str = f"[{agr_color}]{p.cross_platform_agreement:.2f}[/]"
        div_str = "[bold red]DIVERGENCE[/]" if p.divergence_detected else "[dim]--[/]"
        sent_str = " | ".join(f"{plat}: {score:+.2f}" for plat, score in p.platform_sentiments.items())
        table.add_row(f"{p.timestamp_sec:.1f}s", agr_str, sent_str, div_str)

    console.print(table)

    console.print(
        f"\n[bold]Session Consensus Summary[/bold]: Overall Agreement Index: [bold green]{summary['overall_agreement_index']:.2f}[/bold green] | "
        f"Divergence Moments: [bold yellow]{summary['divergence_count']}[/bold yellow] / {summary['total_points']} buckets"
    )


@costream_app.command(name="start")
def costream_start(
    config_file: Path = typer.Option(..., "--config", "-c", help="Path to CoStreamSessionConfig JSON file"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Run co-stream session for N seconds (or forever if omitted)"),
):
    """Launch real-time concurrent multi-stream co-streaming supervisor session."""
    import asyncio
    import json
    from stream_fusion.costream.coordinator import MultiStreamCoordinator
    from stream_fusion.models.schemas import CoStreamSessionConfig

    with open(config_file, "r", encoding="utf-8") as f:
        cfg_dict = json.load(f)

    session_cfg = CoStreamSessionConfig(**cfg_dict)
    coordinator = MultiStreamCoordinator(config=session_cfg)

    async def _run():
        console.print(f"[bold purple]Launching Co-Stream Session[/bold purple]: {session_cfg.session_title} ({len(session_cfg.channels)} channels)")
        await coordinator.start()
        try:
            if duration:
                await asyncio.sleep(duration)
            else:
                while coordinator.is_active:
                    await asyncio.sleep(1.0)
        finally:
            await coordinator.stop()
            console.print("[bold green]Co-stream session stopped cleanly.[/bold green]")

    asyncio.run(_run())


# --- Spec 24: Full-Spectrum Synergy CLI ---

@app.command(name="full-spectrum")
def full_spectrum_cmd(
    media: Path = typer.Argument(..., help="Path to input VOD video or audio file"),
    chat: Optional[Path] = typer.Option(None, "--chat", "-c", help="Path to Twitch or YouTube chat replay JSON"),
    output_dir: Path = typer.Option(Path("./output"), "--out", "-o", help="Output directory for all artifacts"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Limit analysis to N seconds"),
    start_time: Optional[float] = typer.Option(None, "--start", "-s", help="Start time offset in seconds"),
    shorts: int = typer.Option(3, "--shorts", "-k", help="Number of vertical shorts to produce"),
    dry_run_shorts: bool = typer.Option(False, "--dry-run-shorts", help="Stage shorts packages without rendering video"),
    isolate_workers: bool = typer.Option(False, "--isolate-workers", help="Run Whisper and Florence in isolated subprocesses"),
    bounded_buffer: bool = typer.Option(True, "--bounded-buffer/--no-bounded-buffer", help="Process keyframes in sliding window buffers"),
    ground_claims: bool = typer.Option(True, "--ground-claims/--no-ground-claims", help="Fact-check extracted claims against web sources"),
    update_slang: bool = typer.Option(True, "--update-slang/--no-update-slang", help="Extract bursts and update adaptive slang lexicon"),
    sponsor: bool = typer.Option(True, "--sponsor/--no-sponsor", help="Quantify sponsor brand mentions and engagement"),
):
    """Execute complete unified 9-phase multimodal synergy pipeline in a single command."""
    from stream_fusion.models.schemas import FullSpectrumConfig
    from stream_fusion.orchestration.full_spectrum import FullSpectrumPipeline

    cfg = FullSpectrumConfig(
        isolate_gpu_workers=isolate_workers,
        bounded_buffer=bounded_buffer,
        enable_adaptive_slang=update_slang,
        enable_web_grounding=ground_claims,
        enable_stance_tracking=True,
        enable_sponsor_quantifier=sponsor,
        enable_short_production=True,
        short_candidate_count=shorts,
        dry_run_shorts=dry_run_shorts,
    )

    pipeline = FullSpectrumPipeline(config=cfg)
    analysis, manifest = pipeline.run(
        media_input=media,
        chat_input=chat,
        output_dir=output_dir,
        duration_sec=duration,
        start_time_sec=start_time,
    )

    # Print summary report table
    table = Table(title=f"🚀 Full-Spectrum Synergy Manifest: {manifest.stream_id}")
    table.add_column("Stage", style="bold cyan")
    table.add_column("Status", justify="center")
    table.add_column("Duration", justify="right")
    table.add_column("Summary", style="white")

    for stage_name, stg in manifest.stages.items():
        st_color = "green" if stg.status == "SUCCESS" else ("yellow" if stg.status in ("DEGRADED", "SKIPPED") else "red")
        table.add_row(
            stage_name,
            f"[{st_color}]{stg.status}[/]",
            f"{stg.duration_sec:.2f}s",
            stg.output_summary or (stg.error_message or ""),
        )

    console.print(table)

    metrics_table = Table(title="📊 Multimodal Intelligence Yield")
    metrics_table.add_column("Metric", style="bold")
    metrics_table.add_column("Yield", justify="right", style="cyan")
    metrics_table.add_row("Total Media Duration", f"{manifest.effective_media_duration_sec:.1f}s")
    metrics_table.add_row("Audio Segments", str(manifest.total_audio_segments))
    metrics_table.add_row("Visual Keyframes", str(manifest.total_keyframes))
    metrics_table.add_row("Chat Messages", str(manifest.total_chat_messages))
    metrics_table.add_row("Multimodal Fusion Slices", str(manifest.total_fusion_slices))
    metrics_table.add_row("Take Agreement Mean", f"{manifest.take_agreement_mean:+.2f}")
    metrics_table.add_row("Slang Terms Updated", str(manifest.slang_terms_updated))
    metrics_table.add_row("Claims Extracted / Grounded", f"{manifest.claims_extracted_count} / {manifest.grounded_claims_count}")
    metrics_table.add_row("Stance Shifts Recorded", str(manifest.stance_shifts_count))
    metrics_table.add_row("Brand Sponsor Moments", str(manifest.sponsor_mentions_count))
    metrics_table.add_row("9:16 Shorts Produced", str(manifest.shorts_produced_count))
    console.print(metrics_table)


@app.command(name="run-all")
def run_all_cmd(
    media: Path = typer.Argument(..., help="Path to input VOD video or audio file"),
    chat: Optional[Path] = typer.Option(None, "--chat", "-c", help="Path to Twitch or YouTube chat replay JSON"),
    output_dir: Path = typer.Option(Path("./output"), "--out", "-o", help="Output directory for all artifacts"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Limit analysis to N seconds"),
    shorts: int = typer.Option(3, "--shorts", "-k", help="Number of vertical shorts to produce"),
    dry_run_shorts: bool = typer.Option(False, "--dry-run-shorts", help="Stage shorts packages without rendering video"),
):
    """Convenience alias for 'streamfusion full-spectrum'."""
    full_spectrum_cmd(
        media=media,
        chat=chat,
        output_dir=output_dir,
        duration=duration,
        start_time=None,
        shorts=shorts,
        dry_run_shorts=dry_run_shorts,
        isolate_workers=False,
        bounded_buffer=True,
        ground_claims=True,
        update_slang=True,
        sponsor=True,
    )


# --- Spec 25: Targeted Streamer Roster Ingestion & Homelab Harvester CLI ---
harvest_app = typer.Typer(
    name="harvest",
    help="Targeted Streamer Roster Ingestion & Homelab Harvester CLI (Spec 25)",
    no_args_is_help=True,
)
app.add_typer(harvest_app, name="harvest")


@harvest_app.command(name="roster-add")
def harvest_roster_add(
    streamer_id: str = typer.Option(..., "--id", help="Normalized streamer ID slug (e.g. asmongold)"),
    name: str = typer.Option(..., "--name", help="Display name of creator"),
    url: List[str] = typer.Option(..., "--url", help="Platform channel URL(s)"),
    platform: str = typer.Option("TWITCH", "--platform", help="Primary platform (TWITCH, YOUTUBE, KICK)"),
    priority: int = typer.Option(5, "--priority", help="Download priority 1-10"),
    quality: str = typer.Option("best", "--quality", help="Quality preset (best, 1080p, 720p, audio_only)"),
    lookback: int = typer.Option(14, "--lookback", help="Lookback window in days"),
    max_vods: int = typer.Option(5, "--max-vods", help="Max recent VODs to crawl per sync"),
    tags: Optional[str] = typer.Option(None, "--tags", help="Comma-separated tags"),
    db: Path = typer.Option(Path("catalog.db"), "--db", help="Path to catalog database"),
):
    """Adds or updates a target streamer in the harvester catalog."""
    from stream_fusion.harvester.catalog import HarvestCatalog
    from stream_fusion.models.schemas import StreamerTargetRecord

    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else []
    record = StreamerTargetRecord(
        streamer_id=streamer_id,
        display_name=name,
        channel_urls=url,
        primary_platform=platform.upper(),
        quality_preset=quality,
        download_priority=priority,
        lookback_days=lookback,
        max_recent_vods=max_vods,
        tags=tag_list,
    )
    catalog = HarvestCatalog(database_url=db)
    catalog.add_target(record)
    catalog.close()
    console.print(f"[bold green][OK] Added streamer target:[/bold green] {name} ({streamer_id}) [priority: {priority}]")


@harvest_app.command(name="roster-import")
def harvest_roster_import(
    file: Path = typer.Option(..., "--file", "-f", help="Path to roster file (.yaml, .json, or .csv)"),
    db: Path = typer.Option(Path("catalog.db"), "--db", help="Path to catalog database"),
):
    """Imports target streamers from a YAML, JSON, or CSV file."""
    from stream_fusion.harvester.catalog import HarvestCatalog
    from stream_fusion.harvester.roster import RosterLoader

    records = RosterLoader.load(file)
    catalog = HarvestCatalog(database_url=db)
    for r in records:
        catalog.add_target(r)
    catalog.close()
    console.print(f"[bold green][OK] Successfully imported {len(records)} streamer target(s) from {file.name}[/bold green]")


@harvest_app.command(name="roster-list")
def harvest_roster_list(
    db: Path = typer.Option(Path("catalog.db"), "--db", help="Path to catalog database"),
    enabled_only: bool = typer.Option(False, "--enabled-only", help="Filter for only enabled targets"),
):
    """Lists registered target streamers in the catalog."""
    from stream_fusion.harvester.catalog import HarvestCatalog

    catalog = HarvestCatalog(database_url=db)
    targets = catalog.list_targets(enabled_only=enabled_only)
    catalog.close()

    table = Table(title="🎯 Streamer Target Roster")
    table.add_column("Streamer ID", style="bold cyan")
    table.add_column("Display Name", style="white")
    table.add_column("Platform", justify="center")
    table.add_column("Priority", justify="center", style="bold magenta")
    table.add_column("Quality", justify="center")
    table.add_column("Channels", style="dim")
    table.add_column("Status", justify="center")

    for t in targets:
        st_color = "green" if t.enabled else "red"
        st_text = "ENABLED" if t.enabled else "DISABLED"
        table.add_row(
            t.streamer_id,
            t.display_name,
            t.primary_platform,
            str(t.download_priority),
            t.quality_preset,
            ", ".join(t.channel_urls[:2]),
            f"[{st_color}]{st_text}[/]",
        )

    console.print(table)


@harvest_app.command(name="sync")
def harvest_sync(
    db: Path = typer.Option(Path("catalog.db"), "--db", help="Path to catalog database"),
    streamer_id: Optional[str] = typer.Option(None, "--streamer-id", "-s", help="Sync specific streamer ID"),
):
    """Crawls channels to discover and queue new unprocessed VODs."""
    from stream_fusion.harvester.catalog import HarvestCatalog
    from stream_fusion.harvester.crawler import ChannelVodCrawler

    catalog = HarvestCatalog(database_url=db)
    crawler = ChannelVodCrawler()

    console.print("[bold purple]StreamFusion Crawler[/bold purple]: Syncing streamer channels for recent VODs...")
    if streamer_id:
        target = catalog.get_target(streamer_id)
        if not target:
            catalog.close()
            console.print(f"[bold red]Streamer '{streamer_id}' not found in roster.[/bold red]")
            raise typer.Exit(code=1)
        discovered = crawler.discover_target_vods(target, catalog=catalog, auto_queue=True)
        console.print(f"[bold green][OK] Streamer {streamer_id}: {len(discovered)} new VOD(s) queued.[/bold green]")
    else:
        results = crawler.sync_all(catalog=catalog, auto_queue=True)
        total = sum(len(vods) for vods in results.values())
        for s_id, vods in results.items():
            if vods:
                console.print(f"  • [cyan]{s_id}[/cyan]: {len(vods)} new VODs discovered")
        console.print(f"[bold green][OK] Sync complete: {total} total new VOD(s) queued.[/bold green]")

    catalog.close()


@harvest_app.command(name="run")
def harvest_run(
    db: Path = typer.Option(Path("catalog.db"), "--db", help="Path to catalog database"),
    homelab_root: Path = typer.Option(Path("./homelab_storage"), "--homelab-root", help="Root folder for homelab storage"),
    workers: int = typer.Option(3, "--workers", "-w", help="Max concurrent download workers"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Limit number of VODs to process"),
    simulate: bool = typer.Option(False, "--simulate", help="Run in simulation mode without external downloaders"),
):
    """Executes the harvester thread pool to download queued VODs."""
    from stream_fusion.harvester.catalog import HarvestCatalog
    from stream_fusion.harvester.engine import HomelabHarvester

    catalog = HarvestCatalog(database_url=db)
    harvester = HomelabHarvester(
        catalog=catalog,
        homelab_root=homelab_root,
        max_concurrent_workers=workers,
        simulate=simulate,
    )

    console.print(f"[bold purple]StreamFusion Harvester[/bold purple]: Processing queue with {workers} worker threads...")
    harvested = harvester.process_queue(limit=limit)
    catalog.close()

    console.print(f"[bold green][OK] Harvester finished: {len(harvested)} VOD(s) successfully harvested to {homelab_root.resolve()}[/bold green]")


@harvest_app.command(name="status")
def harvest_status(
    db: Path = typer.Option(Path("catalog.db"), "--db", help="Path to catalog database"),
    homelab_root: Path = typer.Option(Path("./homelab_storage"), "--homelab-root", help="Root folder for homelab storage"),
):
    """Displays real-time harvester queues, worker activity, and storage stats."""
    from stream_fusion.harvester.catalog import HarvestCatalog
    from stream_fusion.harvester.engine import HomelabHarvester

    catalog = HarvestCatalog(database_url=db)
    harvester = HomelabHarvester(
        catalog=catalog,
        homelab_root=homelab_root,
    )
    status_rep = harvester.get_status()
    catalog.close()

    table = Table(title="📡 Homelab Harvester Status")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    table.add_row("Active Workers", f"{status_rep.active_workers} / {status_rep.max_workers}")
    table.add_row("Queue Depth (Queued VODs)", str(status_rep.queue_depth))
    table.add_row("Currently Downloading", str(status_rep.downloading_count))
    table.add_row("Harvested / Ready", str(status_rep.harvested_count))
    table.add_row("Errors Encountered", str(status_rep.error_count))
    table.add_row("Free Disk Space", f"{status_rep.free_disk_gb:.1f} GB")

    console.print(table)


@harvest_app.command(name="ingest-to-pipeline")
def harvest_ingest_to_pipeline(
    vod_id: str = typer.Option(..., "--vod-id", help="VOD ID in the catalog to analyze"),
    db: Path = typer.Option(Path("catalog.db"), "--db", help="Path to catalog database"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Custom output directory"),
    duration: Optional[float] = typer.Option(None, "--duration", "-d", help="Limit analysis to N seconds"),
    shorts: int = typer.Option(3, "--shorts", "-k", help="Number of vertical shorts to produce"),
    dry_run_shorts: bool = typer.Option(False, "--dry-run-shorts", help="Stage shorts packages without rendering video"),
    ground_claims: bool = typer.Option(True, "--ground-claims/--no-ground-claims", help="Ground factual claims against live web"),
    update_slang: bool = typer.Option(True, "--update-slang/--no-update-slang", help="Update adaptive slang lexicon from chat"),
    sponsor: bool = typer.Option(True, "--sponsor/--no-sponsor", help="Audit stream sponsors"),
    verify_checksums: bool = typer.Option(True, "--verify-checksums/--no-verify-checksums", help="Verify SHA-256 integrity"),
):
    """Bridges a harvested homelab VOD directly into the Full-Spectrum Pipeline."""
    from stream_fusion.harvester.catalog import HarvestCatalog
    from stream_fusion.harvester.bridge import ingest_to_pipeline

    catalog = HarvestCatalog(database_url=db)
    console.print(f"[bold purple]StreamFusion Pipeline Bridge[/bold purple]: Ingesting {vod_id}...")

    analysis, manifest = ingest_to_pipeline(
        vod_id=vod_id,
        catalog=catalog,
        output_dir=out,
        duration_sec=duration,
        verify_checksums=verify_checksums,
        run_shorts=(shorts > 0),
        dry_run_shorts=dry_run_shorts,
        ground_claims=ground_claims,
        update_slang=update_slang,
        sponsor_audit=sponsor,
        short_candidate_count=shorts,
    )
    catalog.close()

    console.print(f"[bold green][OK] Successfully ingested {vod_id} into Full-Spectrum Pipeline![/bold green]")
    console.print(f"  • Analysis ID: [cyan]{analysis.stream_id}[/cyan]")
    console.print(f"  • Total Slices: [cyan]{manifest.total_fusion_slices}[/cyan]")
    console.print(f"  • Shorts Produced: [cyan]{manifest.shorts_produced_count}[/cyan]")


# --- Spec 26: Homelab 24/7 Scheduler Daemon & Background Supervisor CLI ---
daemon_app = typer.Typer(
    name="daemon",
    help="Homelab 24/7 Scheduler Daemon & Background Supervisor (Spec 26)",
    no_args_is_help=True,
)
app.add_typer(daemon_app, name="daemon")


@daemon_app.command(name="run")
def daemon_run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to daemon YAML/JSON configuration"),
    foreground: bool = typer.Option(True, "--foreground/--background", help="Run supervisor in foreground"),
    auto_analyze: bool = typer.Option(True, "--auto-analyze/--no-auto-analyze", help="Automatically trigger pipeline for harvested VODs"),
    workers: Optional[int] = typer.Option(None, "--workers", "-w", help="Override max concurrent download workers"),
    db: Optional[str] = typer.Option(None, "--db", help="Override catalog database URL"),
    homelab_root: Optional[Path] = typer.Option(None, "--homelab-root", help="Override root homelab storage path"),
    crawl_interval: Optional[int] = typer.Option(None, "--crawl-interval", help="Channel crawl interval in minutes"),
):
    """Starts the StreamFusion 24/7 background supervisor daemon."""
    import yaml
    from stream_fusion.harvester.daemon import HomelabDaemon
    from stream_fusion.models.schemas import DaemonConfig

    d_config = DaemonConfig()

    # Load from config file if provided or default exists
    cfg_path = config or Path("config/daemon.yaml")
    if cfg_path and cfg_path.exists():
        console.print(f"[bold purple]StreamFusion Daemon[/bold purple]: Loading configuration from {cfg_path}...")
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            d_config = DaemonConfig(**data)

    # Apply CLI overrides
    d_config.auto_analyze = auto_analyze
    if workers is not None:
        d_config.max_concurrent_downloads = workers
    if db is not None:
        d_config.catalog_db_url = db
    if homelab_root is not None:
        d_config.homelab_root = str(homelab_root)
    if crawl_interval is not None:
        d_config.crawl_interval_minutes = crawl_interval

    console.print("[bold green]Starting StreamFusion Homelab Daemon...[/bold green]")
    console.print(f"  • Storage Root: [cyan]{d_config.homelab_root}[/cyan]")
    console.print(f"  • Database: [cyan]{d_config.catalog_db_url}[/cyan]")
    console.print(f"  • Crawl Interval: [cyan]{d_config.crawl_interval_minutes}m[/cyan]")
    console.print(f"  • Auto-Analyze: [cyan]{d_config.auto_analyze}[/cyan]")

    daemon = HomelabDaemon(config=d_config)
    try:
        daemon.start(foreground=foreground)
    except KeyboardInterrupt:
        console.print("\n[yellow]Shutdown signal received. Stopping daemon...[/yellow]")
        daemon.stop()
    except Exception as e:
        console.print(f"[bold red]Daemon execution error:[/bold red] {e}")
        raise typer.Exit(code=1)


@daemon_app.command(name="status")
def daemon_status(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to daemon YAML/JSON configuration"),
    pid_file: Path = typer.Option(Path("daemon.pid"), "--pid-file", help="Path to daemon PID lockfile"),
    db: Path = typer.Option(Path("homelab_storage/catalog.db"), "--db", help="Path to catalog database"),
):
    """Displays real-time operational status, queues, and health of the daemon."""
    import psutil
    from stream_fusion.harvester.catalog import HarvestCatalog
    from stream_fusion.harvester.daemon import is_process_running

    pid = None
    is_running = False
    if pid_file.exists():
        try:
            with open(pid_file, "r", encoding="utf-8") as f:
                p_text = f.read().strip()
                if p_text:
                    p_val = int(p_text)
                    if is_process_running(p_val):
                        pid = p_val
                        is_running = True
        except Exception:
            pass

    table = Table(title="🖥️  StreamFusion 24/7 Homelab Daemon Status")
    table.add_column("Property", style="bold")
    table.add_column("Value", style="cyan")

    state_style = "[bold green]RUNNING[/bold green]" if is_running else "[bold yellow]STOPPED[/bold yellow]"
    table.add_row("Daemon State", state_style)
    table.add_row("Process ID (PID)", str(pid) if pid else "N/A")

    if is_running and pid:
        try:
            proc = psutil.Process(pid)
            cpu_pct = proc.cpu_percent(interval=0.1)
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            table.add_row("Daemon CPU", f"{cpu_pct:.1f}%")
            table.add_row("Daemon Memory (RAM)", f"{mem_mb:.1f} MB")
        except Exception:
            pass

    # Read catalog queues if catalog DB exists
    db_file = db if db.exists() else Path("catalog.db")
    if db_file.exists():
        cat = HarvestCatalog(database_url=db_file)
        counts = cat.count_vods_by_status()
        cat.close()
        table.add_section()
        table.add_row("Queued VODs", str(counts.get("QUEUED", 0)))
        table.add_row("Downloading VODs", str(counts.get("DOWNLOADING", 0)))
        table.add_row("Harvested (Ready)", str(counts.get("HARVESTED", 0) + counts.get("READY_FOR_ANALYSIS", 0)))
        table.add_row("Analyzed (Complete)", str(counts.get("ANALYZED", 0)))
        table.add_row("Errors Encountered", str(counts.get("ERROR", 0)))

    console.print(table)


@daemon_app.command(name="stop")
def daemon_stop(
    pid_file: Path = typer.Option(Path("daemon.pid"), "--pid-file", help="Path to daemon PID lockfile"),
):
    """Gracefully terminates the background daemon process."""
    import signal
    import psutil
    from stream_fusion.harvester.daemon import is_process_running

    if not pid_file.exists():
        console.print("[yellow]No daemon PID lockfile found. Daemon is not running.[/yellow]")
        return

    try:
        with open(pid_file, "r", encoding="utf-8") as f:
            pid = int(f.read().strip())
    except Exception as e:
        console.print(f"[bold red]Could not read PID file:[/bold red] {e}")
        raise typer.Exit(code=1)

    if not is_process_running(pid):
        console.print(f"[yellow]Process {pid} is not running. Cleaning up stale PID file.[/yellow]")
        pid_file.unlink(missing_ok=True)
        return

    console.print(f"[bold purple]Sending graceful termination signal to Daemon (PID: {pid})...[/bold purple]")
    try:
        proc = psutil.Process(pid)
        proc.terminate()
        proc.wait(timeout=10)
        console.print("[bold green][OK] Daemon stopped successfully.[/bold green]")
    except psutil.TimeoutExpired:
        console.print("[bold red]Daemon did not exit within timeout. Killing process...[/bold red]")
        proc.kill()
    except Exception as e:
        console.print(f"[bold red]Failed to stop daemon:[/bold red] {e}")
        raise typer.Exit(code=1)


@daemon_app.command(name="trigger-crawl")
def daemon_trigger_crawl(
    db: Path = typer.Option(Path("homelab_storage/catalog.db"), "--db", help="Path to catalog database"),
):
    """Triggers an immediate channel crawl for all target streamers."""
    from stream_fusion.harvester.catalog import HarvestCatalog
    from stream_fusion.harvester.crawler import ChannelVodCrawler

    db_path = db if db.exists() else Path("catalog.db")
    catalog = HarvestCatalog(database_url=db_path)
    crawler = ChannelVodCrawler()

    console.print("[bold purple]StreamFusion Crawler[/bold purple]: Triggering immediate channel crawl...")
    results = crawler.sync_all(catalog=catalog, auto_queue=True)
    total = sum(len(vods) for vods in results.values())
    for s_id, vods in results.items():
        if vods:
            console.print(f"  • [cyan]{s_id}[/cyan]: {len(vods)} new VODs discovered")
    console.print(f"[bold green][OK] Crawl finished: {total} total new VOD(s) queued.[/bold green]")
    catalog.close()


@daemon_app.command(name="init-config")
def daemon_init_config(
    output: Path = typer.Option(Path("config/daemon.yaml"), "--output", "-o", help="Destination config path"),
):
    """Initializes a new daemon configuration file with default settings."""
    import yaml
    from stream_fusion.models.schemas import DaemonConfig

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        console.print(f"[yellow]Configuration file already exists at {output}[/yellow]")
        return

    d_config = DaemonConfig()
    with open(output, "w", encoding="utf-8") as f:
        yaml.dump(d_config.model_dump(mode="json"), f, sort_keys=False)

    console.print(f"[bold green][OK] Created default daemon configuration at {output}[/bold green]")


if __name__ == "__main__":

    app()





