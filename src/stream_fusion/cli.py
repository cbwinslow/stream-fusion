"""StreamFusion Command Line Interface."""

from pathlib import Path
from typing import Optional
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
):
    """Run full multimodal grounding on a video VOD and chat replay."""
    console.print(f"[bold purple]StreamFusion Pipeline[/bold purple]: Processing {video.name}")
    config = StreamFusionConfig()
    config.chat.auto_calibrate_latency = auto_latency
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


if __name__ == "__main__":
    app()


