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


if __name__ == "__main__":
    app()

