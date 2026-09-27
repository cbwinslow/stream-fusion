"""StreamFusion Command Line Interface."""

from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table

from stream_fusion.models.schemas import AudioSegment, VisualKeyframe, ChatMessage
from stream_fusion.chat.analyzer import ChatAnalyzer
from stream_fusion.fusion.matrix import FusionEngine
from stream_fusion.export.html_report import export_html_report

app = typer.Typer(
    name="streamfusion",
    help="Multimodal Livestream VOD & Chat Grounding and Analysis CLI",
    no_args_is_help=True,
)
console = Console()


@app.command()
def analyze(
    url: str = typer.Argument(..., help="Twitch or YouTube VOD URL"),
    duration: float = typer.Option(180.0, "--duration", "-d", help="Max duration in seconds"),
    output_dir: Path = typer.Option(Path("./output"), "--out", "-o", help="Output directory"),
):
    """Analyze a livestream VOD and aligned chat replay."""
    console.print(f"[bold purple]StreamFusion Ingestion[/bold purple]: {url}")
    console.print(f"Max segment duration: {duration}s -> Target: {output_dir}")
    console.print("[yellow]Ingestion pipeline ready for full media demuxing.[/yellow]")



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
    # Note: Streamer paused at 15s. Chat starts reacting around 19s (4s delay).
    raw_chat = []
    # Baseline chat
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
    # Chat burst after the take (19s to 35s)
    react_emotes = ["OMEGALUL", "ICANT", "COOKED", "L", "Aware", "KEKW", "TRUE"]
    for s in range(19, 45):
        # Spikes: 6 messages per second
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

    for s in result.slices[6:16]:  # Show around the pause moment
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


if __name__ == "__main__":
    app()
