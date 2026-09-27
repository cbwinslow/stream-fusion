"""Interactive HTML report generator for StreamFusion results."""

from pathlib import Path
import json
from stream_fusion.models.schemas import StreamAnalysisResult


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>StreamFusion Report: {stream_id}</title>
    <style>
        :root {{
            --bg: #0f1117;
            --surface: #1a1d24;
            --border: #2a2e39;
            --text: #e1e7ec;
            --muted: #8b949e;
            --accent: #9146FF; /* Twitch Purple */
            --streamer: #58a6ff;
            --video: #e3b341;
            --spike: #f85149;
            --pos: #3fb950;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 24px;
        }}
        .header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 16px;
            margin-bottom: 24px;
        }}
        .header h1 {{ margin: 0; font-size: 24px; display: flex; align-items: center; gap: 8px; }}
        .badge {{ background: var(--accent); color: white; padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 600; }}
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-bottom: 24px;
        }}
        .stat-card {{
            background: var(--surface);
            padding: 16px;
            border-radius: 8px;
            border: 1px solid var(--border);
        }}
        .stat-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; font-weight: 600; }}
        .stat-val {{ font-size: 24px; font-weight: 700; margin-top: 4px; }}
        
        .timeline-table {{
            width: 100%;
            border-collapse: collapse;
            background: var(--surface);
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--border);
        }}
        .timeline-table th {{
            background: #21262d;
            padding: 12px 16px;
            text-align: left;
            font-size: 13px;
            color: var(--muted);
            border-bottom: 1px solid var(--border);
        }}
        .timeline-table td {{
            padding: 12px 16px;
            border-bottom: 1px solid var(--border);
            font-size: 13px;
            vertical-align: top;
        }}
        .row-spike {{ background: rgba(248, 81, 73, 0.08); border-left: 4px solid var(--spike); }}
        .streamer-speech {{ color: var(--streamer); font-weight: 500; }}
        .video-speech {{ color: var(--video); font-style: italic; }}
        .emote-tag {{
            display: inline-block;
            background: #2d333b;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            margin-right: 4px;
            margin-bottom: 4px;
        }}
        .spike-tag {{
            background: var(--spike);
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>StreamFusion Grounding Report <span class="badge">VOD Grounded</span></h1>
        <div style="color: var(--muted)">Stream ID: <code>{stream_id}</code></div>
    </div>

    <div class="stats-grid">
        <div class="stat-card">
            <div class="stat-label">Duration</div>
            <div class="stat-val">{duration_str}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Total Chat Messages</div>
            <div class="stat-val">{total_messages}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">Detected Highlights / Spikes</div>
            <div class="stat-val" style="color: var(--spike)">{total_highlights}</div>
        </div>
    </div>

    <h2>Synchronized Multimodal Matrix</h2>
    <table class="timeline-table">
        <thead>
            <tr>
                <th style="width: 80px;">Time</th>
                <th style="width: 250px;">Streamer vs Video Audio</th>
                <th style="width: 250px;">Screen & Visual Context</th>
                <th>Audience Chat Activity (Calibrated)</th>
                <th style="width: 100px;">Signals</th>
            </tr>
        </thead>
        <tbody>
            {table_rows}
        </tbody>
    </table>
</body>
</html>
"""


def format_timestamp(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"


def export_html_report(result: StreamAnalysisResult, output_path: Path):
    """Renders the analysis result to a self-contained interactive HTML file."""
    rows = []
    for s in result.slices:
        spike_class = "row-spike" if s.is_spike_moment else ""
        t_str = format_timestamp(s.start_sec)

        # Audio column
        audio_parts = []
        if s.streamer_transcript:
            audio_parts.append(
                f'<div class="streamer-speech">🗣️ <b>Streamer:</b> "{s.streamer_transcript}"</div>'
            )
        if s.external_audio_transcript:
            audio_parts.append(
                f'<div class="video-speech">📺 <b>Video:</b> "{s.external_audio_transcript}"</div>'
            )
        audio_html = "".join(audio_parts) if audio_parts else '<span style="color:#555">—</span>'

        # Visual column
        vis_html = f"<div><b>[{s.active_scene_type}]</b> {s.visual_description}</div>"
        if s.screen_ocr:
            ocr_preview = ", ".join(s.screen_ocr[:3])
            vis_html += f'<div style="font-size: 11px; color: var(--muted); margin-top:4px;">OCR: {ocr_preview}</div>'

        # Chat column
        emotes_html = "".join(
            [f'<span class="emote-tag">{k} ({v})</span>' for k, v in s.dominant_emotes.items()]
        )
        chat_html = f"""
            <div><b>{s.chat_message_count} msgs</b> ({s.chat_velocity_per_sec:.1f}/s) | Sentiment: {s.chat_sentiment_polarity:+.2f}</div>
            <div style="margin-top: 4px;">{emotes_html}</div>
        """

        # Signals column
        sig_parts = []
        if s.is_spike_moment:
            sig_parts.append('<span class="spike-tag">HYPE / SPIKE</span>')
        sig_html = "".join(sig_parts) if sig_parts else ""

        rows.append(
            f"""<tr class="{spike_class}">
                <td><code>{t_str}</code></td>
                <td>{audio_html}</td>
                <td>{vis_html}</td>
                <td>{chat_html}</td>
                <td>{sig_html}</td>
            </tr>"""
        )

    html = HTML_TEMPLATE.format(
        stream_id=result.stream_id,
        duration_str=format_timestamp(result.duration_sec),
        total_messages=result.total_chat_messages,
        total_highlights=len(result.highlights),
        table_rows="\n".join(rows),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
