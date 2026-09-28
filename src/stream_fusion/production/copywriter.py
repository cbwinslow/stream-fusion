"""Copywriter Agent for Autonomous Short Production (Spec 21).

Crafts viral hooks, platform-optimized metadata, slang hashtags, and virality scoring.
"""

from typing import List, Optional

from stream_fusion.models.schemas import (
    AdaptiveTermEntry,
    ContentAuditReport,
    PlatformCopyBundle,
    ShortCandidate,
    ViralityScoreCard,
)


class CopywriterAgent:
    """Generates viral social copy, hashtags, and virality analytics across platforms."""

    def generate_copy(
        self,
        candidate: ShortCandidate,
        audit: ContentAuditReport,
        adaptive_terms: Optional[List[AdaptiveTermEntry]] = None,
        creator_handle: str = "Streamer",
    ) -> tuple[PlatformCopyBundle, ViralityScoreCard]:
        """Produces platform copy for YouTube Shorts, TikTok, X/Twitter, and virality scorecard."""
        # 1. Synthesize dominant slang into hashtags
        slang_tags = [f"#{term}" for term in candidate.dominant_slang if len(term) >= 3]
        if adaptive_terms:
            for term in adaptive_terms[:3]:
                tag = f"#{term.term}"
                if tag not in slang_tags:
                    slang_tags.append(tag)

        # 2. Viral Hook & Headline
        hook_headline = self._craft_hook_headline(candidate)

        # 3. YouTube Shorts Metadata
        yt_title = self._craft_youtube_title(candidate, audit)
        yt_tags = list(set(["Shorts", "TwitchClips", "Gaming", candidate.primary_emotion.lower()] + [t.strip("#") for t in slang_tags]))
        yt_desc = self._craft_youtube_description(candidate, audit, creator_handle, slang_tags)

        # 4. TikTok Metadata
        tiktok_hashtags = list(set(["#fyp", "#viral", "#gaming", "#streamer", "#twitch"] + slang_tags))
        if audit.ftc_disclosure_required and audit.disclosure_tag:
            tiktok_hashtags.append("#ad")
        tiktok_caption = f"{hook_headline} 😂 Watch till the end! {' '.join(tiktok_hashtags[:8])}"

        # 5. X / Twitter Thread
        twitter_thread = self._craft_twitter_thread(candidate, audit, creator_handle, slang_tags)

        copy_bundle = PlatformCopyBundle(
            youtube_title=yt_title[:100],
            youtube_description=yt_desc,
            youtube_tags=yt_tags[:15],
            tiktok_caption=tiktok_caption[:2200],
            tiktok_hashtags=tiktok_hashtags,
            twitter_thread=twitter_thread,
            hook_headline=hook_headline,
        )

        # 6. Virality Scorecard Calculation
        scorecard = self._compute_virality_scorecard(candidate, audit)

        return copy_bundle, scorecard

    def _craft_hook_headline(self, candidate: ShortCandidate) -> str:
        """Constructs an attention-grabbing hook headline based on narrative arc and emotion."""
        emotion = candidate.primary_emotion.upper()
        if "LAUGHTER" in emotion or "HYSTERICAL" in emotion:
            return f"He couldn't breathe after this happened 💀"
        elif "CONTROVERSY" in emotion or "HOT_TAKE" in emotion:
            return f"The hot take that broke chat instantly 🔥"
        elif "HYPE" in emotion or "EXCITEMENT" in emotion:
            return f"UNREAL play that nobody saw coming 🤯"
        elif candidate.hook_text:
            cleaned = candidate.hook_text.strip().rstrip(".").capitalize()
            return f"'{cleaned}' 👀"
        else:
            return f"Watch this insane stream moment unfold ⚡"

    def _craft_youtube_title(self, candidate: ShortCandidate, audit: ContentAuditReport) -> str:
        """Formats YouTube Shorts title (strictly <= 100 chars)."""
        base = self._craft_hook_headline(candidate)
        suffix = " #Shorts"
        if audit.ftc_disclosure_required:
            suffix = " #ad #Shorts"

        max_len = 100 - len(suffix)
        title = base[:max_len] + suffix
        return title

    def _craft_youtube_description(
        self,
        candidate: ShortCandidate,
        audit: ContentAuditReport,
        creator: str,
        slang_tags: List[str],
    ) -> str:
        """Formats comprehensive YouTube description."""
        lines = [
            f"🎬 Moment captured from @{creator}'s live broadcast!",
            "",
            f"📌 Summary: {candidate.summary}",
            f"⚡ Climax Peak: {int(candidate.peak_timestamp_sec)}s | Duration: {int(candidate.duration_sec)}s",
        ]
        if audit.ftc_disclosure_required:
            brands = ", ".join(audit.sponsor_brand_names) or "Partner"
            lines.append(f"\n📢 Sponsored integration with {brands}. #ad #sponsored")

        lines.extend([
            "",
            "🔔 Subscribe for daily stream highlights & AI-fused reactions!",
            " ".join(slang_tags + ["#Shorts", "#TwitchHighlights", "#ViralClips"]),
        ])
        return "\n".join(lines)

    def _craft_twitter_thread(
        self,
        candidate: ShortCandidate,
        audit: ContentAuditReport,
        creator: str,
        slang_tags: List[str],
    ) -> List[str]:
        """Crafts a 3-part X/Twitter thread."""
        tweet1 = (
            f"🚨 Best moment from @{creator}'s stream today:\n\n"
            f"{self._craft_hook_headline(candidate)}\n\n"
            f"Watch the full clip below 🧵👇"
        )
        if audit.ftc_disclosure_required:
            tweet1 += " (Ad)"

        tweet2 = (
            f"Key Takeaway & Reaction:\n"
            f"\"{candidate.summary}\"\n\n"
            f"Chat burst velocity reached +{candidate.chat_burst_zscore:.1f}σ! {' '.join(slang_tags[:3])}"
        )

        tweet3 = (
            f"Did @{creator} make the right call here?\n\n"
            f"Drop your takes in the replies! RT if this made you laugh 🔁"
        )

        return [tweet1[:280], tweet2[:280], tweet3[:280]]

    def _compute_virality_scorecard(
        self, candidate: ShortCandidate, audit: ContentAuditReport
    ) -> ViralityScoreCard:
        """Computes multi-factor virality score (0-100) and audience completion predictions."""
        # 1. Hook strength: based on emotion and short hook length
        hook_val = 70.0
        if len(candidate.hook_text) > 0:
            hook_val += 15.0
        if candidate.primary_emotion in ["HYSTERICAL_LAUGHTER", "HYPE", "CONTROVERSY"]:
            hook_val += 10.0
        hook_strength = min(100.0, hook_val)

        # 2. Chat resonance: scaled from burst Z-score
        chat_resonance = min(100.0, max(20.0, candidate.chat_burst_zscore * 20.0 + 30.0))

        # 3. Pacing score: optimal duration is 20-35s
        dur = candidate.duration_sec
        if 20.0 <= dur <= 35.0:
            pacing = 95.0
        elif 15.0 <= dur <= 50.0:
            pacing = 80.0
        else:
            pacing = 65.0

        # 4. Meme potential: based on dominant slang occurrences
        meme_pot = min(100.0, 50.0 + len(candidate.dominant_slang) * 12.0)

        # 5. Composite Virality Score
        overall = (
            0.35 * hook_strength +
            0.25 * chat_resonance +
            0.20 * pacing +
            0.20 * meme_pot
        )
        # Apply safety penalty if flagged
        if audit.audit_status.value == "FLAGGED":
            overall *= 0.85
        elif audit.audit_status.value == "REJECTED":
            overall *= 0.30

        # Estimated completion rate
        completion = min(96.0, max(50.0, 85.0 - (dur * 0.3) + (overall * 0.15)))

        return ViralityScoreCard(
            overall_virality_score=round(overall, 1),
            hook_strength=round(hook_strength, 1),
            pacing_score=round(pacing, 1),
            chat_resonance=round(chat_resonance, 1),
            meme_potential=round(meme_pot, 1),
            predicted_completion_rate=round(completion, 1),
        )
