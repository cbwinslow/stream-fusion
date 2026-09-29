/**
 * StreamFusion Studio Client Application (Spec 27)
 */

document.addEventListener("DOMContentLoaded", () => {
  initTabs();
  initWebSocket();
  loadOverview();
  loadRoster();
  loadCatalog();
  loadShorts();
  loadKnowledge();
  loadLiveStatus();
});

// --- Tab Switching ---
function initTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      tabBtns.forEach((b) => b.classList.remove("active"));
      document.querySelectorAll(".view-panel").forEach((p) => p.classList.remove("active"));

      btn.classList.add("active");
      const targetId = btn.getAttribute("data-tab");
      const targetPanel = document.getElementById(targetId);
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });
}

// --- WebSocket Live Connection ---
let ws = null;
let reconnectTimer = null;

function initWebSocket() {
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const wsUrl = `${protocol}//${window.location.host}/ws/live`;
  const pulseDot = document.getElementById("ws-pulse");
  const wsStatusText = document.getElementById("ws-status-text");

  try {
    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
      if (pulseDot) pulseDot.classList.remove("offline");
      if (wsStatusText) wsStatusText.textContent = "WS Connected";
      console.log("[StreamFusion WS] Connected to live event stream.");
      if (reconnectTimer) {
        clearTimeout(reconnectTimer);
        reconnectTimer = null;
      }
    };

    ws.onmessage = (event) => {
      try {
        const envelope = JSON.parse(event.data);
        handleIncomingEnvelope(envelope);
      } catch (err) {
        console.error("WS Parse error", err);
      }
    };

    ws.onclose = () => {
      if (pulseDot) pulseDot.classList.add("offline");
      if (wsStatusText) wsStatusText.textContent = "WS Offline";
      if (!reconnectTimer) {
        reconnectTimer = setTimeout(initWebSocket, 3000);
      }
    };

    ws.onerror = () => {
      ws.close();
    };
  } catch (err) {
    if (pulseDot) pulseDot.classList.add("offline");
    if (wsStatusText) wsStatusText.textContent = "WS Offline";
    reconnectTimer = setTimeout(initWebSocket, 3000);
  }
}

function handleIncomingEnvelope(envelope) {
  const type = envelope.event_type;
  const payload = envelope.payload;

  if (type === "CHAT_MESSAGE" || type === "LIVE_CHAT_MESSAGE") {
    appendLiveChatMessage(payload);
  } else if (type === "MEME_BURST" || type === "BURST_DETECTED") {
    recordLiveBurst(payload);
  } else if (type === "DAEMON_STATUS" || type === "PIPELINE_PROGRESS") {
    loadOverview();
  }
}

// --- View 1: Homelab & Harvester Overview ---
async function loadOverview() {
  try {
    const res = await fetch("/api/system/stats");
    if (!res.ok) return;
    const stats = await res.json();

    document.getElementById("stat-streamers").textContent = stats.streamer_count;
    document.getElementById("stat-vods").textContent = stats.total_vods_count;
    document.getElementById("stat-analyzed").textContent = stats.analyzed_vods_count;
    document.getElementById("stat-free-storage").textContent = `${stats.free_storage_gb} GB`;

    const daemonBadge = document.getElementById("daemon-state-badge");
    if (daemonBadge) {
      daemonBadge.textContent = stats.daemon_state;
      daemonBadge.className = "badge " + (stats.daemon_state === "RUNNING" ? "badge-green" : "badge-yellow");
    }
  } catch (err) {
    console.error("Failed to load overview stats", err);
  }
}

async function loadRoster() {
  try {
    const res = await fetch("/api/roster");
    if (!res.ok) return;
    const targets = await res.json();
    const tbody = document.getElementById("roster-table-body");
    if (!tbody) return;

    if (targets.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-muted);">No streamers configured yet. Add one above!</td></tr>`;
      return;
    }

    tbody.innerHTML = targets
      .map(
        (t) => `
        <tr>
          <td><strong>${t.display_name}</strong></td>
          <td><span class="badge badge-purple">${t.primary_platform}</span></td>
          <td>${t.streamer_id}</td>
          <td>P${t.download_priority}</td>
          <td><span class="badge ${t.enabled ? "badge-green" : "badge-red"}">${t.enabled ? "ENABLED" : "PAUSED"}</span></td>
          <td>
            <button class="btn btn-sm btn-danger" onclick="deleteStreamer('${t.streamer_id}')">Remove</button>
          </td>
        </tr>
      `
      )
      .join("");
  } catch (err) {
    console.error("Failed to load roster", err);
  }
}

async function addStreamerPrompt() {
  const streamerId = prompt("Enter streamer handle / ID (e.g. asmongold):");
  if (!streamerId) return;
  const platform = prompt("Enter platform (twitch, youtube, kick):", "twitch") || "twitch";
  const channelUrl = prompt("Enter channel URL:", `https://twitch.tv/${streamerId}`) || `https://twitch.tv/${streamerId}`;

  const payload = {
    streamer_id: streamerId.toLowerCase().trim(),
    display_name: streamerId.trim(),
    channel_urls: [channelUrl],
    primary_platform: platform.toLowerCase().trim(),
    download_priority: 5,
    enabled: true,
  };

  try {
    const res = await fetch("/api/roster", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    if (res.ok) {
      loadRoster();
      loadOverview();
    } else {
      alert("Failed to add streamer target.");
    }
  } catch (err) {
    alert("Network error adding streamer.");
  }
}

async function deleteStreamer(streamerId) {
  if (!confirm(`Remove streamer '${streamerId}' from tracking roster?`)) return;
  try {
    const res = await fetch(`/api/roster/${streamerId}`, { method: "DELETE" });
    if (res.ok) {
      loadRoster();
      loadOverview();
    }
  } catch (err) {
    alert("Error deleting streamer.");
  }
}

// Daemon Actions
async function daemonAction(action) {
  try {
    const res = await fetch(`/api/daemon/${action}`, { method: "POST" });
    const data = await res.json();
    alert(`Daemon: ${data.message}`);
    loadOverview();
  } catch (err) {
    alert(`Failed to execute daemon ${action}`);
  }
}

// Catalog Actions
async function loadCatalog() {
  try {
    const res = await fetch("/api/catalog?limit=20");
    if (!res.ok) return;
    const vods = await res.json();
    const tbody = document.getElementById("catalog-table-body");
    const vodSelect = document.getElementById("player-vod-select");
    if (!tbody) return;

    if (vodSelect) {
      vodSelect.innerHTML = vods.map((v) => `<option value="${v.vod_id}">${v.title || v.vod_id} (${v.streamer_id})</option>`).join("");
      if (vods.length > 0) {
        loadVodTimeline(vods[0].vod_id);
      }
    }

    if (vods.length === 0) {
      tbody.innerHTML = `<tr><td colspan="6" style="text-align:center; color:var(--text-muted);">No VODs discovered yet. Run a channel crawl!</td></tr>`;
      return;
    }

    tbody.innerHTML = vods
      .map((v) => {
        let badgeClass = "badge-blue";
        if (v.status === "ANALYZED") badgeClass = "badge-green";
        else if (v.status === "FAILED") badgeClass = "badge-red";
        else if (v.status === "DOWNLOADING") badgeClass = "badge-yellow";

        return `
        <tr>
          <td><strong>${v.title || v.vod_id}</strong></td>
          <td>${v.streamer_id}</td>
          <td><span class="badge ${badgeClass}">${v.status}</span></td>
          <td>${Math.round(v.duration_sec)}s</td>
          <td>${Math.round((v.file_size_bytes || 0) / (1024 * 1024))} MB</td>
          <td>
            <button class="btn btn-sm btn-secondary" onclick="analyzeVod('${v.vod_id}')">Analyze</button>
          </td>
        </tr>
      `;
      })
      .join("");
  } catch (err) {
    console.error("Failed to load catalog", err);
  }
}

async function triggerCrawl() {
  try {
    const res = await fetch("/api/catalog/crawl", { method: "POST" });
    const data = await res.json();
    alert(`Crawl finished: Discovered ${data.total_discovered || 0} new VOD(s).`);
    loadCatalog();
    loadOverview();
  } catch (err) {
    alert("Error triggering channel crawl.");
  }
}

async function analyzeVod(vodId) {
  try {
    const res = await fetch(`/api/catalog/analyze/${vodId}`, { method: "POST" });
    const data = await res.json();
    alert(`Analysis: ${data.message}`);
    loadCatalog();
  } catch (err) {
    alert("Error scheduling VOD analysis.");
  }
}

// --- View 2: Multimodal Scrubber & Player ---
let currentTimeline = null;
let currentChat = [];

async function loadVodTimeline(vodId) {
  try {
    const res = await fetch(`/api/vods/${vodId}/timeline`);
    if (!res.ok) return;
    currentTimeline = await res.json();
    renderScrubber(currentTimeline);
    loadVodChat(vodId);
  } catch (err) {
    console.error("Error loading timeline", err);
  }
}

async function loadVodChat(vodId) {
  try {
    const res = await fetch(`/api/vods/${vodId}/chat?limit=300`);
    if (!res.ok) return;
    const data = await res.json();
    currentChat = data.messages || [];
    renderChatWaterfall(currentChat);
  } catch (err) {
    console.error("Error loading chat", err);
  }
}

function renderScrubber(timeline) {
  const duration = timeline.duration_sec || 60;
  const track = document.getElementById("timeline-markers-track");
  if (!track) return;

  track.innerHTML = "";
  // Render bursts
  (timeline.chat_bursts || []).forEach((b) => {
    const marker = document.createElement("div");
    marker.className = "timeline-marker marker-burst";
    const left = ((b.start_sec || 0) / duration) * 100;
    const width = Math.max(2, (((b.end_sec || b.start_sec + 2) - (b.start_sec || 0)) / duration) * 100);
    marker.style.left = `${left}%`;
    marker.style.width = `${width}%`;
    marker.title = `Meme Burst: ${b.dominant_term || "Burst"}`;
    track.appendChild(marker);
  });

  // Render sponsors
  (timeline.sponsors || []).forEach((s) => {
    const marker = document.createElement("div");
    marker.className = "timeline-marker marker-sponsor";
    const left = ((s.start_sec || 0) / duration) * 100;
    const width = Math.max(3, (((s.end_sec || 0) - (s.start_sec || 0)) / duration) * 100);
    marker.style.left = `${left}%`;
    marker.style.width = `${width}%`;
    marker.title = `Sponsor: ${s.brand_name || "Brand"}`;
    track.appendChild(marker);
  });

  // Render claims
  (timeline.claims || []).forEach((c) => {
    const marker = document.createElement("div");
    marker.className = "timeline-marker marker-claim";
    const left = ((c.timestamp_offset || 0) / duration) * 100;
    marker.style.left = `${left}%`;
    marker.style.width = `4px`;
    marker.title = `Claim: ${c.claim_text || "Claim"}`;
    track.appendChild(marker);
  });

  // Render audio waveform canvas
  const canvas = document.getElementById("waveform-canvas");
  if (canvas && timeline.waveform && timeline.waveform.length > 0) {
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = "#6366f1";
    const barWidth = canvas.width / timeline.waveform.length;
    timeline.waveform.forEach((val, i) => {
      const h = val * canvas.height;
      const y = (canvas.height - h) / 2;
      ctx.fillRect(i * barWidth, y, barWidth - 1, h);
    });
  }
}

function renderChatWaterfall(messages) {
  const container = document.getElementById("player-chat-messages");
  if (!container) return;

  if (messages.length === 0) {
    container.innerHTML = `<div style="color:var(--text-muted); text-align:center; padding:20px;">No chat replay messages available.</div>`;
    return;
  }

  container.innerHTML = messages
    .slice(0, 50)
    .map(
      (m) => `
    <div class="chat-message-row">
      <span class="chat-time">${Math.round(m.timestamp_offset || 0)}s</span>
      <span class="chat-author" style="color: ${m.color || '#38bdf8'}">${m.author_name || 'chatter'}:</span>
      <span>${m.content || ''}</span>
    </div>
  `
    )
    .join("");
}

// Scrubber Click seek
function seekTimeline(event) {
  const track = document.getElementById("scrubber-interactive-track");
  if (!track || !currentTimeline) return;
  const rect = track.getBoundingClientRect();
  const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
  const seekSec = ratio * (currentTimeline.duration_sec || 60);

  const playhead = document.getElementById("timeline-playhead");
  if (playhead) playhead.style.left = `${ratio * 100}%`;
  document.getElementById("current-time-display").textContent = `${Math.round(seekSec)}s`;

  // Filter chat around seek position
  const filtered = currentChat.filter((m) => Math.abs((m.timestamp_offset || 0) - seekSec) < 15);
  if (filtered.length > 0) {
    renderChatWaterfall(filtered);
  }
}

// --- View 3: Autonomous Short Studio ---
async function loadShorts() {
  try {
    const res = await fetch("/api/shorts");
    if (!res.ok) return;
    const candidates = await res.json();
    const select = document.getElementById("short-candidate-select");
    if (!select) return;

    if (candidates.length === 0) {
      select.innerHTML = `<option value="candidate_demo">Sample: Unbelievable Broadcast Moment (92.5/100)</option>`;
      inspectShort("candidate_demo");
      return;
    }

    select.innerHTML = candidates.map((c) => `<option value="${c.candidate_id}">${c.title} (${Math.round(c.virality_score)}/100)</option>`).join("");
    if (candidates.length > 0) {
      inspectShort(candidates[0].candidate_id);
    }
  } catch (err) {
    console.error("Error loading shorts", err);
  }
}

async function inspectShort(candidateId) {
  try {
    const res = await fetch(`/api/shorts/${candidateId}`);
    if (!res.ok) return;
    const c = await res.json();

    document.getElementById("short-virality-score").textContent = `${c.virality_score || 92.5} / 100`;
    document.getElementById("short-hook-score").textContent = Math.round((c.hook_efficacy || 0.9) * 100) + "%";
    document.getElementById("short-meme-score").textContent = Math.round((c.meme_density || 0.85) * 100) + "%";
    document.getElementById("short-retention-score").textContent = Math.round((c.retention_prediction || 0.88) * 100) + "%";

    const script = c.script || {};
    document.getElementById("script-hook-text").textContent = script.hook || "Wait, did he actually just say that?!";
    document.getElementById("script-build-text").textContent = script.build || "Look at chat reaction moving at lightspeed.";
    document.getElementById("script-punchline-text").textContent = script.punchline || "Absolute peak cinema.";
  } catch (err) {
    console.error("Error inspecting short", err);
  }
}

async function renderShortCandidate() {
  const select = document.getElementById("short-candidate-select");
  const candidateId = select ? select.value : "candidate_demo";

  const payload = {
    candidate_id: candidateId,
    vertical_width: 1080,
    vertical_height: 1920,
    subtitle_style: "DYNAMIC_WORD_HIGHLIGHT",
    reaction_layout: "SPLIT_CAM_GAME",
    render_full_video: true,
    auto_publish: false,
  };

  try {
    const res = await fetch(`/api/shorts/${candidateId}/render`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    alert(`Render: ${data.message}`);
  } catch (err) {
    alert("Error rendering short.");
  }
}

// --- View 4: Knowledge & Stance Explorer ---
async function loadKnowledge() {
  try {
    const [claimsRes, stancesRes, sponsorsRes] = await Promise.all([
      fetch("/api/knowledge/claims"),
      fetch("/api/knowledge/stances"),
      fetch("/api/knowledge/sponsors"),
    ]);

    if (claimsRes.ok) {
      const claims = await claimsRes.json();
      const tbody = document.getElementById("claims-table-body");
      if (tbody) {
        tbody.innerHTML = claims
          .map((c) => {
            let badgeColor = "badge-blue";
            if (c.verdict === "TRUE") badgeColor = "badge-green";
            else if (c.verdict === "FALSE") badgeColor = "badge-red";
            else if (c.verdict === "MIXED") badgeColor = "badge-yellow";

            const source = (c.grounding_sources && c.grounding_sources[0]) || {};
            return `
            <tr>
              <td><strong>${c.claim_text}</strong></td>
              <td>${c.streamer_id}</td>
              <td><span class="badge ${badgeColor}">${c.verdict}</span></td>
              <td>${Math.round((c.confidence_score || 0.9) * 100)}%</td>
              <td><a href="${source.url || '#'}" target="_blank" style="color:var(--accent-cyan); text-decoration:none;">${source.title || 'Source'}</a></td>
            </tr>
          `;
          })
          .join("");
      }
    }

    if (stancesRes.ok) {
      const stances = await stancesRes.json();
      const container = document.getElementById("stances-cards-container");
      if (container) {
        container.innerHTML = stances
          .map((s) => {
            const isPositive = (s.net_polarity || 0) >= 0;
            return `
            <div class="card">
              <div class="card-title">
                <span>${s.entity}</span>
                <span class="badge ${isPositive ? 'badge-green' : 'badge-red'}">${s.stance_label}</span>
              </div>
              <div class="stat-value" style="color:${isPositive ? 'var(--accent-green)' : 'var(--accent-red)'}">
                ${s.net_polarity > 0 ? '+' : ''}${s.net_polarity}
              </div>
              <div class="stat-sub">${s.observation_count} observations • ${s.key_topics ? s.key_topics.join(', ') : ''}</div>
            </div>
          `;
          })
          .join("");
      }
    }
  } catch (err) {
    console.error("Error loading knowledge", err);
  }
}

// --- View 5: Live Tail Monitor ---
async function loadLiveStatus() {
  try {
    const res = await fetch("/api/live/status");
    if (!res.ok) return;
    const data = await res.json();
    document.getElementById("live-active-count").textContent = data.active_streams_count || 0;
  } catch (err) {
    console.error("Error loading live status", err);
  }
}

function appendLiveChatMessage(msg) {
  const container = document.getElementById("live-chat-waterfall");
  if (!container) return;

  const row = document.createElement("div");
  row.className = "chat-message-row";
  row.innerHTML = `
    <span class="chat-time">${new Date().toLocaleTimeString()}</span>
    <span class="chat-author" style="color: ${msg.color || '#a855f7'}">${msg.author_name || 'viewer'}:</span>
    <span>${msg.content || ''}</span>
  `;
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function recordLiveBurst(burst) {
  const meter = document.getElementById("live-burst-meter");
  if (meter) {
    meter.style.width = "100%";
    setTimeout(() => {
      meter.style.width = "20%";
    }, 2500);
  }
  const termSpan = document.getElementById("live-burst-term");
  if (termSpan) {
    termSpan.textContent = `Burst: ${burst.dominant_term || 'POG'} (${burst.burst_type || 'BURST'})`;
  }
}
