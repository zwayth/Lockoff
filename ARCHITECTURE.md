# Why Lockoff and Clip Factory are separate

Three projects were compared before deciding what Lockoff should become:

| Project | What it is | Verdict |
|---|---|---|
| **Clip Factory** (`D:\AI Video Editor\Clip Factory`) | Unattended engine: scrapes reels from four approved accounts, re-encodes to 1080×1920, queues through Buffer, runs every 15 min as a Windows scheduled task | **Leave running, do not touch.** It is already delivering a full day of posts. |
| **PeakCut** (`C:\peakcut`, upstream clone) | Self-hosted web studio: rolling live buffer, per-creator workspaces, activity log, post queue, direct YouTube/Instagram/TikTok upload | **Take the capabilities, not the architecture.** Its direct Instagram route needs a Meta Business app, a public HTTPS address and app review. |
| **Lockoff** (this repo) | Branded standalone editor: same core as Clip Factory (identical `requirements.txt` and clipping pipeline) with its own web UI at `127.0.0.1:8765` | **Become the local PeakCut-equivalent.** |

## The decision

**Do not merge the folders.** Keep Clip Factory as the proven automation
engine and Lockoff as the interactive product. Copying the Instagram/scrape
code into Lockoff would create two schedulers fighting over the same Buffer
slot limit and double-posting the same reels.

**Merge the relevant features into Lockoff instead:**

| Feature | Source | How it was merged |
|---|---|---|
| Rolling live buffer | PeakCut `studio/live.py` | `lockoff/live.py` — same 6-second segment idea, live transcription, "clip that"/trigger/loudness detection, rendered through Lockoff's existing `factory.render_clip`. Local-only; no server, auth, chat IRC or dual AI judges. |
| Creator workspaces | PeakCut `studio/creators.py` | `lockoff/workspaces.py` — named channel + credit + trigger phrases + activity log in one JSON file. |
| Post queue | PeakCut `studio/posting.py` | `lockoff/queue.py` + copied `lockoff/social.py` — uses **Buffer**, not the Instagram Graph API. Respects the 10-post cap. |
| Fresh media URLs | PeakCut's 1-hour signed links | `refresh_due_urls()` — re-upload local MP4s for posts due within 2 h and swap the link with Buffer `editPost`. Same fix proven in Clip Factory. |
| Web dashboard | PeakCut `web/` | Existing `lockoff/ui.html` extended with a **Live & publish** view (start/stop/clip, workspaces, activity, queue). |

**Deliberately left out:**

- PeakCut's FastAPI server, accounts, sign-in, per-user quotas and Docker deploy
- Direct Instagram Graph API upload (needs Meta app review + public HTTPS)
- Twitch IRC chat reader and the two-judge AI decision loop
- Auto-record when a creator goes live (start a session manually for now)

## Isolation rules

These must stay true so the products never interfere:

1. **Separate config** — Clip Factory uses `social.json`; Lockoff uses `lockoff_social.json`.
2. **Separate state** — `insta_state.json` / `insta_media.json` vs `lockoff_queue.json`.
3. **Separate credentials** — same Buffer token may be used, but each product has its own copy and its own channel/org settings.
4. **Separate schedulers** — only Clip Factory has a Windows scheduled task. Lockoff publishes when you press *Queue*.
5. **Separate jobs** — Clip Factory stores renders under its own `Raw/`; Lockoff stores finished clips under `Jobs/` and live sessions under `Jobs/_live/`.
6. **Separate ports** — Lockoff UI is `127.0.0.1:8765`; PeakCut (if ever run) uses its own port.

## Running Lockoff

Lockoff has no `.venv` of its own yet. The code was validated with Clip
Factory's interpreter without importing any Clip Factory module:

```powershell
cd "D:\AI Video Editor\Lockoff"
& "D:\AI Video Editor\Clip Factory\.venv\Scripts\python.exe" -m lockoff doctor
```

For a permanent install, run `setup.bat` in this folder first, then
`Start Lockoff.bat`.
