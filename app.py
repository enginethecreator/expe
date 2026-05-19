"""
music_app.py — YouTube Music FastAPI server + embedded UI
==========================================================

Endpoints:
  GET  /                          — serves the UI (single HTML page)
  GET  /search?q=&limit=          — search YouTube Music, returns track list
  GET  /info?url=                 — full metadata for a track or album/playlist
  POST /download                  — download single track as m4a/mp3
  POST /download/album            — download all tracks in an album/playlist
  GET  /file?path=                — serve a downloaded file
  GET  /jobs/{job_id}             — poll album download job progress

URL patterns yt-dlp handles for YouTube Music:
  Single track : https://music.youtube.com/watch?v=VIDEO_ID
  Album        : https://music.youtube.com/playlist?list=PLAYLIST_ID
  Search       : https://music.youtube.com/search?q=QUERY   (flat-playlist extract)

Audio format strategy:
  bestaudio[ext=m4a]/bestaudio  → FFmpegExtractAudio → m4a or mp3
  m4a is preferred: it preserves AAC without re-encoding, keeps quality intact.
  mp3 always re-encodes — useful for compatibility but lossy.

Metadata embedding:
  EmbedThumbnail + FFmpegMetadata postprocessors write album art, title,
  artist, album, track number into the output file via mutagen/ffmpeg.
"""

import os
import re
import uuid
import asyncio
import shutil
import functools
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel

# ── Setup ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="yt-music", version="1.0.0")

DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "./music"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
COOKIES_FILE = Path(os.environ.get("COOKIES_FILE", "./cookies.txt"))

executor = ThreadPoolExecutor(max_workers=4)

# In-memory job store for album downloads
# { job_id: { status, total, done, tracks: [...], error? } }
jobs: dict[str, dict] = {}

# ── Models ─────────────────────────────────────────────────────────────────────

class DownloadRequest(BaseModel):
    url: str
    fmt: Optional[str] = "m4a"      # m4a | mp3

class AlbumDownloadRequest(BaseModel):
    url: str
    fmt: Optional[str] = "m4a"

# ── Helpers ────────────────────────────────────────────────────────────────────

BASE_OPTS = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "socket_timeout": 30,
    "retries": 3,
    "nocheckcertificate": True,
    "cookiefile": str(COOKIES_FILE) if COOKIES_FILE.exists() else None,
    "extractor_args": {
        "youtube": {
            "player_client": ["default"],
        },
    },
}

def _human_bytes(b: int | None) -> str:
    if not b:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f} {unit}"
        b /= 1024
    return f"{b:.1f} TB"


def _audio_opts(fmt: str, output_template: str) -> dict:
    """
    Build yt-dlp opts for audio extraction.
    m4a: bestaudio[ext=m4a] → no re-encode, best quality
    mp3: bestaudio → FFmpegExtractAudio re-encode to mp3
    """
    preferred = fmt if fmt in ("m4a", "mp3") else "m4a"

    opts = {
        **BASE_OPTS,
        "noplaylist": True,
        "format": "bestaudio[ext=m4a]/bestaudio",
        "outtmpl": output_template,
        "writethumbnail": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": preferred,
                "preferredquality": "0",        # best quality VBR
            },
            {
                "key": "EmbedThumbnail",        # embed album art
            },
            {
                "key": "FFmpegMetadata",        # embed title/artist/album
                "add_metadata": True,
            },
        ],
    }
    return opts


# ── Worker functions ───────────────────────────────────────────────────────────

def _search(query: str, limit: int = 10) -> list[dict]:
    """
    Search YouTube Music using the music.youtube.com/search extractor.
    Returns flat list of track dicts with id, title, uploader, duration, thumbnail.
    """
    search_url = f"https://music.youtube.com/search?q={query}"

    opts = {
        **BASE_OPTS,
        "quiet": True,
        "extract_flat": "in_playlist",  # flat-playlist: don't fetch each video
        "skip_download": True,
        "playlistend": limit,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(search_url, download=False)
        info = ydl.sanitize_info(info)

    entries = info.get("entries") or []
    results = []

    for e in entries[:limit]:
        if not e:
            continue
        vid_id = e.get("id") or e.get("url", "").split("v=")[-1]
        results.append({
            "id":         vid_id,
            "title":      e.get("title") or e.get("ie_key"),
            "uploader":   e.get("uploader") or e.get("channel"),
            "duration":   e.get("duration"),
            "duration_string": e.get("duration_string"),
            "thumbnail":  e.get("thumbnail"),
            "url":        f"https://music.youtube.com/watch?v={vid_id}",
        })

    return results


def _fetch_info(url: str) -> dict:
    """
    Fetch metadata for a single track or an album/playlist.
    Detects type from URL pattern.
    """
    is_playlist = "playlist?list=" in url or ("list=" in url and "watch" not in url)

    opts = {
        **BASE_OPTS,
        "skip_download": True,
        "noplaylist": not is_playlist,
        "extract_flat": "in_playlist" if is_playlist else False,
    }

    with yt_dlp.YoutubeDL(opts) as ydl:
        raw = ydl.extract_info(url, download=False)
        raw = ydl.sanitize_info(raw)

    if is_playlist or raw.get("_type") == "playlist":
        entries = raw.get("entries") or []
        tracks = []
        for i, e in enumerate(entries):
            if not e:
                continue
            vid_id = e.get("id") or ""
            tracks.append({
                "index":    i + 1,
                "id":       vid_id,
                "title":    e.get("title"),
                "uploader": e.get("uploader") or e.get("channel"),
                "duration_string": e.get("duration_string"),
                "url":      f"https://music.youtube.com/watch?v={vid_id}",
            })
        return {
            "type":        "album",
            "id":          raw.get("id"),
            "title":       raw.get("title"),
            "uploader":    raw.get("uploader") or raw.get("channel"),
            "thumbnail":   raw.get("thumbnail"),
            "track_count": len(tracks),
            "tracks":      tracks,
        }
    else:
        filesize = raw.get("filesize") or raw.get("filesize_approx")
        return {
            "type":            "track",
            "id":              raw.get("id"),
            "title":           raw.get("title"),
            "uploader":        raw.get("uploader") or raw.get("channel"),
            "duration":        raw.get("duration"),
            "duration_string": raw.get("duration_string"),
            "thumbnail":       raw.get("thumbnail"),
            "webpage_url":     raw.get("webpage_url"),
            "filesize_human":  _human_bytes(filesize),
            "upload_date":     raw.get("upload_date"),
        }


def _download_track(url: str, fmt: str) -> dict:
    """Download a single track. Returns { filename, file_path, title, filesize_human }."""
    uid = os.urandom(4).hex()
    output_template = str(DOWNLOADS_DIR / f"%(artist)s - %(title)s [{uid}].%(ext)s")
    # Fallback outtmpl if no artist tag: just use title
    # yt-dlp fills %(artist)s from metadata; if absent it leaves the literal string,
    # so we use a sanitize hook instead via the outtmpl fallback syntax:
    output_template = str(DOWNLOADS_DIR / f"%(artist,uploader)s - %(title)s [{uid}].%(ext)s")

    final_path: dict[str, str | None] = {"value": None}

    def progress_hook(d: dict):
        if d["status"] == "finished":
            final_path["value"] = d.get("filename") or d.get("info_dict", {}).get("filepath")

    opts = _audio_opts(fmt, output_template)
    opts["progress_hooks"] = [progress_hook]

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info = ydl.sanitize_info(info)

    # Resolve final file — postprocessors change the extension
    if not final_path["value"] or not Path(final_path["value"]).exists():
        matches = [
            f for f in DOWNLOADS_DIR.glob(f"*{uid}*")
            if f.suffix not in (".part", ".ytdl", ".tmp", ".jpg", ".png", ".webp")
        ]
        if matches:
            final_path["value"] = str(matches[0])

    if not final_path["value"] or not Path(final_path["value"]).exists():
        raise FileNotFoundError("Downloaded file not found on disk")

    file_path = Path(final_path["value"])
    return {
        "title":         info.get("title"),
        "artist":        info.get("artist") or info.get("uploader"),
        "filename":      file_path.name,
        "file_path":     str(file_path),
        "ext":           file_path.suffix.lstrip("."),
        "filesize_human": _human_bytes(file_path.stat().st_size),
    }


def _download_album_worker(job_id: str, url: str, fmt: str):
    """
    Background thread: fetches album info then downloads each track.
    Updates jobs[job_id] incrementally so the polling endpoint sees progress.
    """
    try:
        # Step 1 — fetch track list
        jobs[job_id]["status"] = "fetching"
        info = _fetch_info(url)

        if info["type"] != "album":
            # Single track passed to album endpoint — just download it
            jobs[job_id]["total"] = 1
            result = _download_track(url, fmt)
            jobs[job_id]["tracks"].append(result)
            jobs[job_id]["done"] = 1
            jobs[job_id]["status"] = "completed"
            return

        tracks = info["tracks"]
        jobs[job_id]["total"] = len(tracks)
        jobs[job_id]["album_title"] = info["title"]
        jobs[job_id]["status"] = "downloading"

        # Step 2 — download each track sequentially
        for track in tracks:
            try:
                result = _download_track(track["url"], fmt)
                jobs[job_id]["tracks"].append({**result, "index": track["index"]})
            except Exception as e:
                jobs[job_id]["tracks"].append({
                    "index":  track["index"],
                    "title":  track["title"],
                    "error":  str(e),
                })
            jobs[job_id]["done"] += 1

        jobs[job_id]["status"] = "completed"

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def ui():
    return HTMLResponse(content=_HTML)


@app.get("/search")
async def search(q: str, limit: int = 10):
    if not q:
        raise HTTPException(400, "Missing q param")
    if limit > 25:
        limit = 25
    try:
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(executor, functools.partial(_search, q, limit))
        return {"success": True, "query": q, "results": results}
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/info")
async def info(url: str):
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, _fetch_info, url)
        return {"success": True, "data": data}
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/download")
async def download_track(req: DownloadRequest):
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            functools.partial(_download_track, req.url, req.fmt or "m4a")
        )
        return {
            "success": True,
            "data": {
                **result,
                "fetch_url": f"/file?path={result['filename']}",
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/download/album")
async def download_album(req: AlbumDownloadRequest):
    """
    Spawns a background job, returns job_id immediately.
    Poll /jobs/{job_id} for progress.
    """
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status":      "starting",
        "total":       0,
        "done":        0,
        "tracks":      [],
        "album_title": None,
        "error":       None,
    }
    loop = asyncio.get_event_loop()
    loop.run_in_executor(
        executor,
        functools.partial(_download_album_worker, job_id, req.url, req.fmt or "m4a")
    )
    return {"success": True, "job_id": job_id}


@app.get("/jobs/{job_id}")
async def job_status(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return {"success": True, "data": job}


@app.get("/file")
async def serve_file(path: str):
    file_path = DOWNLOADS_DIR / path
    if not file_path.exists():
        raise HTTPException(404, f"File not found: {path}")
    if not str(file_path.resolve()).startswith(str(DOWNLOADS_DIR.resolve())):
        raise HTTPException(403, "Access denied")
    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


# ── Embedded UI ────────────────────────────────────────────────────────────────

_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>YT Music</title>
<link href="https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,300;0,400;0,500;1,300&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:           #0a0a0b;
    --surface:      #111113;
    --surface2:     #18181c;
    --border:       #222228;
    --border2:      #2e2e38;
    --accent:       #c8f542;
    --accent-dim:   rgba(200,245,66,0.12);
    --accent-glow:  rgba(200,245,66,0.06);
    --text:         #e8e8ec;
    --text-2:       #a0a0b0;
    --text-3:       #585868;
    --danger:       #f04444;
    --success:      #42d18e;
    --warning:      #f5a623;
    --mono:         "DM Mono", monospace;
    --sans:         "DM Sans", sans-serif;
  }

  html, body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    font-size: 14px;
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* ── Layout ── */
  .app {
    max-width: 860px;
    margin: 0 auto;
    padding: 48px 24px 80px;
  }

  /* ── Header ── */
  .header {
    margin-bottom: 48px;
  }
  .header-eyebrow {
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.2em;
    color: var(--accent);
    text-transform: uppercase;
    margin-bottom: 10px;
  }
  .header h1 {
    font-size: 28px;
    font-weight: 300;
    letter-spacing: -0.02em;
    color: var(--text);
    line-height: 1.2;
  }
  .header h1 span { color: var(--accent); }

  /* ── Tabs ── */
  .tabs {
    display: flex;
    gap: 2px;
    border-bottom: 1px solid var(--border);
    margin-bottom: 32px;
  }
  .tab {
    background: none;
    border: none;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.12em;
    color: var(--text-3);
    padding: 10px 16px;
    cursor: pointer;
    border-bottom: 2px solid transparent;
    margin-bottom: -1px;
    transition: color 0.15s, border-color 0.15s;
  }
  .tab:hover { color: var(--text-2); }
  .tab.active { color: var(--accent); border-bottom-color: var(--accent); }

  /* ── Panel ── */
  .panel { display: none; }
  .panel.active { display: block; }

  /* ── Form elements ── */
  .field { margin-bottom: 16px; }
  .field label {
    display: block;
    font-family: var(--mono);
    font-size: 10px;
    letter-spacing: 0.14em;
    color: var(--text-3);
    text-transform: uppercase;
    margin-bottom: 7px;
  }
  input[type="text"], select {
    width: 100%;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 3px;
    color: var(--text);
    font-family: var(--mono);
    font-size: 13px;
    padding: 10px 14px;
    outline: none;
    transition: border-color 0.15s;
  }
  input[type="text"]:focus, select:focus { border-color: var(--accent); }
  select { cursor: pointer; }
  select option { background: var(--surface2); }

  /* ── Buttons ── */
  .btn {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    padding: 10px 20px;
    border-radius: 3px;
    border: none;
    font-family: var(--mono);
    font-size: 11px;
    letter-spacing: 0.1em;
    font-weight: 500;
    cursor: pointer;
    transition: opacity 0.15s, background 0.15s;
    white-space: nowrap;
  }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-primary { background: var(--accent); color: #0a0a0b; }
  .btn-primary:not(:disabled):hover { opacity: 0.88; }
  .btn-ghost {
    background: transparent;
    border: 1px solid var(--border2);
    color: var(--text-2);
  }
  .btn-ghost:not(:disabled):hover { border-color: var(--accent); color: var(--accent); }
  .btn-success { background: var(--success); color: #0a0a0b; }
  .btn-danger-ghost {
    background: transparent;
    border: 1px solid var(--border2);
    color: var(--danger);
  }

  .row { display: flex; gap: 10px; align-items: flex-end; }
  .row .field { flex: 1; margin-bottom: 0; }

  /* ── Status / messages ── */
  .status {
    font-family: var(--mono);
    font-size: 11px;
    padding: 10px 14px;
    border-radius: 3px;
    margin-top: 16px;
    display: none;
  }
  .status.visible { display: block; }
  .status.loading { background: var(--accent-glow); border: 1px solid var(--border2); color: var(--accent); }
  .status.error   { background: rgba(240,68,68,0.08); border: 1px solid rgba(240,68,68,0.3); color: var(--danger); }
  .status.success { background: rgba(66,209,142,0.08); border: 1px solid rgba(66,209,142,0.3); color: var(--success); }

  /* ── Search results ── */
  .results { margin-top: 24px; display: flex; flex-direction: column; gap: 2px; }
  .result-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 10px 12px;
    border-radius: 3px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: background 0.1s, border-color 0.1s;
  }
  .result-item:hover { background: var(--surface); border-color: var(--border); }
  .result-thumb {
    width: 44px;
    height: 44px;
    object-fit: cover;
    border-radius: 2px;
    flex-shrink: 0;
    background: var(--surface2);
  }
  .result-thumb-placeholder {
    width: 44px;
    height: 44px;
    border-radius: 2px;
    background: var(--surface2);
    flex-shrink: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--text-3);
    font-size: 18px;
  }
  .result-meta { flex: 1; min-width: 0; }
  .result-title {
    font-size: 13px;
    font-weight: 500;
    color: var(--text);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .result-sub {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-3);
    margin-top: 3px;
  }
  .result-actions { display: flex; gap: 6px; flex-shrink: 0; }

  /* ── Track card (single download result) ── */
  .track-card {
    margin-top: 20px;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 4px;
    padding: 18px 20px;
    display: none;
  }
  .track-card.visible { display: flex; gap: 16px; align-items: flex-start; }
  .track-card img {
    width: 72px;
    height: 72px;
    object-fit: cover;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .track-card-info { flex: 1; min-width: 0; }
  .track-card-title {
    font-size: 14px;
    font-weight: 500;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .track-card-sub {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-3);
    margin-bottom: 12px;
  }
  .track-card-actions { display: flex; gap: 8px; flex-wrap: wrap; }

  /* ── Album track list ── */
  .album-header {
    display: flex;
    align-items: center;
    gap: 16px;
    margin-bottom: 20px;
    padding: 16px;
    background: var(--surface);
    border: 1px solid var(--border2);
    border-radius: 4px;
  }
  .album-thumb {
    width: 80px;
    height: 80px;
    object-fit: cover;
    border-radius: 2px;
    flex-shrink: 0;
  }
  .album-info-title { font-size: 16px; font-weight: 500; margin-bottom: 4px; }
  .album-info-sub {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-3);
    margin-bottom: 10px;
  }

  .track-list { display: flex; flex-direction: column; gap: 2px; margin-bottom: 16px; }
  .track-row {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 10px;
    border-radius: 3px;
    border: 1px solid transparent;
    transition: background 0.1s;
  }
  .track-row:hover { background: var(--surface); }
  .track-num {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-3);
    width: 20px;
    text-align: right;
    flex-shrink: 0;
  }
  .track-row-title { flex: 1; font-size: 13px; }
  .track-row-dur {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-3);
    flex-shrink: 0;
  }

  /* ── Job progress ── */
  .job-progress { margin-top: 20px; display: none; }
  .job-progress.visible { display: block; }
  .progress-bar-wrap {
    background: var(--surface2);
    border-radius: 2px;
    height: 4px;
    margin: 12px 0;
    overflow: hidden;
  }
  .progress-bar {
    height: 4px;
    background: var(--accent);
    border-radius: 2px;
    transition: width 0.4s ease;
  }
  .progress-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-3);
    display: flex;
    justify-content: space-between;
  }
  .job-tracks { margin-top: 12px; display: flex; flex-direction: column; gap: 4px; }
  .job-track-item {
    font-family: var(--mono);
    font-size: 11px;
    padding: 6px 10px;
    border-radius: 2px;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .job-track-item.done    { color: var(--success); background: rgba(66,209,142,0.06); }
  .job-track-item.pending { color: var(--text-3); }
  .job-track-item.error   { color: var(--danger); background: rgba(240,68,68,0.06); }
  .job-track-icon { flex-shrink: 0; }

  /* ── Mono label ── */
  .mono-label {
    font-family: var(--mono);
    font-size: 10px;
    color: var(--text-3);
    letter-spacing: 0.12em;
  }

  /* ── Divider ── */
  .divider {
    border: none;
    border-top: 1px solid var(--border);
    margin: 28px 0;
  }
</style>
</head>
<body>
<div class="app">

  <header class="header">
    <div class="header-eyebrow">yt-dlp · music</div>
    <h1>YouTube<span> Music</span><br>Downloader</h1>
  </header>

  <!-- Tabs -->
  <div class="tabs">
    <button class="tab active" onclick="switchTab('search')">SEARCH</button>
    <button class="tab" onclick="switchTab('single')">SINGLE TRACK</button>
    <button class="tab" onclick="switchTab('album')">ALBUM / PLAYLIST</button>
  </div>

  <!-- ── SEARCH PANEL ── -->
  <div class="panel active" id="panel-search">
    <div class="row">
      <div class="field">
        <label>Search YouTube Music</label>
        <input type="text" id="search-query" placeholder="artist, song title, album…" onkeydown="if(event.key==='Enter') doSearch()"/>
      </div>
      <div class="field" style="width:90px">
        <label>Results</label>
        <select id="search-limit">
          <option value="5">5</option>
          <option value="10" selected>10</option>
          <option value="20">20</option>
        </select>
      </div>
      <button class="btn btn-primary" onclick="doSearch()" id="search-btn">SEARCH →</button>
    </div>
    <div class="status" id="search-status"></div>
    <div class="results" id="search-results"></div>
  </div>

  <!-- ── SINGLE TRACK PANEL ── -->
  <div class="panel" id="panel-single">
    <div class="field">
      <label>YouTube Music URL</label>
      <input type="text" id="single-url" placeholder="https://music.youtube.com/watch?v=…"/>
    </div>
    <div class="row" style="margin-bottom:16px">
      <div class="field" style="width:120px">
        <label>Format</label>
        <select id="single-fmt">
          <option value="m4a" selected>M4A (best quality)</option>
          <option value="mp3">MP3 (compatible)</option>
        </select>
      </div>
      <button class="btn btn-primary" onclick="doSingleInfo()" id="single-info-btn">FETCH INFO →</button>
    </div>

    <div class="status" id="single-status"></div>

    <div class="track-card" id="single-card">
      <img id="single-thumb" src="" alt=""/>
      <div class="track-card-info">
        <div class="track-card-title" id="single-title"></div>
        <div class="track-card-sub" id="single-sub"></div>
        <div class="track-card-actions">
          <button class="btn btn-primary" onclick="doSingleDownload()" id="single-dl-btn">↓ DOWNLOAD</button>
          <a class="btn btn-success" id="single-save-btn" style="display:none;text-decoration:none">↓ SAVE FILE</a>
        </div>
      </div>
    </div>
  </div>

  <!-- ── ALBUM PANEL ── -->
  <div class="panel" id="panel-album">
    <div class="field">
      <label>Album / Playlist URL</label>
      <input type="text" id="album-url" placeholder="https://music.youtube.com/playlist?list=…"/>
    </div>
    <div class="row" style="margin-bottom:16px">
      <div class="field" style="width:120px">
        <label>Format</label>
        <select id="album-fmt">
          <option value="m4a" selected>M4A</option>
          <option value="mp3">MP3</option>
        </select>
      </div>
      <button class="btn btn-ghost" onclick="doAlbumInfo()" id="album-info-btn">PREVIEW →</button>
      <button class="btn btn-primary" onclick="doAlbumDownload()" id="album-dl-btn" disabled>↓ DOWNLOAD ALL</button>
    </div>

    <div class="status" id="album-status"></div>

    <!-- Album header preview -->
    <div id="album-header" style="display:none" class="album-header">
      <img class="album-thumb" id="album-thumb" src="" alt=""/>
      <div>
        <div class="album-info-title" id="album-title"></div>
        <div class="album-info-sub" id="album-sub"></div>
      </div>
    </div>
    <div class="track-list" id="album-tracklist"></div>

    <!-- Job progress -->
    <div class="job-progress" id="job-progress">
      <div class="progress-label">
        <span id="job-label">Downloading…</span>
        <span id="job-count">0 / 0</span>
      </div>
      <div class="progress-bar-wrap">
        <div class="progress-bar" id="progress-bar" style="width:0%"></div>
      </div>
      <div class="job-tracks" id="job-tracks"></div>
    </div>
  </div>

</div>

<script>
// ── Tab switching ──────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  event.target.classList.add('active');
  document.getElementById('panel-' + name).classList.add('active');
}

// ── Status helpers ─────────────────────────────────────────────────────────────
function setStatus(id, type, msg) {
  const el = document.getElementById(id);
  el.className = 'status visible ' + type;
  el.textContent = msg;
}
function clearStatus(id) {
  const el = document.getElementById(id);
  el.className = 'status';
  el.textContent = '';
}
function setBtn(id, disabled, text) {
  const btn = document.getElementById(id);
  btn.disabled = disabled;
  if (text) btn.textContent = text;
}

// ── Duration formatter ─────────────────────────────────────────────────────────
function fmtDur(secs) {
  if (!secs) return '';
  const m = Math.floor(secs / 60);
  const s = String(Math.floor(secs % 60)).padStart(2, '0');
  return m + ':' + s;
}

// ── SEARCH ────────────────────────────────────────────────────────────────────
let searchResults = [];

async function doSearch() {
  const q = document.getElementById('search-query').value.trim();
  const limit = document.getElementById('search-limit').value;
  if (!q) return;

  setBtn('search-btn', true, 'SEARCHING…');
  setStatus('search-status', 'loading', '⟳ Searching YouTube Music…');
  document.getElementById('search-results').innerHTML = '';

  try {
    const res = await fetch('/search?q=' + encodeURIComponent(q) + '&limit=' + limit);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || data.error || 'Search failed');

    searchResults = data.results;
    clearStatus('search-status');
    renderSearchResults(data.results);
  } catch (e) {
    setStatus('search-status', 'error', '✕ ' + e.message);
  } finally {
    setBtn('search-btn', false, 'SEARCH →');
  }
}

function renderSearchResults(results) {
  const el = document.getElementById('search-results');
  if (!results.length) {
    el.innerHTML = '<div style="color:var(--text-3);font-family:var(--mono);font-size:11px;padding:16px 0">No results found.</div>';
    return;
  }
  el.innerHTML = results.map((r, i) => `
    <div class="result-item">
      ${r.thumbnail
        ? `<img class="result-thumb" src="${r.thumbnail}" alt="" onerror="this.style.display='none'"/>`
        : `<div class="result-thumb-placeholder">♪</div>`}
      <div class="result-meta">
        <div class="result-title">${escHtml(r.title || '—')}</div>
        <div class="result-sub">${escHtml(r.uploader || '')}${r.duration ? ' · ' + fmtDur(r.duration) : ''}</div>
      </div>
      <div class="result-actions">
        <button class="btn btn-ghost" style="font-size:10px;padding:6px 12px" onclick="downloadFromSearch(${i}, 'm4a')">M4A</button>
        <button class="btn btn-ghost" style="font-size:10px;padding:6px 12px" onclick="downloadFromSearch(${i}, 'mp3')">MP3</button>
      </div>
    </div>
  `).join('');
}

// Inline download triggered from search result
const searchDownloadState = {};

async function downloadFromSearch(idx, fmt) {
  const track = searchResults[idx];
  if (!track) return;

  const key = idx + '_' + fmt;
  if (searchDownloadState[key] === 'loading') return;
  searchDownloadState[key] = 'loading';

  setStatus('search-status', 'loading', `⟳ Downloading "${track.title}" as ${fmt.toUpperCase()}…`);

  try {
    const res = await fetch('/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: track.url, fmt }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || data.error || 'Download failed');

    setStatus('search-status', 'success',
      `✓ "${data.data.title}" ready — click to save`);

    // Inject save link into the result row
    const rows = document.querySelectorAll('.result-item');
    if (rows[idx]) {
      const actions = rows[idx].querySelector('.result-actions');
      const a = document.createElement('a');
      a.href = '/file?path=' + encodeURIComponent(data.data.filename);
      a.className = 'btn btn-success';
      a.style = 'font-size:10px;padding:6px 12px;text-decoration:none';
      a.textContent = '↓ SAVE';
      actions.appendChild(a);
    }
  } catch (e) {
    setStatus('search-status', 'error', '✕ ' + e.message);
  } finally {
    searchDownloadState[key] = 'idle';
  }
}

// ── SINGLE TRACK ──────────────────────────────────────────────────────────────
let singleInfo = null;

async function doSingleInfo() {
  const url = document.getElementById('single-url').value.trim();
  if (!url) return;

  setBtn('single-info-btn', true, 'FETCHING…');
  setStatus('single-status', 'loading', '⟳ Fetching track info…');
  document.getElementById('single-card').classList.remove('visible');
  document.getElementById('single-save-btn').style.display = 'none';

  try {
    const res = await fetch('/info?url=' + encodeURIComponent(url));
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || 'Failed to fetch info');
    singleInfo = data.data;
    clearStatus('single-status');
    renderSingleCard(data.data);
  } catch (e) {
    setStatus('single-status', 'error', '✕ ' + e.message);
  } finally {
    setBtn('single-info-btn', false, 'FETCH INFO →');
  }
}

function renderSingleCard(info) {
  document.getElementById('single-thumb').src = info.thumbnail || '';
  document.getElementById('single-title').textContent = info.title || '—';
  document.getElementById('single-sub').textContent =
    [info.uploader, info.duration_string, info.filesize_human].filter(Boolean).join(' · ');
  document.getElementById('single-dl-btn').textContent = '↓ DOWNLOAD';
  document.getElementById('single-card').classList.add('visible');
}

async function doSingleDownload() {
  const url = document.getElementById('single-url').value.trim();
  const fmt = document.getElementById('single-fmt').value;
  if (!url) return;

  setBtn('single-dl-btn', true, '⟳ DOWNLOADING…');
  setStatus('single-status', 'loading', '⟳ Downloading audio…');

  try {
    const res = await fetch('/download', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, fmt }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || data.error || 'Download failed');

    setStatus('single-status', 'success', `✓ Ready · ${data.data.filesize_human}`);
    const saveBtn = document.getElementById('single-save-btn');
    saveBtn.href = '/file?path=' + encodeURIComponent(data.data.filename);
    saveBtn.style.display = 'inline-flex';
    setBtn('single-dl-btn', false, '↓ DOWNLOAD AGAIN');
  } catch (e) {
    setStatus('single-status', 'error', '✕ ' + e.message);
    setBtn('single-dl-btn', false, '↓ DOWNLOAD');
  }
}

// ── ALBUM ─────────────────────────────────────────────────────────────────────
let albumInfo = null;
let currentJobId = null;
let pollInterval = null;

async function doAlbumInfo() {
  const url = document.getElementById('album-url').value.trim();
  if (!url) return;

  setBtn('album-info-btn', true, 'FETCHING…');
  setStatus('album-status', 'loading', '⟳ Fetching album info…');
  document.getElementById('album-header').style.display = 'none';
  document.getElementById('album-tracklist').innerHTML = '';
  document.getElementById('album-dl-btn').disabled = true;

  try {
    const res = await fetch('/info?url=' + encodeURIComponent(url));
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || 'Failed to fetch album info');
    albumInfo = data.data;
    clearStatus('album-status');
    renderAlbumPreview(data.data);
    document.getElementById('album-dl-btn').disabled = false;
  } catch (e) {
    setStatus('album-status', 'error', '✕ ' + e.message);
  } finally {
    setBtn('album-info-btn', false, 'PREVIEW →');
  }
}

function renderAlbumPreview(info) {
  document.getElementById('album-thumb').src = info.thumbnail || '';
  document.getElementById('album-title').textContent = info.title || '—';
  document.getElementById('album-sub').textContent =
    [info.uploader, info.track_count + ' tracks'].filter(Boolean).join(' · ');
  document.getElementById('album-header').style.display = 'flex';

  const list = document.getElementById('album-tracklist');
  list.innerHTML = (info.tracks || []).map(t => `
    <div class="track-row">
      <span class="track-num">${t.index}</span>
      <span class="track-row-title">${escHtml(t.title || '—')}</span>
      <span class="track-row-dur">${t.duration_string || ''}</span>
    </div>
  `).join('');
}

async function doAlbumDownload() {
  const url = document.getElementById('album-url').value.trim();
  const fmt = document.getElementById('album-fmt').value;
  if (!url) return;

  setBtn('album-dl-btn', true, '⟳ STARTING…');
  setStatus('album-status', 'loading', '⟳ Starting album download job…');

  try {
    const res = await fetch('/download/album', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url, fmt }),
    });
    const data = await res.json();
    if (!res.ok || !data.success) throw new Error(data.detail || 'Failed to start job');

    currentJobId = data.job_id;
    clearStatus('album-status');
    document.getElementById('job-progress').classList.add('visible');
    startPolling(data.job_id);
  } catch (e) {
    setStatus('album-status', 'error', '✕ ' + e.message);
    setBtn('album-dl-btn', false, '↓ DOWNLOAD ALL');
  }
}

function startPolling(jobId) {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(() => pollJob(jobId), 1500);
}

async function pollJob(jobId) {
  try {
    const res = await fetch('/jobs/' + jobId);
    const data = await res.json();
    if (!res.ok || !data.success) return;

    const job = data.data;
    updateJobUI(job);

    if (job.status === 'completed' || job.status === 'error') {
      clearInterval(pollInterval);
      pollInterval = null;
      setBtn('album-dl-btn', false, '↓ DOWNLOAD ALL');
      if (job.status === 'completed') {
        setStatus('album-status', 'success',
          `✓ ${job.done} track${job.done !== 1 ? 's' : ''} downloaded`);
      } else {
        setStatus('album-status', 'error', '✕ ' + (job.error || 'Job failed'));
      }
    }
  } catch (e) { /* network blip — keep polling */ }
}

function updateJobUI(job) {
  const total = job.total || 1;
  const done  = job.done || 0;
  const pct   = total > 0 ? Math.round((done / total) * 100) : 0;

  document.getElementById('progress-bar').style.width = pct + '%';
  document.getElementById('job-count').textContent = done + ' / ' + total;
  document.getElementById('job-label').textContent =
    job.status === 'fetching'   ? 'Fetching track list…' :
    job.status === 'completed'  ? 'Complete' :
    job.status === 'error'      ? 'Error' :
    'Downloading…';

  const tracksEl = document.getElementById('job-tracks');
  tracksEl.innerHTML = job.tracks.map(t => {
    if (t.error) {
      return `<div class="job-track-item error"><span class="job-track-icon">✕</span>${escHtml(t.title || 'Track ' + t.index)} — ${escHtml(t.error)}</div>`;
    }
    return `<div class="job-track-item done">
      <span class="job-track-icon">✓</span>
      <span style="flex:1">${escHtml(t.title || t.filename || '—')}</span>
      <span style="color:var(--text-3);font-size:10px">${t.filesize_human || ''}</span>
      <a href="/file?path=${encodeURIComponent(t.filename)}" class="btn btn-ghost" style="font-size:9px;padding:3px 8px;text-decoration:none">SAVE</a>
    </div>`;
  }).join('');
}

// ── Utility ───────────────────────────────────────────────────────────────────
function escHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}
</script>
</body>
</html>
"""
