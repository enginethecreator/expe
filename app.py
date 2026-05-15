# 
"""
app.py — yt-dlp FastAPI server (v5.0.0, >= 2026.03.17 compliant)

Endpoints:
  GET  /                          — health check
  GET  /info?url=                 — full video metadata + all formats
  GET  /formats?url=              — formats only (lightweight)
  GET  /transcript?url=&lang=     — transcript/subtitles as structured JSON
  POST /download                  — download video/audio with robust format selection
  GET  /download/file?path=       — serve a previously downloaded file
  POST /quick                     — zero-config download: yt-dlp picks best available format

Design:
  - All yt-dlp work runs in a ThreadPoolExecutor (yt-dlp is sync/blocking)
    so FastAPI's async event loop is never blocked
  - Downloads are saved to ./downloads/, served back via FileResponse
  - Transcripts are extracted in-memory (no file written to disk)
  - Format selection uses deep fallback chains to survive YouTube's SABR/format
    availability changes (no hardcoded ext constraints unless absolutely required)

YouTube format availability context (2025+):
  - YouTube increasingly forces SABR (Server-Side Adaptive Bitrate) streaming
    on web clients from datacenter IPs, leaving zero video formats via web client.
  - The android/mweb clients still expose direct-URL formats in most cases.
  - Hardcoding ext= filters (e.g. [ext=mp4]) causes "Requested format is not available"
    when YouTube only serves that resolution in vp9/webm.
  - The safest selector strategy: prefer codec+container but fall all the way back
    to yt-dlp's unconstrained default ("b") so the download never hard-fails.
  - /quick uses format="b" (yt-dlp's "best" shorthand) with no constraints at all
    — reliably downloads something on every working IP/session.
"""

import os
import re
import asyncio
import shutil
import json
import urllib.request
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

# ── Setup ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="yt-dlp server", version="2.1.0")

DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "./downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
COOKIES_FILE = Path(os.environ.get("COOKIES_FILE", "./cookies.txt"))
print("FFMPEG PATH:", shutil.which("ffmpeg"))
executor = ThreadPoolExecutor(max_workers=4)


# ── Models ─────────────────────────────────────────────────────────────────────

class DownloadRequest(BaseModel):
    url: str
    format_id: Optional[str] = None      # explicit format_id from /formats
    quality: Optional[str] = "best"      # best | 1080p | 720p | 480p | 360p | audio
    ext: Optional[str] = "mp4"           # mp4 | webm | mkv | m4a | mp3


class QuickDownloadRequest(BaseModel):
    url: str


# ── Helpers ────────────────────────────────────────────────────────────────────

BASE_OPTS = {
    "quiet": False,
    "no_warnings": False,
    "noplaylist": True,
    "socket_timeout": 30,
    "retries": 3,
    "fragment_retries": 3,
    "concurrent_fragment_downloads": 3,
    "nocheckcertificate": True,
    # Required since yt-dlp 2025.11.12: a JS runtime to solve YouTube's n-challenge.
    # node is available in the Railway environment via nixpacks.toml (nodejs_22).
    # Empty dict means "find node in PATH" — no hardcoded path needed.
    "js_runtimes": {"node": {}},
    "extractor_args": {
        "youtube": {
          #  "player_client": ["android", "web"],
            "remote_components": ["ejs:github"]
        },
    },
}

def is_auth_error(exception: Exception) -> bool:
    """Check if the error message indicates a need for cookies."""
    err_msg = str(exception).lower()
    return "sign in" in err_msg or "bot" in err_msg or "confirm your age" in err_msg
  

def classify_format(f: dict) -> dict | None:
    vcodec = f.get("vcodec", "none")
    acodec = f.get("acodec", "none")
    has_video = vcodec not in (None, "none")
    has_audio = acodec not in (None, "none")

    if not has_video and not has_audio:
        return None

    if has_video and has_audio:
        kind = "video+audio"
    elif has_video:
        kind = "video-only"
    else:
        kind = "audio-only"

    filesize = f.get("filesize") or f.get("filesize_approx")

    return {
        "format_id": f.get("format_id"),
        "ext": f.get("ext"),
        "type": kind,
        "resolution": f.get("resolution") or (
            f"{f['height']}p" if f.get("height") else "audio only"
        ),
        "width": f.get("width"),
        "height": f.get("height"),
        "fps": f.get("fps"),
        "vcodec": vcodec if has_video else None,
        "acodec": acodec if has_audio else None,
        "tbr": f.get("tbr"),
        "vbr": f.get("vbr"),
        "abr": f.get("abr"),
        "asr": f.get("asr"),
        "filesize": filesize,
        "filesize_human": _human_bytes(filesize),
        "format_note": f.get("format_note"),
        "protocol": f.get("protocol"),
        "dynamic_range": f.get("dynamic_range"),
    }


def _human_bytes(b: int | None) -> str:
    if not b:
        return "unknown"
    for unit in ["B", "KB", "MB", "GB"]:
        if b < 1024:
            return f"{b:.1f}{unit}"
        b /= 1024
    return f"{b:.1f}TB"


def _resolve_format_selector(
    format_id: str | None,
    quality: str | None,
    ext: str | None,
) -> str:
    """
    Build a yt-dlp format selector string with deep fallback chains.

    The key insight: YouTube increasingly serves video-only streams in vp9/webm
    and may not have mp4 available at every resolution. Constraining to [ext=mp4]
    causes hard failures. Instead we:
      1. Try to get the preferred codec/container combo
      2. Fall back to any codec at that resolution
      3. Fall back to best combined format at any resolution
      4. As last resort, let yt-dlp pick freely ("b")

    This ensures the download ALWAYS works even when YouTube is serving limited
    format sets (SABR enforcement, datacenter IP restrictions, etc.).

    Selector syntax reminder:
      /   = fallback: try left, if not available try right
      *   = "if no audio, merge with best audio" wildcard
      bv  = bestvideo (alias)
      ba  = bestaudio (alias)
      b   = best combined (single file, no merge needed)
    """

    # Explicit format_id takes top priority — caller knows exactly what they want.
    # Still add "+bestaudio/format_id" fallback in case it's video-only.
    if format_id:
        return f"{format_id}+bestaudio/{format_id}"

    q = (quality or "best").lower()
    e = (ext or "mp4").lower()

    # Audio-only path — no video needed, just best audio + optional post-process
    if q == "audio":
        # bestaudio in preferred container, fallback to any audio, fallback to best
        if e == "mp3":
            # mp3 always comes from post-processing bestaudio — selector is simple
            return "bestaudio/best"
        elif e == "m4a":
            return "bestaudio[ext=m4a]/bestaudio/best"
        else:
            return "bestaudio/best"

    # Video path — build height-constrained selectors with generous fallbacks.
    # Pattern:
    #   bv*[height<=H]+ba/bv[height<=H]+ba/b[height<=H]/b
    #
    # bv*  = bestvideo that already has audio, or best video-only (merge if needed)
    # /b   = last resort: best single-file combined stream yt-dlp can find

    height_map = {
        "1080p": 1080,
        "720p": 720,
        "480p": 480,
        "360p": 360,
    }

    if q in height_map:
        h = height_map[q]
        # Tier 1: best video (any codec) at or below height + best audio → merged
        # Tier 2: best combined stream at or below height (no merge)
        # Tier 3: unconstrained best (ignores height preference but never fails)
        return (
            f"bestvideo[height<={h}]+bestaudio"
            f"/bestvideo[height<={h}]+bestaudio[ext=m4a]"
            f"/best[height<={h}]"
            f"/bestvideo+bestaudio"
            f"/best"
        )

    # "best" quality — no height constraint, just get the best available
    # Prefer h264+aac in mp4 container for maximum compatibility, but don't
    # hard-require it — fall back all the way to "b".
    if e in ("mp4", "m4a"):
        return (
            "bestvideo[ext=mp4]+bestaudio[ext=m4a]"
            "/bestvideo[ext=mp4]+bestaudio"
            "/bestvideo+bestaudio"
            "/best"
        )
    elif e == "webm":
        return (
            "bestvideo[ext=webm]+bestaudio[ext=webm]"
            "/bestvideo[ext=webm]+bestaudio"
            "/bestvideo+bestaudio"
            "/best"
        )
    else:
        # mkv or any other container: no ext constraint, rely on merge_output_format
        return "bestvideo+bestaudio/best"


# ── Sync worker functions ──────────────────────────────────────────────────────

def _fetch_info(url: str, use_cookies: bool = False) -> dict:
    opts = {
        **BASE_OPTS,
        "skip_download": True,
    }

    # Apply cookies only during retry
    if use_cookies and COOKIES_FILE.exists():
        opts["cookiefile"] = str(COOKIES_FILE)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            raw = ydl.extract_info(url, download=False)
            raw = ydl.sanitize_info(raw)

        formats = [classify_format(f) for f in raw.get("formats", [])]
        formats = [f for f in formats if f is not None]

        thumbnails = sorted(
            raw.get("thumbnails") or [],
            key=lambda t: (t.get("width") or 0) * (t.get("height") or 0),
            reverse=True,
        )

        return {
            "id": raw.get("id"),
            "title": raw.get("title"),
            "description": (raw.get("description") or "")[:500] or None,
            "uploader": raw.get("uploader"),
            "uploader_url": raw.get("uploader_url"),
            "channel_id": raw.get("channel_id"),
            "upload_date": raw.get("upload_date"),
            "timestamp": raw.get("timestamp"),
            "duration": raw.get("duration"),
            "duration_string": raw.get("duration_string"),
            "view_count": raw.get("view_count"),
            "like_count": raw.get("like_count"),
            "comment_count": raw.get("comment_count"),
            "age_limit": raw.get("age_limit"),
            "categories": raw.get("categories"),
            "tags": (raw.get("tags") or [])[:20],
            "is_live": raw.get("is_live"),
            "was_live": raw.get("was_live"),
            "chapters": raw.get("chapters"),
            "thumbnail": raw.get("thumbnail"),
            "thumbnails": thumbnails[:5],
            "webpage_url": raw.get("webpage_url"),
            "playability_status": raw.get("availability"),
            "has_subtitles": bool(raw.get("subtitles")),
            "has_auto_captions": bool(raw.get("automatic_captions")),
            "subtitle_languages": list((raw.get("subtitles") or {}).keys()),
            "auto_caption_languages": list(
                (raw.get("automatic_captions") or {}).keys()
            )[:10],
            "format_count": len(formats),
            "formats": formats,
            "formats_grouped": {
                "combined": [
                    f for f in formats if f["type"] == "video+audio"
                ],
                "video_only": [
                    f for f in formats if f["type"] == "video-only"
                ],
                "audio_only": [
                    f for f in formats if f["type"] == "audio-only"
                ],
            },
        }

    except Exception as e:
        # Retry once with cookies on auth/bot detection
        if (
            not use_cookies
            and is_auth_error(e)
            and COOKIES_FILE.exists()
        ):
            print(
                f"[RETRY] Auth/Bot error detected for info {url}. "
                f"Retrying with cookies..."
            )

            return _fetch_info(url, use_cookies=True)

        print(f"[ERROR] fetch_info failed: {str(e)}")
        raise e


def _fetch_transcript(url: str, lang: str, use_cookies: bool = False) -> dict:
    # 1. Prepare options
    opts = {
        **BASE_OPTS,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3",
    }

    if use_cookies and COOKIES_FILE.exists():
        opts["cookiefile"] = str(COOKIES_FILE)

    try:
        # 2. Extract Info
        with yt_dlp.YoutubeDL(opts) as ydl:
            raw = ydl.extract_info(url, download=False)
            raw = ydl.sanitize_info(raw)

            subtitles = raw.get("subtitles") or {}
            auto_captions = raw.get("automatic_captions") or {}

            source = None
            source_type = None

            # 3. Find language source
            if lang in subtitles:
                source = subtitles[lang]
                source_type = "manual"
            elif lang in auto_captions:
                source = auto_captions[lang]
                source_type = "auto"
            else:
                # Fuzzy matching for locale codes (e.g., 'en' matches 'en-US')
                for key in list(subtitles.keys()) + list(auto_captions.keys()):
                    if key.startswith(lang):
                        source = subtitles.get(key) or auto_captions.get(key)
                        source_type = "manual" if key in subtitles else "auto"
                        lang = key
                        break

            if not source:
                available = list(subtitles.keys()) + list(auto_captions.keys())
                raise ValueError(
                    f"No transcript for lang '{lang}'. "
                    f"Available: {available or 'none'}"
                )

            # 4. Fetch the JSON3 content
            json3_entry = next((s for s in source if s.get("ext") == "json3"), None)

            if not json3_entry or not json3_entry.get("url"):
                raise ValueError("Transcript URL not found in yt-dlp response")

            with urllib.request.urlopen(json3_entry["url"], timeout=10) as resp:
                raw_json = json.loads(resp.read().decode("utf-8"))

            # 5. Parse segments
            segments = []
            for event in raw_json.get("events", []):
                if "segs" not in event:
                    continue
                start_ms = event.get("tStartMs", 0)
                duration_ms = event.get("dDurationMs", 0)
                text = "".join(seg.get("utf8", "") for seg in event["segs"]).strip()
                if text and text != "\n":
                    segments.append({
                        "start": round(start_ms / 1000, 2),
                        "duration": round(duration_ms / 1000, 2),
                        "text": text,
                    })

            full_text = " ".join(s["text"] for s in segments)

            return {
                "video_id": raw.get("id"),
                "title": raw.get("title"),
                "language": lang,
                "type": source_type,
                "segment_count": len(segments),
                "full_text": full_text,
                "segments": segments,
            }

    except Exception as e:
        # 6. Corrected Retry Logic
        if not use_cookies and is_auth_error(e) and COOKIES_FILE.exists():
            print(f"[RETRY] Auth/Bot error detected for transcript {url}. Retrying with cookies...")
            return _fetch_transcript(url, lang, use_cookies=True)
        
        print(f"[ERROR] fetch_transcript failed: {str(e)}")
        raise e

      


def _run_download(url: str, format_id: str | None, quality: str, ext: str) -> dict:
    """
    Downloads with the robust fallback selector from _resolve_format_selector.
    merge_output_format is always set so FFmpeg re-muxes when needed.
    For mp3/audio requests, FFmpegExtractAudio post-processor is added.
    """
    selector = _resolve_format_selector(format_id, quality, ext)

    uid = os.urandom(4).hex()
    output_template = str(DOWNLOADS_DIR / f"%(title)s [{uid}].%(ext)s")

    final_path: dict[str, str | None] = {"value": None}

    def progress_hook(d: dict):
        if d["status"] == "finished":
            final_path["value"] = d.get("filename") or d.get("info_dict", {}).get("filepath")

    is_audio_extract = ext == "mp3" or quality == "audio"
    preferred_codec = ext if ext in ("mp3", "m4a") else "mp3"

    merge_fmt = ext if ext in ("mp4", "mkv", "webm") else "mp4"

    opts: dict = {
        **BASE_OPTS,
        "format": selector,
        "outtmpl": output_template,
        "progress_hooks": [progress_hook],
        "merge_output_format": merge_fmt,
    }

    if is_audio_extract:
        opts["postprocessors"] = [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": preferred_codec,
        }]

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        info = ydl.sanitize_info(info)

    if not final_path["value"]:
        matches = list(DOWNLOADS_DIR.glob(f"*{uid}*"))
        matches = [f for f in matches if f.suffix not in (".part", ".ytdl", ".tmp")]
        if matches:
            final_path["value"] = str(matches[0])

    if not final_path["value"] or not Path(final_path["value"]).exists():
        raise FileNotFoundError("Downloaded file not found on disk")

    file_path = Path(final_path["value"])
    return {
        "title": info.get("title"),
        "file_path": str(file_path),
        "filename": file_path.name,
        "ext": file_path.suffix.lstrip("."),
        "filesize": file_path.stat().st_size,
        "filesize_human": _human_bytes(file_path.stat().st_size),
        "format_selector": selector,
        "format_id_used": info.get("format_id"),
        "resolution": info.get("resolution") or info.get("format_note"),
    }


def _run_quick_download(url: str) -> dict:
    uid = os.urandom(4).hex()
    opts = {
        "format": "bestvideo+bestaudio/best",
        "outtmpl": str(DOWNLOADS_DIR / f"%(title)s [{uid}].%(ext)s"),
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])

    matches = [
        f for f in DOWNLOADS_DIR.glob(f"*{uid}*")
        if f.suffix not in (".part", ".ytdl", ".tmp")
    ]
    if not matches:
        raise FileNotFoundError("Downloaded file not found on disk")

    file_path = matches[0]
    return {
        "filename": file_path.name,
        "ext": file_path.suffix.lstrip("."),
        "filesize_human": _human_bytes(file_path.stat().st_size),
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "service": "yt-dlp-server", "version": "2.1.0"}


@app.get("/info")
async def video_info(url: str):
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


@app.get("/formats")
async def video_formats(url: str):
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, _fetch_info, url)
        return {
            "success": True,
            "data": {
                "id": data["id"],
                "title": data["title"],
                "format_count": data["format_count"],
                "formats": data["formats"],
                "formats_grouped": data["formats_grouped"],
            },
        }
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/transcript")
async def video_transcript(url: str, lang: str = "en"):
    if not url:
        raise HTTPException(400, "Missing url param")
    try:
        loop = asyncio.get_event_loop()
        data = await loop.run_in_executor(executor, _fetch_transcript, url, lang)
        return {"success": True, "data": data}
    except ValueError as e:
        raise HTTPException(404, str(e))
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/download")
async def download_video(req: DownloadRequest):
    """
    Downloads with robust fallback format selectors.

    Body:
      { "url": "...", "quality": "720p", "ext": "mp4" }
      { "url": "...", "format_id": "137" }
      { "url": "...", "quality": "audio", "ext": "mp3" }

    Format selectors now use deep fallback chains — if the preferred
    codec/resolution isn't available, it falls back gracefully rather
    than failing with "Requested format is not available".
    """
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            executor,
            _run_download,
            req.url,
            req.format_id,
            req.quality or "best",
            req.ext or "mp4",
        )
        return {
            "success": True,
            "data": {
                **result,
                "fetch_url": f"/download/file?path={result['filename']}",
            },
        }
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/quick")
async def quick_download(req: QuickDownloadRequest):
    """Body: { "url": "..." }"""
    try:
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, _run_quick_download, req.url)
        return {
            "success": True,
            "data": {**result, "fetch_url": f"/download/file?path={result['filename']}"},
        }
    except FileNotFoundError as e:
        raise HTTPException(500, str(e))
    except yt_dlp.utils.DownloadError as e:
        raise HTTPException(400, str(e))
    except Exception as e:
        raise HTTPException(500, str(e))


@app.get("/download/file")
async def serve_file(path: str):
    """
    Serves a previously downloaded file by filename.
    """
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
