# app.py — yt-dlp FastAPI server (v5.0.0, >= 2026.03.17 compliant)

import os
import asyncio
import threading
import time
import shutil
import uuid
from pathlib import Path
from typing import Optional, List, Dict

import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

# ── Setup ─────────────────────────────────────────────

app = FastAPI(title="yt-dlp server", version="5.0.0")

DOWNLOADS_DIR = Path(os.environ.get("DOWNLOADS_DIR", "./downloads"))
DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
print("FFMPEG PATH:", shutil.which("ffmpeg"))
CLEANUP_AFTER_MINUTES = int(os.environ.get("CLEANUP_AFTER_MINUTES", 10))

progress_store: Dict[str, dict] = {}

# Enable verbose via env
VERBOSE = os.environ.get("YTDLP_VERBOSE", "0") == "1"

BASE_OPTS = {
    "quiet": not VERBOSE,
    "no_warnings": not VERBOSE,
    "noplaylist": True,
    "socket_timeout": 30,
    "retries": 3,
    "fragment_retries": 3,
    "concurrent_fragment_downloads": 3,
    "nocheckcertificate": True,
}

# ── Models ────────────────────────────────────────────

class InfoRequest(BaseModel):
    url: str

class DownloadRequest(BaseModel):
    url: str
    format_id: Optional[str] = None


# ── Format Normalizer (>= 2026.03.17 safe selectors) ──

def normalize_selector(format_id: Optional[str]) -> str:
    if format_id:
        return f"{format_id}+bestaudio/{format_id}"
    return "b"


def simplify_formats(formats: List[dict]) -> dict:
    """
    Return only relevant formats:
    - 1080p, 720p, 360p video
    - best audio
    """
    video_targets = [1080, 720, 360]

    video_map = {h: None for h in video_targets}
    best_audio = None

    for f in formats:
        if f.get("vcodec") != "none" and f.get("height") in video_targets:
            h = f.get("height")
            if not video_map[h]:
                video_map[h] = {
                    "format_id": f["format_id"],
                    "ext": f["ext"],
                    "height": h,
                    "filesize": f.get("filesize") or f.get("filesize_approx"),
                }

        if f.get("acodec") != "none" and f.get("vcodec") == "none":
            if not best_audio or (f.get("abr") or 0) > (best_audio.get("abr") or 0):
                best_audio = {
                    "format_id": f["format_id"],
                    "ext": f["ext"],
                    "abr": f.get("abr"),
                    "filesize": f.get("filesize") or f.get("filesize_approx"),
                }

    return {
        "video": [v for v in video_map.values() if v],
        "audio": best_audio,
    }


# ── Core Logic ────────────────────────────────────────

def fetch_info(url: str):
    opts = {**BASE_OPTS, "skip_download": True}

    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)

    formats = info.get("formats", [])
    simplified = simplify_formats(formats)

    result = {
        "title": info.get("title"),
        "duration": info.get("duration"),
        "formats": simplified,
    }

    print("INFO RESULT:", result)  # tracing

    return result


def download_video(url: str, download_id: str, format_id: Optional[str]):
    selector = normalize_selector(format_id)

    output_template = str(DOWNLOADS_DIR / f"%(title)s-{download_id[:8]}.%(ext)s")

    def hook(d):
        if d["status"] == "downloading":
            total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            downloaded = d.get("downloaded_bytes", 0)
            percent = (downloaded / total * 100) if total else 0

            progress_store[download_id].update({
                "status": "downloading",
                "progress": round(percent, 2),
                "text": f"{round(percent,2)}%",
            })

        elif d["status"] == "finished":
            progress_store[download_id].update({
                "status": "processing",
                "progress": 95,
                "text": "processing...",
            })

    ydl_opts = {
        **BASE_OPTS,
        "format": selector,
        "outtmpl": output_template,
        "progress_hooks": [hook],
        "merge_output_format": "mp4",
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filepath = ydl.prepare_filename(info)

        file = Path(filepath)

        result = {
            "title": info.get("title"),
            "filename": file.name,
            "filepath": str(file),
        }

        print("DOWNLOAD RESULT:", result)  # tracing

        progress_store[download_id].update({
            "status": "completed",
            "progress": 100,
            "text": "done",
            **result
        })

        threading.Thread(target=cleanup_file, args=(file,), daemon=True).start()

    except Exception as e:
        progress_store[download_id].update({
            "status": "error",
            "error": str(e),
        })


def cleanup_file(file: Path):
    time.sleep(CLEANUP_AFTER_MINUTES * 60)
    try:
        if file.exists():
            file.unlink()
    except:
        pass


# ── UI ────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>
<title>Downloader</title>
<style>
body{font-family:sans-serif;background:#0f172a;color:#fff;text-align:center;padding-top:60px}
input,select{width:400px;padding:10px;border-radius:8px;border:none;margin:5px}
button{padding:12px 20px;border:none;background:#3b82f6;color:#fff;border-radius:8px}
.bar{width:400px;height:8px;background:#333;margin:20px auto;border-radius:4px}
.fill{height:100%;width:0;background:#22c55e}
</style>
</head>
<body>

<h2>yt-dlp advanced downloader</h2>

<input id="url" placeholder="paste url"/><br>
<button onclick="getInfo()">get formats</button><br>

<select id="formats"></select><br>
<button onclick="start()">download</button>

<div class="bar"><div id="fill" class="fill"></div></div>
<p id="text"></p>

<script>
let currentId=null;

async function getInfo(){
    const url=document.getElementById("url").value;

    const res=await fetch("/info",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url})});
    const data=await res.json();

    const select=document.getElementById("formats");
    select.innerHTML="";

    data.formats.video.forEach(f=>{
        const opt=document.createElement("option");
        opt.value=f.format_id;
        opt.text=`${f.height}p (${(f.filesize||0)/1000000}MB)`;
        select.appendChild(opt);
    });

    if(data.formats.audio){
        const opt=document.createElement("option");
        opt.value=data.formats.audio.format_id;
        opt.text=`audio (${(data.formats.audio.filesize||0)/1000000}MB)`;
        select.appendChild(opt);
    }
}

async function start(){
    const url=document.getElementById("url").value;
    const format_id=document.getElementById("formats").value;

    const res=await fetch("/download",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({url,format_id})});
    const data=await res.json();

    currentId=data.download_id;
    poll();
}

async function poll(){
    if(!currentId)return;

    const res=await fetch("/progress/"+currentId);
    const data=await res.json();

    document.getElementById("fill").style.width=data.progress+"%";
    document.getElementById("text").innerText=data.text||data.status;

    if(data.status==="completed"){
        window.location="/download/"+currentId;
        return;
    }

    if(data.status==="error"){
        document.getElementById("text").innerText=data.error;
        return;
    }

    setTimeout(poll,1000);
}
</script>

</body>
</html>
"""


# ── API ───────────────────────────────────────────────

@app.post("/info")
async def info(req: InfoRequest):
    try:
        data = await asyncio.to_thread(fetch_info, req.url)
        return data
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/download")
async def download(req: DownloadRequest):
    download_id = str(uuid.uuid4())

    progress_store[download_id] = {
        "status": "starting",
        "progress": 0,
        "text": "starting...",
    }

    threading.Thread(
        target=download_video,
        args=(req.url, download_id, req.format_id),
        daemon=True
    ).start()

    return {"download_id": download_id}


@app.get("/progress/{download_id}")
async def progress(download_id: str):
    data = progress_store.get(download_id)
    if not data:
        return {"status": "not_found"}
    return data


@app.get("/download/{download_id}")
async def serve(download_id: str):
    data = progress_store.get(download_id)

    if not data or data.get("status") != "completed":
        raise HTTPException(404, "file not ready")

    path = Path(data["filepath"])

    if not path.exists():
        raise HTTPException(404, "file missing")

    return FileResponse(path=str(path), filename=data["filename"])


@app.get("/health")
def health():
    return {"status": "ok"}
