from __future__ import annotations

"""Video download, Whisper transcription, and frame extraction.

Architecture (adapted from reference whisper_transcribe.py)
-----------------------------------------------------------
  1. Download full video with yt-dlp (NOT audio-only — frames are needed)
  2. Run OpenAI Whisper API on the video file → transcript text
  3. Extract frames with ffmpeg:
       - 1 frame every FRAME_SAMPLE_RATE seconds (default 2s)
       - Plus the thumbnail at t=0
     Frames are named "{post_id}_fr{ts:05d}" (ts = seconds).
  4. Store transcript in tiktok_fin_posts.transcript
  5. Store frames as JPEG BLOBs in tiktok_frames table

A failed-ID log (JSON file) tracks videos that consistently fail so we
don't retry them endlessly on subsequent runs.

Parallel workers: ThreadPoolExecutor with TRANSCRIPTION_WORKERS threads.
File-size cap: 25MB (OpenAI Whisper API limit).
"""

import json
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from pipeline.logger import get_logger
from pipeline.tiktok_collection.storage import FinCollectionStorage

log = get_logger(__name__)

_MAX_FILE_BYTES = 24 * 1024 * 1024   # 24 MB (below Whisper 25 MB limit)
_MIN_FILE_BYTES = 1_000               # skip suspiciously small files
_FRAME_JPEG_QUALITY = 85              # ffmpeg JPEG quality (1-100)


# ---------------------------------------------------------------------------
# frame_id helpers
# ---------------------------------------------------------------------------

def frame_id(post_id: str, ts_sec: int) -> str:
    return f"{post_id}_fr{ts_sec:05d}"


def thumbnail_id(post_id: str) -> str:
    return f"{post_id}_thumb"


# ---------------------------------------------------------------------------
# yt-dlp download
# ---------------------------------------------------------------------------

def _download_video(url: str, output_path: Path, timeout: int = 120) -> bool:
    """Download full video (not audio-only) via yt-dlp.  Returns True on success."""
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", str(output_path),
        "--no-warnings",
        "--quiet",
        "--no-playlist",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return result.returncode == 0 and output_path.exists()


# ---------------------------------------------------------------------------
# OpenAI Whisper transcription
# ---------------------------------------------------------------------------

def _transcribe(openai_api_key: str, video_path: Path) -> str | None:
    """Send video to OpenAI Whisper API. Returns transcript text or None."""
    from openai import OpenAI
    client = OpenAI(api_key=openai_api_key)
    with open(video_path, "rb") as f:
        response = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="text",
        )
    text = response.strip() if isinstance(response, str) else str(response).strip()
    return text or None


# ---------------------------------------------------------------------------
# Frame extraction via ffmpeg
# ---------------------------------------------------------------------------

def _extract_frames(
    video_path: Path,
    post_id: str,
    frames_dir: Path,
    sample_rate: int,
) -> list[tuple[str, int, Path]]:
    """Extract frames using ffmpeg.

    Returns list of (frame_id, timestamp_sec, jpeg_path).
    Also extracts the thumbnail (t=0) tagged as thumbnail_id(post_id).
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    results: list[tuple[str, int, Path]] = []

    # Thumbnail at t=0
    thumb_path = frames_dir / f"{thumbnail_id(post_id)}.jpg"
    cmd = [
        "ffmpeg", "-y", "-ss", "0",
        "-i", str(video_path),
        "-vframes", "1",
        "-q:v", str(int((100 - _FRAME_JPEG_QUALITY) / 10) + 1),
        str(thumb_path),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=30)
    if r.returncode == 0 and thumb_path.exists() and thumb_path.stat().st_size > 0:
        results.append((thumbnail_id(post_id), 0, thumb_path))

    # Frames at 1-per-sample_rate-seconds using select filter
    # Output pattern: {post_id}_fr{ts:05d}_%05d.jpg — we rename after
    frame_pattern = frames_dir / f"{post_id}_fr%05d.jpg"
    cmd = [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-vf", f"fps=1/{sample_rate}",
        "-q:v", str(int((100 - _FRAME_JPEG_QUALITY) / 10) + 1),
        str(frame_pattern),
    ]
    r = subprocess.run(cmd, capture_output=True, timeout=120)
    if r.returncode != 0:
        log.warning("ffmpeg_frame_extraction_failed", post_id=post_id,
                    stderr=r.stderr.decode(errors="replace")[:300])

    # Collect produced frames; derive actual timestamp from index
    frame_files = sorted(frames_dir.glob(f"{post_id}_fr*.jpg"))
    for idx, fpath in enumerate(frame_files):
        ts = (idx + 1) * sample_rate  # index 1 → t=sample_rate, etc.
        fid = frame_id(post_id, ts)
        # Rename to canonical frame_id
        canonical = frames_dir / f"{fid}.jpg"
        fpath.rename(canonical)
        results.append((fid, ts, canonical))

    return results


# ---------------------------------------------------------------------------
# Per-video processing
# ---------------------------------------------------------------------------

def _process_one(
    post_id: str,
    url: str,
    openai_api_key: str,
    sample_rate: int,
) -> dict[str, Any]:
    """Download + transcribe + extract frames for a single video.

    Returns a result dict:
      {post_id, success, transcript, frames: [(frame_id, ts, bytes)], error}
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        video_path = tmpdir_path / f"{post_id}.mp4"
        frames_dir = tmpdir_path / "frames"

        # Step 1: Download full video
        try:
            ok = _download_video(url, video_path)
            if not ok:
                # yt-dlp may use a different extension
                candidates = list(tmpdir_path.glob(f"{post_id}.*"))
                if candidates:
                    video_path = candidates[0]
                else:
                    return _fail(post_id, "download_failed")
        except subprocess.TimeoutExpired:
            return _fail(post_id, "download_timeout")
        except Exception as exc:
            return _fail(post_id, f"download_error:{exc}")

        size = video_path.stat().st_size
        if size > _MAX_FILE_BYTES:
            return _fail(post_id, f"file_too_large:{size}")
        if size < _MIN_FILE_BYTES:
            return _fail(post_id, f"file_too_small:{size}")

        # Step 2: Transcribe via Whisper
        transcript: str | None = None
        if openai_api_key:
            try:
                transcript = _transcribe(openai_api_key, video_path)
            except Exception as exc:
                log.warning("whisper_error", post_id=post_id, exc=str(exc))
        else:
            log.warning("whisper_skipped_no_key", post_id=post_id)

        # Step 3: Extract frames
        frames: list[tuple[str, int, bytes]] = []
        try:
            raw_frames = _extract_frames(video_path, post_id, frames_dir, sample_rate)
            for fid, ts, fpath in raw_frames:
                with open(fpath, "rb") as fh:
                    frames.append((fid, ts, fh.read()))
        except Exception as exc:
            log.warning("frame_extraction_error", post_id=post_id, exc=str(exc))

        return {
            "post_id": post_id,
            "success": True,
            "transcript": transcript,
            "frames": frames,
            "error": None,
        }


def _fail(post_id: str, reason: str) -> dict[str, Any]:
    return {"post_id": post_id, "success": False, "transcript": None, "frames": [], "error": reason}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_transcription(
    posts: list[dict[str, Any]],
    storage: FinCollectionStorage,
    openai_api_key: str,
    sample_rate: int,
    workers: int,
    failed_log_path: Path,
) -> tuple[int, int]:
    """Transcribe + extract frames for all posts that don't yet have a transcript.

    Returns (n_success, n_fail).
    Skips posts already having a transcript stored in the DB.
    Skips post_ids in the failed log.
    """
    # Load previously failed IDs
    failed_ids: set[str] = set()
    if failed_log_path.exists():
        with open(failed_log_path, encoding="utf-8") as f:
            failed_ids = set(json.load(f))

    # Build pending list: posts without transcripts and not previously failed
    pending = []
    for post in posts:
        pid = str(post.get("post_id", ""))
        if not pid or not post.get("url"):
            continue
        if pid in failed_ids:
            continue
        row = storage.get_post(pid)
        if row and row.get("transcript"):
            continue
        pending.append((pid, str(post["url"])))

    if not pending:
        log.info("transcription_nothing_to_do")
        return 0, 0

    log.info("transcription_start", n=len(pending), workers=workers)

    n_success = 0
    n_fail = 0
    new_failed: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_pid = {
            executor.submit(_process_one, pid, url, openai_api_key, sample_rate): pid
            for pid, url in pending
        }

        for future in as_completed(future_to_pid):
            result = future.result()
            pid = result["post_id"]

            if not result["success"]:
                log.warning("transcription_failed", post_id=pid, error=result["error"])
                n_fail += 1
                new_failed.append(pid)
                continue

            # Store transcript
            if result["transcript"]:
                storage.set_transcript(pid, result["transcript"])

            # Store frames
            for fid, ts, img_bytes in result["frames"]:
                is_thumb = fid.endswith("_thumb")
                storage.upsert_frame(fid, pid, ts, img_bytes, is_thumbnail=is_thumb)

            storage.commit()
            n_success += 1
            log.debug("transcription_video_done", post_id=pid,
                      n_frames=len(result["frames"]), has_transcript=bool(result["transcript"]))

    # Persist updated failed log
    all_failed = sorted(failed_ids | set(new_failed))
    failed_log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_log_path, "w", encoding="utf-8") as f:
        json.dump(all_failed, f)

    log.info("transcription_done", n_success=n_success, n_fail=n_fail)
    return n_success, n_fail
