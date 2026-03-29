"""
TP5 - Step 3: Speech-to-Text (STT / transcription)
Exercises 3.1 and 3.2

Requires: OPENAI_API_KEY in .env
          pip install pydub
          ffmpeg available in PATH

Place your audio file at: inputs/audio/meeting.mp3
(or adjust AUDIO_FILE below)
"""

import json
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from pydub import AudioSegment
from tp5_log import log, log_step, log_success, log_warning

load_dotenv(override=True)

client = OpenAI()

INPUT_DIR  = Path(__file__).parent / "inputs"  / "audio"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "transcriptions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIO_FILE = INPUT_DIR / "meeting.mp3"

# =============================================================================
# STEP 3 — Speech-to-Text
# =============================================================================

@dataclass
class ChunkResult:
    index: int
    start_ms: int
    end_ms: int
    text: str
    duration_s: float  # Ex 3.1: processing time per chunk


def transcribe_mp3(
    mp3_path: str | Path,
    chunk_minutes: int = 2,
    response_format: str = "verbose_json",  # Ex 3.2: switch to "json" to get timestamps
) -> tuple[str, List[ChunkResult]]:
    """
    Transcribe a long MP3 by splitting it into chunks.

    Returns:
        full_text: The complete concatenated transcript.
        chunks:    List of ChunkResult with per-chunk timing (Ex 3.1) and text.
    """
    audio = AudioSegment.from_file(str(mp3_path))
    chunk_ms = chunk_minutes * 60 * 1000
    chunks: List[ChunkResult] = []

    log(f"Audio duration: {len(audio) / 1000:.1f}s  |  chunk size: {chunk_minutes}min")

    with tempfile.TemporaryDirectory() as workdir:
        for index, start in enumerate(range(0, len(audio), chunk_ms)):
            segment = audio[start : start + chunk_ms]

            if len(segment) < 1_000:   # skip sub-second fragments
                log_warning(f"Chunk {index} too short ({len(segment)}ms), skipping.")
                continue

            chunk_path = Path(workdir) / f"chunk_{index}.mp3"
            segment.export(chunk_path, format="mp3")

            # Ex 3.1: measure time per chunk
            t0 = time.perf_counter()
            with chunk_path.open("rb") as chunk_file:
                result = client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=chunk_file,
                    response_format=response_format,
                )
            elapsed = time.perf_counter() - t0

            chunks.append(ChunkResult(
                index=index,
                start_ms=start,
                end_ms=start + len(segment),
                text=result.text.strip(),
                duration_s=round(elapsed, 3),
            ))
            log(f"  Chunk {index}: {elapsed:.2f}s → '{result.text[:60]}...'")

    full_text = " ".join(c.text for c in chunks)
    return full_text, chunks


# =============================================================================
# Ex 3.1 — Chunk timing analysis
# =============================================================================

def run_chunk_size_comparison(audio_path: Path) -> None:
    """Test different chunk sizes and print a comparison table."""
    log_step("Ex 3.1 — Chunk size vs processing time")

    for chunk_min in [1, 2, 5]:
        t_start = time.perf_counter()
        _, chunks = transcribe_mp3(audio_path, chunk_minutes=chunk_min)
        total_time = time.perf_counter() - t_start

        avg_chunk_time = sum(c.duration_s for c in chunks) / max(len(chunks), 1)
        log_success(
            f"chunk={chunk_min}min | n_chunks={len(chunks)} | "
            f"total={total_time:.2f}s | avg_per_chunk={avg_chunk_time:.2f}s"
        )

    print(
        "\nEx 3.1 Observation:\n"
        "  Larger chunks → fewer API calls → lower per-request overhead, but longer\n"
        "  individual calls. Smaller chunks → more parallelisable, faster failure\n"
        "  recovery, but higher aggregate overhead. 2-min chunks is a good default.\n"
    )


# =============================================================================
# Ex 3.2 — Parse timestamps from verbose_json
# =============================================================================

@dataclass
class TimelineEntry:
    chunk_index: int
    global_start_s: float
    global_end_s: float
    text: str


def build_timeline(chunks: List[ChunkResult]) -> List[TimelineEntry]:
    """
    Reconstruct a global timeline from per-chunk verbose_json responses.
    Each chunk's segment timestamps are offset by the chunk's global start.
    """
    timeline: List[TimelineEntry] = []
    for chunk in chunks:
        chunk_offset_s = chunk.start_ms / 1000.0
        # verbose_json includes .segments with start/end per sentence
        # Here we use the full chunk as one entry (extend for sub-chunk granularity)
        timeline.append(TimelineEntry(
            chunk_index=chunk.index,
            global_start_s=chunk_offset_s,
            global_end_s=chunk.end_ms / 1000.0,
            text=chunk.text,
        ))
    return timeline


def save_timeline_json(timeline: List[TimelineEntry], out_path: Path) -> None:
    data = [
        {
            "chunk": e.chunk_index,
            "start_s": e.global_start_s,
            "end_s": e.global_end_s,
            "text": e.text,
        }
        for e in timeline
    ]
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log_success(f"Timeline saved to {out_path}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    if not AUDIO_FILE.exists():
        log_warning(
            f"Audio file not found: {AUDIO_FILE}\n"
            "  → Run tp5_tts.py first to generate a sample, then rename it to\n"
            "    inputs/audio/meeting.mp3, or place your own MP3 there."
        )
    else:
        log_step("Step 3 — Transcription")
        full_text, chunks = transcribe_mp3(AUDIO_FILE, chunk_minutes=2)

        # Save plain transcript
        txt_out = OUTPUT_DIR / "meeting.txt"
        txt_out.write_text(full_text, encoding="utf-8")
        log_success(f"Transcript saved to {txt_out}")

        # Ex 3.2: build and save timeline
        log_step("Ex 3.2 — Timeline from verbose_json")
        timeline = build_timeline(chunks)
        save_timeline_json(timeline, OUTPUT_DIR / "meeting_timeline.json")

        # Ex 3.1: chunk timing report
        log_step("Ex 3.1 — Per-chunk timing")
        for c in chunks:
            log(f"  Chunk {c.index} ({c.start_ms//1000}s–{c.end_ms//1000}s): "
                f"{c.duration_s}s API time")

        run_chunk_size_comparison(AUDIO_FILE)
