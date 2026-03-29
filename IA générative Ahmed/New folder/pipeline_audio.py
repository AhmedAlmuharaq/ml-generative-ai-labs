"""
TP5 - Step 6 module: audio pipeline functions
Imported by tp5_pipeline.py and reusable across projects.
"""

from pathlib import Path
from openai import OpenAI
from tp5_log import log_success

client = OpenAI()

OUTPUT_DIR = Path(__file__).parent / "outputs" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def synthesize_message(
    message: str,
    out_path: str | Path,
    voice: str = "alloy",
    language: str = "en-US",
) -> Path:
    """
    Generate an MP3 from text using OpenAI TTS.

    Args:
        message:  Text to synthesise.
        out_path: Output file path (.mp3).
        voice:    TTS voice (alloy, verse, nova, echo, fable, onyx, shimmer, solaria).
        language: BCP-47 tag embedded as a pronunciation hint (e.g. 'en-US', 'fr-FR').

    Returns:
        Path to the generated MP3.
    """
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    instructions = f"Speak in {language} with clear and natural pronunciation."

    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=message,
        instructions=instructions,
    ) as response:
        response.stream_to_file(output)

    log_success(f"Audio saved → {output}")
    return output
