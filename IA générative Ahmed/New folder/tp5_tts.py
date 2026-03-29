"""
TP5 - Step 2: Text-to-Speech (TTS)
Exercises 2.1 and 2.2

Requires: OPENAI_API_KEY in .env
Output files are saved to: outputs/audio/
"""

import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tp5_log import log, log_success, log_step, log_warning

load_dotenv(override=True)

client = OpenAI()
OUTPUT_DIR = Path(__file__).parent / "outputs" / "audio"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STEP 2 — Text-to-Speech
# =============================================================================

def synthesize_message(
    message: str,
    out_path: str | Path,
    voice: str = "alloy",
    language: str = "en-US",  # Ex 2.2: language parameter
) -> Path:
    """
    Generate an MP3 file from a text message using OpenAI TTS.

    Args:
        message:  The text to synthesise.
        out_path: Destination file path (.mp3).
        voice:    TTS voice name (alloy, verse, nova, solaria, echo, fable, onyx, shimmer).
        language: BCP-47 language tag used as a hint in the prompt (e.g. 'en-US', 'fr-FR').
                  OpenAI does not expose a language parameter directly — we embed it in the
                  instructions field to influence pronunciation.

    Returns:
        Path to the generated MP3 file.
    """
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Ex 2.2: embed language hint so the model adjusts pronunciation
    instructions = f"Speak in {language} with clear and natural pronunciation."

    t0 = time.perf_counter()
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=message,
        instructions=instructions,
    ) as response:
        response.stream_to_file(output)

    elapsed = time.perf_counter() - t0
    log_success(f"TTS done in {elapsed:.2f}s → {output}")
    return output


# =============================================================================
# Ex 2.1 — Compare at least three voices
# =============================================================================

SAMPLE_TEXT_EN = (
    "Hello! Welcome to the multimodality lab. "
    "Today we will explore text-to-speech and beyond."
)

SAMPLE_TEXT_FR = (
    "Bonjour ! Bienvenue dans le laboratoire de multimodalité. "
    "Aujourd'hui, nous explorons la synthèse vocale et bien plus encore."
)

# Ex 2.1: voices to compare
VOICES = ["alloy", "verse", "nova", "echo", "fable"]

log_step("Ex 2.1 — Voice comparison")
for voice in VOICES:
    path = synthesize_message(
        SAMPLE_TEXT_EN,
        out_path=OUTPUT_DIR / f"voice_{voice}.mp3",
        voice=voice,
        language="en-US",
    )
    log(f"Voice '{voice}' → {path.name}")

print(
    "\nEx 2.1 Notes:\n"
    "  alloy  — neutral, balanced, gender-neutral\n"
    "  verse  — expressive, slightly warmer tone\n"
    "  nova   — clear and professional, slightly higher pitch\n"
    "  echo   — deeper, authoritative\n"
    "  fable  — storytelling tone, more dramatic\n"
    "Listen to each file and pick the voice that fits your use case best.\n"
)

# =============================================================================
# Ex 2.2 — Language variation + content types
# =============================================================================

log_step("Ex 2.2 — Language & content variation")

CONTENT_SAMPLES = [
    {
        "label": "dialogue_en",
        "text": "Sure, I can help you with that. What would you like to know?",
        "voice": "nova",
        "language": "en-US",
    },
    {
        "label": "instructions_fr",
        "text": "Veuillez déposer votre billet dans le lecteur et patienter.",
        "voice": "verse",
        "language": "fr-FR",
    },
    {
        "label": "storytelling_en",
        "text": (
            "Once upon a time, in a land where machines could dream, "
            "a humble algorithm discovered the secret of human language."
        ),
        "voice": "fable",
        "language": "en-US",
    },
]

for sample in CONTENT_SAMPLES:
    path = synthesize_message(
        message=sample["text"],
        out_path=OUTPUT_DIR / f"{sample['label']}.mp3",
        voice=sample["voice"],
        language=sample["language"],
    )
    log(f"[{sample['label']}] voice={sample['voice']} lang={sample['language']} → {path.name}")
