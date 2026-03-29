"""
TP5 - Step 6: Full multimodal pipeline — Vision → Text → (Translation) → Audio
Exercises 6.1 and 6.2

Requires: OPENAI_API_KEY in .env
          Images in inputs/images/

Pipeline:
  image ──► describe_image (vision) ──► summary text
                                              │
                                    (Ex 6.1) translate EN→FR
                                              │
                                        synthesize_message (TTS)
                                              │
                                         audio file (MP3)
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

from pipeline_audio import synthesize_message
from pipeline_vision import describe_image
from tp5_log import log, log_step, log_success, log_error, log_warning

load_dotenv(override=True)

client = OpenAI()

INPUT_DIR  = Path(__file__).parent / "inputs"  / "images"
OUTPUT_DIR = Path(__file__).parent / "outputs"
INPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# Ex 6.1 — Translation helper (English → target language)
# =============================================================================

def translate_text(text: str, target_language: str = "French") -> str:
    """
    Translate text into target_language using the ChatCompletion API.

    Ex 6.1: added before TTS so the audio is in the desired language.
    """
    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": (
                    f"You are a professional translator. "
                    f"Translate the user's text into {target_language}. "
                    "Return ONLY the translated text, nothing else."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.2,
        max_output_tokens=600,
    )
    texts = [
        content.text.strip()
        for item in response.output
        for content in item.content
        if content.type == "output_text"
    ]
    return "\n".join(texts)


# =============================================================================
# STEP 6 — Core pipeline
# =============================================================================

def visual_analysis_to_audio(
    image_path: str | Path,
    question: str,
    voice: str = "verse",
    translate_to: str | None = None,   # Ex 6.1: set to "French", "Spanish", etc. or None
    out_audio: str | Path | None = None,
) -> Path:
    """
    Full pipeline: image → vision analysis → optional translation → TTS audio.

    Args:
        image_path:   Path to the image to analyse.
        question:     The analytical question / instruction for the vision model.
        voice:        TTS voice to use.
        translate_to: If set, translate the summary into this language before TTS.
        out_audio:    Output MP3 path. Defaults to outputs/audio/analysis.mp3.

    Returns:
        Path to the generated MP3 file.
    """
    image_path = Path(image_path)
    if out_audio is None:
        out_audio = OUTPUT_DIR / "audio" / f"analysis_{image_path.stem}.mp3"

    # Step 1: vision analysis
    log_step("Step 1 — Vision analysis")
    summary = describe_image(image_path, question)
    log(f"Summary:\n{summary}\n")

    # Save text summary
    txt_out = OUTPUT_DIR / "vision" / f"analysis_{image_path.stem}.txt"
    txt_out.parent.mkdir(parents=True, exist_ok=True)
    txt_out.write_text(summary, encoding="utf-8")
    log_success(f"Summary saved → {txt_out}")

    # Step 2 (Ex 6.1): optional translation
    tts_text = summary
    tts_language = "en-US"
    if translate_to:
        log_step(f"Ex 6.1 — Translating into {translate_to}")
        tts_text = translate_text(summary, target_language=translate_to)
        log(f"Translated:\n{tts_text}\n")
        # Map language name to BCP-47 tag for TTS
        lang_map = {
            "french": "fr-FR", "spanish": "es-ES", "german": "de-DE",
            "italian": "it-IT", "portuguese": "pt-PT", "arabic": "ar-SA",
        }
        tts_language = lang_map.get(translate_to.lower(), "en-US")

    # Step 3: TTS
    log_step("Step 3 — Text-to-Speech")
    audio_path = synthesize_message(
        message=f"Here is my analysis: {tts_text}",
        out_path=out_audio,
        voice=voice,
        language=tts_language,
    )
    return audio_path


# =============================================================================
# Ex 6.2 — CLI interface (argparse)
# =============================================================================

def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tp5_pipeline",
        description="Multimodal pipeline: image → analysis text → (translation) → audio",
    )
    parser.add_argument("image", nargs="?", help="Path to the image to analyse")
    parser.add_argument(
        "--question", "-q",
        default="Identify the main elements of this image and explain their significance.",
        help="Question or instruction for the vision model",
    )
    parser.add_argument(
        "--voice", "-v",
        default="verse",
        choices=["alloy", "verse", "nova", "echo", "fable", "onyx", "shimmer", "solaria"],
        help="TTS voice (default: verse)",
    )
    parser.add_argument(
        "--translate", "-t",
        default=None,
        metavar="LANGUAGE",
        help="Translate the summary before TTS (e.g. French, Spanish, German)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="Output MP3 path (default: outputs/audio/analysis_<image>.mp3)",
    )
    return parser


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = build_cli()
    args = parser.parse_args()

    # If no image provided via CLI, use the first one found in inputs/images/
    if args.image:
        image_path = Path(args.image)
    else:
        candidates = (
            list(INPUT_DIR.glob("*.png")) +
            list(INPUT_DIR.glob("*.jpg")) +
            list(INPUT_DIR.glob("*.jpeg"))
        )
        if not candidates:
            log_error(
                f"No images found in {INPUT_DIR}. "
                "Provide an image path as argument or place images in inputs/images/."
            )
            raise SystemExit(1)
        image_path = candidates[0]
        log_warning(f"No image specified — using first found: {image_path.name}")

    audio_path = visual_analysis_to_audio(
        image_path=image_path,
        question=args.question,
        voice=args.voice,
        translate_to=args.translate,
        out_audio=args.output,
    )

    log_success(f"\nDone! Audio analysis available at: {audio_path.resolve()}")
    print(
        "\nUsage examples:\n"
        "  python tp5_pipeline.py inputs/images/dashboard.png\n"
        "  python tp5_pipeline.py inputs/images/chart.png --translate French --voice verse\n"
        "  python tp5_pipeline.py inputs/images/photo.jpg -q 'List the KPIs' -t Spanish -o out.mp3\n"
    )
