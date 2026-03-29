"""
TP5 - Step 4: Vision — image understanding
Exercises 4.1 and 4.2

Requires: OPENAI_API_KEY in .env
          pip install pillow

Place test images in: inputs/images/
"""

import base64
import json
import mimetypes
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tp5_log import log, log_step, log_success, log_error

load_dotenv(override=True)

client = OpenAI()

INPUT_DIR  = Path(__file__).parent / "inputs"  / "images"
OUTPUT_DIR = Path(__file__).parent / "outputs" / "vision"
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STEP 4 — Vision: describe / analyse an image
# =============================================================================

def encode_image_to_data_url(image_path: str | Path) -> str:
    """Encode a local image as a base64 data URL ready for the OpenAI API."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None:
        raise ValueError(f"Cannot infer MIME type for: {path}")

    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


def describe_image(
    image_path: str | Path,
    question: str,
    model: str = "gpt-4.1-mini",
    max_tokens: int = 300,
    output_format: str = "text",   # Ex 4.2: "text" | "bullet" | "json"
) -> str:
    """
    Send an image + question to the vision model and return the answer.

    Args:
        image_path:    Path to a local image file.
        question:      The question or instruction to send alongside the image.
        model:         OpenAI model with vision capability.
        max_tokens:    Maximum tokens in the response.
        output_format: Controls the formatting constraint injected into the prompt.
                       "text"   → free-form prose
                       "bullet" → bullet-point list
                       "json"   → strict JSON object

    Returns:
        The model's textual response.
    """
    data_url = encode_image_to_data_url(image_path)

    # Ex 4.2: format constraint injected into the question
    format_instructions = {
        "text":   "",
        "bullet": " Format your answer as a concise bullet-point list (max 6 bullets).",
        "json":   (
            " Respond ONLY with a valid JSON object using keys: "
            "'summary' (string), 'key_points' (list of strings), 'concerns' (list of strings). "
            "No markdown, no explanation outside the JSON."
        ),
    }
    full_question = question + format_instructions.get(output_format, "")

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text",  "text": full_question},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        temperature=0.1,
        max_output_tokens=max_tokens,
    )

    texts = []
    for item in response.output:
        for content in item.content:
            if content.type == "output_text":
                texts.append(content.text.strip())
    return "\n".join(texts)


# =============================================================================
# STEP 4 — Demo
# =============================================================================

QUESTION = "What can you deduce from this image? Describe the main elements and their significance."

log_step("Step 4 — Vision demo")

# Find any image in the inputs folder to use as a test
test_images = list(INPUT_DIR.glob("*.png")) + list(INPUT_DIR.glob("*.jpg")) + list(INPUT_DIR.glob("*.jpeg"))

if not test_images:
    log_error(
        f"No images found in {INPUT_DIR}. "
        "Place at least one PNG/JPG image there before running this script."
    )
else:
    test_image = test_images[0]
    log(f"Using image: {test_image.name}")

    # Base analysis
    answer_text = describe_image(test_image, QUESTION, model="gpt-4.1-mini")
    log_success("gpt-4.1-mini answer (prose):")
    print(answer_text)

    # ---------- Ex 4.1: compare gpt-4.1-mini vs gpt-4.1 ----------
    log_step("Ex 4.1 — Model comparison: gpt-4.1-mini vs gpt-4.1")

    for model in ["gpt-4.1-mini", "gpt-4.1"]:
        log(f"Querying {model}...")
        try:
            answer = describe_image(test_image, QUESTION, model=model, max_tokens=400)
            out_file = OUTPUT_DIR / f"analysis_{model.replace('.', '_')}.txt"
            out_file.write_text(answer, encoding="utf-8")
            log_success(f"{model} → {out_file.name}")
            print(f"\n[{model}]\n{answer}\n")
        except Exception as e:
            log_error(f"{model} failed: {e}")

    print(
        "Ex 4.1 Observation:\n"
        "  gpt-4.1-mini is faster and cheaper but may miss subtle details.\n"
        "  gpt-4.1 produces richer descriptions and handles complex documents better.\n"
        "  For handwritten text or technical diagrams, prefer gpt-4.1.\n"
    )

    # ---------- Ex 4.2: format constraints ----------
    log_step("Ex 4.2 — Output format constraints")

    for fmt in ["text", "bullet", "json"]:
        log(f"Format: {fmt}")
        answer = describe_image(test_image, QUESTION, output_format=fmt)
        out_file = OUTPUT_DIR / f"analysis_{fmt}.txt"
        out_file.write_text(answer, encoding="utf-8")
        print(f"\n[{fmt}]\n{answer}\n")

        # Validate JSON if requested
        if fmt == "json":
            try:
                parsed = json.loads(answer)
                log_success("JSON parsed successfully.")
                log(f"  Keys: {list(parsed.keys())}")
            except json.JSONDecodeError as e:
                log_error(f"JSON parse error (model did not comply with format): {e}")

    print(
        "Ex 4.2 Observation:\n"
        "  Format constraints via prompt injection are generally respected by GPT-4 models.\n"
        "  For production use, combine with Pydantic structured output (TP2 technique)\n"
        "  to guarantee schema compliance and raise a validation error on failure.\n"
    )
