"""
TP5 - Step 5: Image generation
Exercises 5.1 and 5.2

Requires: OPENAI_API_KEY in .env
Output files saved to: outputs/images/
"""

import base64
import time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from tp5_log import log, log_step, log_success, log_warning

load_dotenv(override=True)

client = OpenAI()

OUTPUT_DIR = Path(__file__).parent / "outputs" / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# STEP 5 — Image generation
# =============================================================================

def generate_image(
    prompt: str,
    out_path: str | Path,
    size: str = "1024x1024",
    quality: str = "high",
) -> Path:
    """
    Generate an image from a text prompt and save it locally.

    Args:
        prompt:   Descriptive text prompt for the image.
        out_path: Destination file path (.png).
        size:     Image dimensions: '1024x1024', '512x512', '1792x1024', '1024x1792'.
        quality:  'standard' (faster, cheaper) or 'high' (more detailed).

    Returns:
        Path to the saved PNG file.
    """
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=size,
        quality=quality,
    )
    elapsed = time.perf_counter() - t0

    image_b64 = result.data[0].b64_json
    output.write_bytes(base64.b64decode(image_b64))
    log_success(f"Image generated in {elapsed:.2f}s → {output} ({size}, quality={quality})")
    return output


# =============================================================================
# STEP 5 — Base demo
# =============================================================================

log_step("Step 5 — Base image generation")

base_path = generate_image(
    prompt=(
        "A minimalist poster illustrating the meeting between audio and vision in AI, "
        "clean lines, soft gradient background, two symbolic icons: a sound wave and an eye."
    ),
    out_path=OUTPUT_DIR / "multimodal_poster.png",
)

# =============================================================================
# Ex 5.1 — Size and quality comparison
# =============================================================================

log_step("Ex 5.1 — Size & quality comparison")

CONFIGS = [
    {"size": "1024x1024", "quality": "standard", "label": "square_standard"},
    {"size": "1024x1024", "quality": "high",     "label": "square_high"},
    {"size": "1792x1024", "quality": "standard", "label": "landscape_standard"},
    {"size": "1792x1024", "quality": "high",     "label": "landscape_high"},
]

PROMPT_EX51 = (
    "A futuristic city skyline at dusk, where holographic displays project AI models "
    "communicating through light beams, photorealistic style."
)

timing_results = []
for cfg in CONFIGS:
    t0 = time.perf_counter()
    path = generate_image(
        prompt=PROMPT_EX51,
        out_path=OUTPUT_DIR / f"{cfg['label']}.png",
        size=cfg["size"],
        quality=cfg["quality"],
    )
    elapsed = time.perf_counter() - t0
    timing_results.append({**cfg, "time_s": round(elapsed, 2), "path": path.name})

print("\nEx 5.1 — Size/quality timing table:")
header = f"{'Label':<25} {'Size':<12} {'Quality':<10} {'Time(s)':>8}"
print(header)
print("-" * len(header))
for r in timing_results:
    print(f"{r['label']:<25} {r['size']:<12} {r['quality']:<10} {r['time_s']:>8.2f}")

print(
    "\nObservation:\n"
    "  'high' quality takes longer but produces more detailed, coherent images.\n"
    "  Landscape (1792x1024) is useful for banner/presentation assets.\n"
    "  For prototyping use 'standard' to save time and cost.\n"
)

# =============================================================================
# Ex 5.2 — Three different prompts for peer voting
# =============================================================================

log_step("Ex 5.2 — Three prompts for peer voting")

VOTE_PROMPTS = [
    {
        "label": "vote_1_abstract",
        "prompt": (
            "Abstract digital art representing neural networks as interconnected glowing nodes "
            "on a dark background, with flowing data streams between them, vibrant neon colors."
        ),
    },
    {
        "label": "vote_2_nature",
        "prompt": (
            "A tranquil mountain lake at sunrise reflecting an aurora borealis, "
            "photorealistic, ultra-high detail, serene atmosphere."
        ),
    },
    {
        "label": "vote_3_retro",
        "prompt": (
            "Retro-futuristic robot reading a book in a cozy 1970s living room, "
            "warm lighting, vintage illustration style, pastel color palette."
        ),
    },
]

print("Generating 3 candidate images for peer voting...")
for item in VOTE_PROMPTS:
    path = generate_image(
        prompt=item["prompt"],
        out_path=OUTPUT_DIR / f"{item['label']}.png",
        size="1024x1024",
        quality="standard",
    )
    log(f"  {item['label']} → {path.name}")

print(
    "\nEx 5.2 Instructions:\n"
    "  Open the three images from outputs/images/ and vote with your partner:\n"
    "    1. Which image best matches the prompt intent?\n"
    "    2. Which has the highest visual quality?\n"
    "    3. Which would you use in a real project?\n"
    "  Note your observations on prompt wording vs image outcome.\n"
)
