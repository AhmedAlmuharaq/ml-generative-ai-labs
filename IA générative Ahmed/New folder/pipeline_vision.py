"""
TP5 - Step 6 module: vision pipeline functions
Imported by tp5_pipeline.py and reusable across projects.
"""

import base64
import mimetypes
from pathlib import Path
from openai import OpenAI
from tp5_log import log_success

client = OpenAI()


def encode_image_to_data_url(image_path: str | Path) -> str:
    """Encode a local image as a base64 data URL for the OpenAI API."""
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
    max_tokens: int = 400,
) -> str:
    """
    Send an image + question to a vision model and return the text answer.

    Args:
        image_path: Path to a local PNG/JPG image.
        question:   Instruction or question about the image.
        model:      OpenAI model with vision capability.
        max_tokens: Maximum tokens in the response.

    Returns:
        The model's answer as a plain string.
    """
    data_url = encode_image_to_data_url(image_path)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text",  "text": question},
                    {"type": "input_image", "image_url": data_url},
                ],
            }
        ],
        temperature=0.1,
        max_output_tokens=max_tokens,
    )

    texts = [
        content.text.strip()
        for item in response.output
        for content in item.content
        if content.type == "output_text"
    ]
    result = "\n".join(texts)
    log_success("Vision analysis complete.")
    return result
