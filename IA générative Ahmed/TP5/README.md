# TP5 — Multimodality and Generative AI

## Objectives
- Set up audio pipelines: Text-to-Speech (TTS) and Speech-to-Text (STT).
- Analyse images with vision models (gpt-4.1-mini, gpt-4.1).
- Generate images from text prompts.
- Compose a full multimodal pipeline: image → analysis → (translation) → audio.

## File
| File | Description |
|------|-------------|
| `tp5_multimodal.ipynb` | Complete notebook: all steps and exercises |

## Setup
```bash
pip install openai python-dotenv pydub pillow rich
```
Also install **ffmpeg** and add it to your PATH (required by pydub for audio).

Add to `../.env`:
```
OPENAI_API_KEY=your_key
```

## Folder structure needed
```
TP5/
  inputs/
    audio/meeting.mp3     <- for Step 3 (STT)
    images/*.png|jpg      <- for Steps 4 and 6
  outputs/
    audio/                <- auto-created
    images/               <- auto-created
    transcriptions/       <- auto-created
    vision/               <- auto-created
```

## Steps covered
| Step | Topic | Exercises |
|------|-------|-----------|
| 2 | Text-to-Speech | 2.1 (voice comparison), 2.2 (language variation) |
| 3 | Speech-to-Text | 3.1 (chunk timing), 3.2 (timeline JSON) |
| 4 | Vision | 4.1 (mini vs full model), 4.2 (format constraints) |
| 5 | Image generation | 5.1 (size/quality), 5.2 (peer voting) |
| 6 | Full pipeline | 6.1 (translation step), 6.2 (CLI interface) |
