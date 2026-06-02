Local voice-cloning + dubbing demo
=================================

This repository includes a local demo to test voice cloning for a single reference audio.

Prerequisites
-------------
- Python 3.10+
- FFmpeg installed and available on PATH (for full dubbing pipeline)
- Enough disk space (several GB) and optionally a GPU for model inference

Quick steps
-----------
1. Install Python requirements:

```bash
pip install -r requirements.txt
```

2. Copy `.env.example` → `.env` and edit values (set `TTS_PROVIDER=local`)

3. Place a clean reference WAV file (mono/stereo OK) named `reference.wav`.

4. Run the demo to synthesize text using the reference voice:

```bash
python scripts/run_local_demo.py reference.wav "Assalomu alaykum" out.wav
```

Notes
-----
- The first run may install `fish-speech` and download model weights — this can take a long time.

- This repository now defaults to fully-local mode (`TTS_PROVIDER=local`) to avoid using external APIs (Naga/Google/Fish cloud). To re-enable cloud providers, set `TTS_PROVIDER` to `naga`, `google`, or `fish` and provide the corresponding API keys in your `.env`.
- For full video dubbing you can run the bot or call `dubbing.dub_media()`; that requires `demucs` and `ffmpeg`.
- Cloning quality depends on reference length and clarity — longer clean speech yields better results.
- Respect privacy and obtain permission before cloning real people's voices.
