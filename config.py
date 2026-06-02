import os

from dotenv import load_dotenv


load_dotenv(override=False)


# By default run in fully-local mode to avoid external API usage.
# Set TTS_PROVIDER to 'naga'/'google'/'fish' to enable cloud providers.
TTS_PROVIDER = os.getenv("TTS_PROVIDER", "local").strip().lower()


def _must_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


TELEGRAM_BOT_TOKEN = _must_env("TELEGRAM_BOT_TOKEN")
NAGA_API_KEY = os.getenv("NAGA_API_KEY", "")
if TTS_PROVIDER != "local" and not NAGA_API_KEY:
    raise RuntimeError("Missing required environment variable: NAGA_API_KEY")

NAGA_BASE_URL = os.getenv("NAGA_BASE_URL", "https://api.naga.ac/v1")

NAGA_ASR_MODEL = os.getenv("NAGA_ASR_MODEL", "whisper-large-v3:free")
NAGA_TEXT_MODEL = os.getenv("NAGA_TEXT_MODEL", "glm-4.5-air:free")
NAGA_TTS_MODEL = os.getenv("NAGA_TTS_MODEL", "gpt-4o-mini-tts:free")
NAGA_TTS_VOICE = os.getenv("NAGA_TTS_VOICE", "alloy")
NAGA_TTS_RESPONSE_FORMAT = os.getenv("NAGA_TTS_RESPONSE_FORMAT", "wav")
NAGA_TTS_VOICES = [
    voice.strip()
    for voice in os.getenv(
        "NAGA_TTS_VOICES",
        "alloy,ash,ballad,coral,echo,sage,shimmer,verse",
    ).split(",")
    if voice.strip()
]

MAX_DOWNLOAD_MB = int(os.getenv("MAX_DOWNLOAD_MB", "100"))
MAX_SPEAKERS = int(os.getenv("MAX_SPEAKERS", "6"))
VAD_AGGRESSIVENESS = int(os.getenv("VAD_AGGRESSIVENESS", "3"))
SEGMENT_MAX_MS = int(os.getenv("SEGMENT_MAX_MS", "6500"))
SEGMENT_MIN_MS = int(os.getenv("SEGMENT_MIN_MS", "1800"))
SPEAKER_MATCH_THRESHOLD = float(os.getenv("SPEAKER_MATCH_THRESHOLD", "0.055"))
TTS_MIN_PLAYBACK_RATIO = float(os.getenv("TTS_MIN_PLAYBACK_RATIO", "0.85"))
TTS_MAX_PLAYBACK_RATIO = float(os.getenv("TTS_MAX_PLAYBACK_RATIO", "1.60"))
TTS_SPLIT_MAX_CHARS = int(os.getenv("TTS_SPLIT_MAX_CHARS", "220"))
TTS_CHUNK_PAUSE_MS = int(os.getenv("TTS_CHUNK_PAUSE_MS", "80"))

WORK_DIR = os.getenv("WORK_DIR", os.path.join(os.getcwd(), "work"))

PREMIUM_EMOJI_LONG_VIDEO = os.getenv("PREMIUM_EMOJI_LONG_VIDEO", "6005986106703613755")
PREMIUM_EMOJI_ANIME = os.getenv("PREMIUM_EMOJI_ANIME", "5321020607758888660")
PREMIUM_EMOJI_MOVIE = os.getenv("PREMIUM_EMOJI_MOVIE", "5375464961822695044")
PREMIUM_EMOJI_SHORTS = os.getenv("PREMIUM_EMOJI_SHORTS", "5334681713316479679")

# TTS Provider options: naga, google, gemini, fish, local

# Google / Gemini AI Studio TTS
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_TTS_MODEL = os.getenv("GEMINI_TTS_MODEL", "gemini-2.5-flash-preview-tts")
GEMINI_TTS_VOICE = os.getenv("GEMINI_TTS_VOICE", "Kore")
GEMINI_TTS_MALE_VOICES = [
    voice.strip()
    for voice in os.getenv(
        "GEMINI_TTS_MALE_VOICES",
        "Puck,Charon,Fenrir,Orus,Iapetus,Enceladus",
    ).split(",")
    if voice.strip()
]
GEMINI_TTS_FEMALE_VOICES = [
    voice.strip()
    for voice in os.getenv(
        "GEMINI_TTS_FEMALE_VOICES",
        "Kore,Aoede,Leda,Callirrhoe,Autonoe",
    ).split(",")
    if voice.strip()
]

# Fish Audio TTS
FISH_API_KEY = os.getenv("FISH_API_KEY", "")
FISH_BASE_URL = os.getenv("FISH_BASE_URL", "https://api.fish.audio")
FISH_TTS_MODEL = os.getenv("FISH_TTS_MODEL", "s2-pro")
FISH_TTS_DEFAULT_REFERENCE_ID = os.getenv("FISH_TTS_DEFAULT_REFERENCE_ID", "")
FISH_TTS_BOY_MODELS = [
    value.strip()
    for value in os.getenv("FISH_TTS_BOY_MODELS", "").split(",")
    if value.strip()
]
FISH_TTS_GIRL_MODELS = [
    value.strip()
    for value in os.getenv("FISH_TTS_GIRL_MODELS", "").split(",")
    if value.strip()
]
FISH_TTS_MAN_MODELS = [
    value.strip()
    for value in os.getenv("FISH_TTS_MAN_MODELS", "").split(",")
    if value.strip()
]
FISH_TTS_WOMAN_MODELS = [
    value.strip()
    for value in os.getenv("FISH_TTS_WOMAN_MODELS", "").split(",")
    if value.strip()
]
FISH_TTS_OLD_MODELS = [
    value.strip()
    for value in os.getenv("FISH_TTS_OLD_MODELS", "").split(",")
    if value.strip()
]

# ── Local mode (fully offline) ────────────────────────────────────────────────
# Set TTS_PROVIDER=local to activate Fish Speech 1.5 (S2 open-source) +
# local Whisper ASR + local NLLB-200 translation.
# Everything is auto-installed and downloaded on first run.

# Where Fish Speech 1.5 model weights will be stored
FISH_LOCAL_MODEL_DIR = os.getenv(
    "FISH_LOCAL_MODEL_DIR",
    os.path.join(os.getcwd(), "models", "fish-speech-1.5"),
)
FISH_LOCAL_HF_REPO = os.getenv("FISH_LOCAL_HF_REPO", "fishaudio/fish-speech-1.5")

# Local NLLB translation model: 1.3B is high quality and fits in 3-4 GB VRAM
LOCAL_NLLB_MODEL = os.getenv("LOCAL_NLLB_MODEL", "facebook/nllb-200-distilled-1.3B")

# Whisper model size: tiny / base / small / medium / large-v3 (default)
# Larger = better accuracy, slower, more VRAM
LOCAL_WHISPER_SIZE = os.getenv("LOCAL_WHISPER_SIZE", "large-v3")

# Inference device: auto (GPU if available, else CPU) | cuda | cpu
LOCAL_DEVICE = os.getenv("LOCAL_DEVICE", "auto")

# Optional separate cache dir for NLLB-200 translation model (~1.5 GB)
LOCAL_TRANSLATOR_CACHE = os.getenv("LOCAL_TRANSLATOR_CACHE", "")

# ── Dubbing quality / performance tunables ────────────────────────────────────
# Apply noise reduction to speaker reference audio before cloning
# (noisereduce library, CPU-friendly spectral gating)
REFERENCE_NOISE_REDUCE = os.getenv("REFERENCE_NOISE_REDUCE", "1").strip().lower() not in {"0", "false", "no", "off"}

# Normalize the dialog track to broadcast loudness (-16 LUFS) using pyloudnorm
# Disable for faster processing if you don't care about final volume consistency
DIALOG_LOUDNESS_NORMALIZE = os.getenv("DIALOG_LOUDNESS_NORMALIZE", "1").strip().lower() not in {"0", "false", "no", "off"}
DIALOG_TARGET_LUFS = float(os.getenv("DIALOG_TARGET_LUFS", "-16.0"))

# Use silence-aware splitting inside long segments (recommended)
SPLIT_AT_SILENCE = os.getenv("SPLIT_AT_SILENCE", "1").strip().lower() not in {"0", "false", "no", "off"}

# Use agglomerative clustering for speaker diarization (more stable than threshold)
USE_AGGLOMERATIVE_SPEAKER_CLUSTERING = os.getenv(
    "USE_AGGLOMERATIVE_SPEAKER_CLUSTERING", "1"
).strip().lower() not in {"0", "false", "no", "off"}

# Number of parallel TTS worker threads (1 = sequential).
# Fish Speech server is single-process, but pipelining requests across
# segments can still overlap network I/O with synthesis.
TTS_PARALLEL_WORKERS = int(os.getenv("TTS_PARALLEL_WORKERS", "2"))

# Max segments per NLLB translation batch (longer batches = more VRAM but faster)
TRANSLATE_BATCH_SIZE = int(os.getenv("TRANSLATE_BATCH_SIZE", "8"))

# Auto-detect emotion per-segment from audio features (prosody heuristics)
AUTO_DETECT_EMOTION = os.getenv("AUTO_DETECT_EMOTION", "1").strip().lower() not in {"0", "false", "no", "off"}

# Fetch word-level timestamps from Whisper for tempo matching
USE_WORD_TIMESTAMPS = os.getenv("USE_WORD_TIMESTAMPS", "1").strip().lower() not in {"0", "false", "no", "off"}

# Max number of reference segments to combine for voice cloning
REFERENCE_MAX_SEGMENTS = int(os.getenv("REFERENCE_MAX_SEGMENTS", "4"))

# Minimum duration (seconds) for a reference segment to be considered usable
REFERENCE_MIN_SECONDS = float(os.getenv("REFERENCE_MIN_SECONDS", "3.0"))

# Minimum voiced-frame ratio for a reference to be considered usable (0-1)
REFERENCE_MIN_VOICED_RATIO = float(os.getenv("REFERENCE_MIN_VOICED_RATIO", "0.3"))

# Minimum SNR (dB) for a reference to be considered usable
REFERENCE_MIN_SNR_DB = float(os.getenv("REFERENCE_MIN_SNR_DB", "6.0"))
