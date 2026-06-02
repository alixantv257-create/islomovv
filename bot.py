from __future__ import annotations

import asyncio
import logging
import os
import pathlib
import shutil
from typing import Optional, Tuple

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

import config
from dubbing import dub_media
from ffmpeg_utils import ensure_dir
from client_interface import SpeechClient
from fish_tts_client import FishTTSClient, FishTTSConfig
from naga_client import NagaClient, NagaClientConfig
from google_tts_client import GoogleTTSClient, GoogleTTSConfig
from hybrid_client import HybridClient
from local_speech_client import LocalSpeechClient


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

PREMIUM_EMOJIS = {
    "long_video": config.PREMIUM_EMOJI_LONG_VIDEO,
    "anime": config.PREMIUM_EMOJI_ANIME,
    "movie": config.PREMIUM_EMOJI_MOVIE,
    "shorts": config.PREMIUM_EMOJI_SHORTS,
}

FALLBACK_EMOJIS = {
    "long_video": "",
    "anime": "",
    "movie": "",
    "shorts": "",
}


def _valid_custom_emoji_id(value: Optional[str]) -> bool:
    if not value:
        return False
    return value.isdigit() and int(value) > 0


def _safe_filename(name: str) -> str:
    name = name.strip().replace("\\", "_").replace("/", "_")
    if not name:
        return "input"
    return name


def _detect_input_kind(msg) -> Tuple[Optional[object], Optional[str], Optional[str], Optional[int]]:
    video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
    audio_exts = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

    if msg.video:
        return msg.video, msg.video.file_name or "video.mp4", "video", msg.video.file_size

    if msg.document:
        filename = msg.document.file_name or "upload"
        ext = pathlib.Path(filename).suffix.lower()
        mime_type = (msg.document.mime_type or "").lower()
        if mime_type.startswith("video/") or ext in video_exts:
            return msg.document, filename, "video", msg.document.file_size
        if mime_type.startswith("audio/") or ext in audio_exts:
            return msg.document, filename, "audio", msg.document.file_size

    if msg.audio:
        return msg.audio, msg.audio.file_name or "audio.mp3", "audio", msg.audio.file_size

    return None, None, None, None


def _main_menu() -> InlineKeyboardMarkup:
    keyboard = [
        [
            _menu_button("Uzun Video Tarjima", "menu_long_video", "long_video"),
            _menu_button("Anime Tarjima", "menu_anime", "anime"),
        ],
        [
            _menu_button("Kino Tarjima", "menu_movie", "movie"),
            _menu_button("Shorts Video Tarjima", "menu_shorts", "shorts"),
        ],
    ]
    return InlineKeyboardMarkup(keyboard)


def _menu_button(text: str, callback_data: str, emoji_key: str) -> InlineKeyboardButton:
    emoji_id = PREMIUM_EMOJIS.get(emoji_key)
    fallback = FALLBACK_EMOJIS.get(emoji_key, "")
    label = f"{fallback} {text}".strip()
    if not _valid_custom_emoji_id(emoji_id):
        return InlineKeyboardButton(label, callback_data=callback_data)
    try:
        return InlineKeyboardButton(
            label,
            callback_data=callback_data,
            icon_custom_emoji_id=emoji_id,
        )
    except TypeError:
        return InlineKeyboardButton(
            label,
            callback_data=callback_data,
            api_kwargs={"icon_custom_emoji_id": emoji_id},
        )


def _emoji_tag(emoji_id: str, fallback: str) -> str:
    if not _valid_custom_emoji_id(emoji_id):
        return fallback
    return f'<tg-emoji emoji-id="{emoji_id}">{fallback}</tg-emoji>'


def _menu_text(key: str) -> str:
    texts = {
        "menu_long_video": (
            f"{_emoji_tag(PREMIUM_EMOJIS['long_video'], FALLBACK_EMOJIS['long_video'])} Uzun video tarjima tanlandi.\n\n"
            "Menga uzun video yuboring. Men odam ovozlarini uzb tilga dub qilaman va fonni saqlab qolaman."
        ),
        "menu_anime": (
            f"{_emoji_tag(PREMIUM_EMOJIS['anime'], FALLBACK_EMOJIS['anime'])} Anime tarjima tanlandi.\n\n"
            "Anime video yuboring. Men dialoglarni ajratib, uzb dub variantini tayyorlayman."
        ),
        "menu_movie": (
            f"{_emoji_tag(PREMIUM_EMOJIS['movie'], FALLBACK_EMOJIS['movie'])} Kino tarjima tanlandi.\n\n"
            "Kino video yuboring. Men odam ovozlarini uzb tilga o‘girib, qayta miks qilaman."
        ),
        "menu_shorts": (
            f"{_emoji_tag(PREMIUM_EMOJIS['shorts'], FALLBACK_EMOJIS['shorts'])} Shorts video tarjima tanlandi.\n\n"
            "Qisqa video yuboring. Men tezroq dub qilib, tayyor natijani qaytaraman."
        ),
    }
    return texts.get(
        key,
        "Video yoki audio yuboring. Men uni uzb dub qilib beraman.",
    )


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        logger.info("Received /start from chat_id=%s", update.effective_chat.id)
    await update.message.reply_text(
        "Assalomu aleykum "
        f"{_emoji_tag(PREMIUM_EMOJIS['anime'], FALLBACK_EMOJIS['anime'])}\n\n"
        "Kerakli bo‘limni tanlang yoki to‘g‘ridan-to‘g‘ri video yuboring.",
        reply_markup=_main_menu(),
        parse_mode=ParseMode.HTML,
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.effective_chat:
        logger.info("Received /help from chat_id=%s", update.effective_chat.id)
    await update.message.reply_text(
        "Assalomu aleykum "
        f"{_emoji_tag(PREMIUM_EMOJIS['long_video'], FALLBACK_EMOJIS['long_video'])}\n\n"
        "Quyidagi tugmalardan birini tanlang yoki video yuboring.\n"
        "Bot: dialogni ajratadi → transkripsiya → uzb tarjima → TTS → fon bilan miks → natijani qaytaradi.",
        reply_markup=_main_menu(),
        parse_mode=ParseMode.HTML,
    )


async def handle_menu_click(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    if not query:
        return

    if update.effective_chat:
        logger.info(
            "Menu clicked: chat_id=%s callback=%s",
            update.effective_chat.id,
            query.data,
        )
    await query.answer()
    await query.message.reply_text(
        _menu_text(query.data),
        reply_markup=_main_menu(),
        parse_mode=ParseMode.HTML,
    )


def _build_client() -> SpeechClient:
    # ── LOCAL mode: Fish Speech 1.5 (S2 open-source) + Whisper + NLLB ────────
    if config.TTS_PROVIDER == "local":
        logger.info(
            "Using LOCAL mode: Fish Speech 1.5 + Whisper (%s) + NLLB-200 "
            "| device=%s | model_dir=%s",
            config.LOCAL_WHISPER_SIZE,
            config.LOCAL_DEVICE,
            config.FISH_LOCAL_MODEL_DIR,
        )
        return LocalSpeechClient(
            model_dir=config.FISH_LOCAL_MODEL_DIR,
            whisper_size=config.LOCAL_WHISPER_SIZE,
            device=config.LOCAL_DEVICE,
            translator_cache_dir=config.LOCAL_TRANSLATOR_CACHE or None,
        )

    # ── Fish Audio API + Naga hybrid ─────────────────────────────────
    if config.TTS_PROVIDER == "fish" and config.FISH_API_KEY:
        # If a Naga API key is available use the full HybridClient (Naga+Fish).
        # Otherwise create a small composite client that uses local ASR/translator
        # (faster-whisper + NLLB) together with Fish cloud TTS so we avoid
        # constructing NagaClient when it's not configured.
        fish_cfg = FishTTSConfig(
            api_key=config.FISH_API_KEY,
            base_url=config.FISH_BASE_URL,
            model=config.FISH_TTS_MODEL,
            default_reference_id=config.FISH_TTS_DEFAULT_REFERENCE_ID,
        )
        if config.NAGA_API_KEY:
            naga_cfg = NagaClientConfig(
                base_url=config.NAGA_BASE_URL,
                api_key=config.NAGA_API_KEY,
                asr_model=config.NAGA_ASR_MODEL,
                text_model=config.NAGA_TEXT_MODEL,
                tts_model=config.NAGA_TTS_MODEL,
                tts_voice=config.NAGA_TTS_VOICE,
                tts_response_format=config.NAGA_TTS_RESPONSE_FORMAT,
            )
            return HybridClient(naga_cfg, fish_config=fish_cfg)

        # Composite client: local ASR/translator + Fish cloud TTS
        from local_speech_client import _LocalWhisper
        from local_translator import LocalTranslator

        class _LocalAsrFishTts(SpeechClient):
            def __init__(self) -> None:
                self._wh = _LocalWhisper.get(config.LOCAL_WHISPER_SIZE)
                self._translator = LocalTranslator.get_instance(config.LOCAL_TRANSLATOR_CACHE or None)
                self._tts = FishTTSClient(fish_cfg)

            def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
                return self._wh.transcribe(wav_path, language=language)

            def translate_to_uzbek(self, text: str) -> str:
                return self._translator.translate(text)

            def tts(self, text: str, **kwargs) -> bytes:
                # forward kwargs (speed, voice, instructions, emotion, reference_audio, reference_text)
                return self._tts.tts(text, **kwargs)

        return _LocalAsrFishTts()
    if config.GOOGLE_API_KEY:
        # HYBRID: NAGA for ASR+Translation, GOOGLE for TTS!
        naga_cfg = NagaClientConfig(
            base_url=config.NAGA_BASE_URL,
            api_key=config.NAGA_API_KEY,
            asr_model=config.NAGA_ASR_MODEL,
            text_model=config.NAGA_TEXT_MODEL,
            tts_model=config.NAGA_TTS_MODEL,
            tts_voice=config.NAGA_TTS_VOICE,
            tts_response_format=config.NAGA_TTS_RESPONSE_FORMAT,
        )
        google_cfg = GoogleTTSConfig(
            api_key=config.GOOGLE_API_KEY,
            voice=config.GEMINI_TTS_VOICE,
            tts_model=config.GEMINI_TTS_MODEL,
        )
        return HybridClient(naga_cfg, google_cfg)
    else:
        # Fallback to Naga if no Google key
        cfg = NagaClientConfig(
            base_url=config.NAGA_BASE_URL,
            api_key=config.NAGA_API_KEY,
            asr_model=config.NAGA_ASR_MODEL,
            text_model=config.NAGA_TEXT_MODEL,
            tts_model=config.NAGA_TTS_MODEL,
            tts_voice=config.NAGA_TTS_VOICE,
            tts_response_format=config.NAGA_TTS_RESPONSE_FORMAT,
        )
        return NagaClient(cfg)


def _tts_voice_settings() -> Tuple[list[str], str]:
    if config.TTS_PROVIDER == "fish" and config.FISH_API_KEY:
        voices = [f"boy:{value}" for value in config.FISH_TTS_BOY_MODELS]
        voices += [f"girl:{value}" for value in config.FISH_TTS_GIRL_MODELS]
        voices += [f"man:{value}" for value in config.FISH_TTS_MAN_MODELS]
        voices += [f"woman:{value}" for value in config.FISH_TTS_WOMAN_MODELS]
        voices += [f"old:{value}" for value in config.FISH_TTS_OLD_MODELS]
        return voices, config.FISH_TTS_DEFAULT_REFERENCE_ID
    if config.GOOGLE_API_KEY and config.TTS_PROVIDER in {"google", "gemini"}:
        voices = [f"m:{voice}" for voice in config.GEMINI_TTS_MALE_VOICES]
        voices += [f"f:{voice}" for voice in config.GEMINI_TTS_FEMALE_VOICES]
        return voices, config.GEMINI_TTS_VOICE
    return config.NAGA_TTS_VOICES, config.NAGA_TTS_VOICE


async def handle_media(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    msg = update.message
    if not msg:
        return

    media, filename, input_kind, file_size = _detect_input_kind(msg)
    logger.info(
        "Incoming media: chat_id=%s message_id=%s kind=%s filename=%s size=%s",
        msg.chat_id,
        msg.message_id,
        input_kind,
        filename,
        file_size,
    )

    if not media:
        logger.warning("Unsupported message payload: chat_id=%s message_id=%s", msg.chat_id, msg.message_id)
        await msg.reply_text("Video yoki audio yuboring.")
        return

    if file_size and file_size > config.MAX_DOWNLOAD_MB * 1024 * 1024:
        logger.warning(
            "File too large: chat_id=%s message_id=%s size=%s limit_mb=%s",
            msg.chat_id,
            msg.message_id,
            file_size,
            config.MAX_DOWNLOAD_MB,
        )
        await msg.reply_text(f"Fayl katta. Limit: {config.MAX_DOWNLOAD_MB} MB")
        return

    status = await msg.reply_text("Qabul qilindi. Ishlayapman...")

    user_dir = os.path.join(config.WORK_DIR, str(msg.chat_id), str(msg.message_id))
    ensure_dir(user_dir)
    logger.info("Created work dir: %s", user_dir)

    try:
        ext = pathlib.Path(filename).suffix or ".bin"
        local_in = os.path.join(user_dir, _safe_filename("input") + ext)

        tg_file = await media.get_file()
        await tg_file.download_to_drive(custom_path=local_in)
        logger.info("Downloaded media to %s", local_in)

        client = _build_client()
        tts_voices, default_tts_voice = _tts_voice_settings()
        profile_store_path = os.path.join(config.WORK_DIR, str(msg.chat_id), "speaker_profiles.json")
        logger.info("Starting dubbing pipeline for chat_id=%s message_id=%s", msg.chat_id, msg.message_id)

        def _work() -> Tuple[str, str]:
            return dub_media(
                client=client,
                input_path=local_in,
                work_dir=user_dir,
                has_video=input_kind == "video",
                available_voices=tts_voices,
                default_voice=default_tts_voice,
                max_speakers=config.MAX_SPEAKERS,
                profile_store_path=profile_store_path,
                vad_aggressiveness=config.VAD_AGGRESSIVENESS,
                max_segment_ms=config.SEGMENT_MAX_MS,
                min_segment_ms=config.SEGMENT_MIN_MS,
                speaker_match_threshold=config.SPEAKER_MATCH_THRESHOLD,
                tts_min_playback_ratio=config.TTS_MIN_PLAYBACK_RATIO,
                tts_max_playback_ratio=config.TTS_MAX_PLAYBACK_RATIO,
                tts_split_max_chars=config.TTS_SPLIT_MAX_CHARS,
                tts_chunk_pause_ms=config.TTS_CHUNK_PAUSE_MS,
            )

        try:
            out_path, output_kind = await asyncio.to_thread(_work)
        except Exception as e:
            logger.exception(
                "Dubbing failed: chat_id=%s message_id=%s error=%s",
                msg.chat_id,
                msg.message_id,
                e,
            )
            await status.edit_text(f"Xatolik: {e}")
            return

        logger.info(
            "Dubbing finished: chat_id=%s message_id=%s output_kind=%s output_path=%s",
            msg.chat_id,
            msg.message_id,
            output_kind,
            out_path,
        )
        await status.edit_text("Tayyor. Yuboryapman...")
        with open(out_path, "rb") as f:
            if output_kind == "video":
                await msg.reply_video(video=f)
            else:
                await msg.reply_audio(audio=f)
        await status.edit_text("Tayyor")
        logger.info("Result sent: chat_id=%s message_id=%s", msg.chat_id, msg.message_id)
    finally:
        # Cleanup to prevent disk space exhaustion
        if os.path.exists(user_dir):
            try:
                shutil.rmtree(user_dir)
                logger.info("Cleaned up work dir: %s", user_dir)
            except Exception as e:
                logger.error("Failed to clean up %s: %s", user_dir, e)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.exception("Unhandled telegram error: %s", context.error)


def main() -> None:
    ensure_dir(config.WORK_DIR)
    logger.info("Bot is starting. work_dir=%s", config.WORK_DIR)

    app = Application.builder().token(config.TELEGRAM_BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CallbackQueryHandler(handle_menu_click, pattern=r"^menu_"))
    app.add_handler(
        MessageHandler(filters.VIDEO | filters.Document.ALL | filters.AUDIO, handle_media)
    )
    app.add_error_handler(on_error)
    logger.info("Polling started")
    app.run_polling()


if __name__ == "__main__":
    main()
