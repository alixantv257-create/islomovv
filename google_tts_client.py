from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass
from typing import Optional

from client_interface import SpeechClient

# Emotion mapping for perfect anime dubbing
EMOTION_CONFIG = {
    "gazab": {
        "speaking_rate": 1.25,
        "pitch": 3.0,
        "style": "angry",
        "instructions": "Speak with intense anger, raised voice, sharp and forceful delivery"
    },
    "qaygu": {
        "speaking_rate": 0.80,
        "pitch": -2.5,
        "style": "sad",
        "instructions": "Speak with deep sadness, slow and deliberate pace, somber tone"
    },
    "yiglash": {
        "speaking_rate": 0.75,
        "pitch": -3.5,
        "style": "sad",
        "instructions": "Speak while crying, voice cracking, emotional sobbing between words"
    },
    "qorquv": {
        "speaking_rate": 1.30,
        "pitch": 3.5,
        "style": "scared",
        "instructions": "Speak with fear, trembling voice, quick and breathless delivery"
    },
    "hayrat": {
        "speaking_rate": 1.15,
        "pitch": 2.0,
        "style": "excited",
        "instructions": "Speak with surprise and amazement, elevated pitch, enthusiastic delivery"
    },
    "xursandchilik": {
        "speaking_rate": 1.05,
        "pitch": 1.0,
        "style": "friendly",
        "instructions": "Speak with happiness and joy, bright and cheerful tone"
    },
    "kulish": {
        "speaking_rate": 1.0,
        "pitch": 0.5,
        "style": "friendly",
        "instructions": "Speak with laughter, playful tone, light and humorous delivery"
    },
    "sovuqqonlik": {
        "speaking_rate": 0.90,
        "pitch": -1.5,
        "style": "neutral",
        "instructions": "Speak with coldness and detachment, calm and steady tone"
    },
    "qahramonona": {
        "speaking_rate": 1.0,
        "pitch": 0.8,
        "style": "confident",
        "instructions": "Speak heroically, powerful and confident tone, inspiring delivery"
    },
    "yovuz_qahramon": {
        "speaking_rate": 0.95,
        "pitch": -2.0,
        "style": "angry",
        "instructions": "Speak villainously, dark and menacing tone, slow and deliberate"
    }
}


@dataclass(frozen=True)
class GoogleTTSConfig:
    api_key: str
    voice: str = "Kore"
    tts_model: str = "gemini-2.5-flash-preview-tts"


class GoogleTTSClient(SpeechClient):
    """
    Complete Google Cloud client:
    - ASR (Speech-to-Text): Google Speech-to-Text (fallback to whisper if needed)
    - Translation: Google Translate API
    - TTS (Text-to-Speech): Google Cloud TTS with emotions
    """
    
    def __init__(self, cfg: GoogleTTSConfig):
        self._cfg = cfg
        
    def transcribe(self, wav_path: str, language: Optional[str] = None) -> str:
        try:
            return self._call_google_stt(wav_path, language)
        except Exception as e:
            print(f"Google STT failed: {str(e)}, falling back to local whisper")
            try:
                return self._fallback_whisper(wav_path, language)
            except Exception as e2:
                print(f"Whisper also failed: {str(e2)}, returning empty")
                return ""
        
    def translate_to_uzbek(self, text: str) -> str:
        try:
            return self._call_google_translate(text)
        except Exception as e:
            print(f"Google Translate failed: {str(e)}, returning original text")
            return text
        
    def tts(
        self,
        text: str,
        *,
        speed: Optional[float] = None,
        voice: Optional[str] = None,
        instructions: Optional[str] = None,
        emotion: Optional[str] = None,
        reference_audio: Optional[bytes] = None,  # ignored by Google TTS
        reference_text: str = "",
    ) -> bytes:
        if not text.strip():
            return b""

        # Get emotion settings
        emotion_settings = EMOTION_CONFIG.get(emotion, {})
        final_speed = speed if speed is not None else emotion_settings.get("speaking_rate", 1.0)
        final_pitch = emotion_settings.get("pitch", 0.0)
        try:
            return self._call_gemini_preview_tts(
                text=text,
                voice_name=voice or self._cfg.voice,
                speed=final_speed,
                pitch=final_pitch,
                instructions=instructions,
                emotion=emotion,
            )
        except Exception as e:
            print(f"Gemini preview TTS failed: {str(e)}")
            raise

    def _call_google_stt(self, wav_path: str, language: Optional[str] = None) -> str:
        import requests
        
        # First read and encode the audio file
        with open(wav_path, "rb") as f:
            audio_content = base64.b64encode(f.read()).decode("utf-8")
        
        url = f"https://speech.googleapis.com/v1/speech:recognize?key={self._cfg.api_key}"
        
        request_data = {
            "config": {
                "encoding": "LINEAR16",
                "sampleRateHertz": 16000,
                "languageCode": language or "auto",
                "audioChannelCount": 1,
                "enableWordTimeOffsets": False
            },
            "audio": {
                "content": audio_content
            }
        }
        
        response = requests.post(url, json=request_data, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        if "results" in result and len(result["results"]) > 0:
            return result["results"][0]["alternatives"][0]["transcript"]
        return ""

    def _fallback_whisper(self, wav_path: str, language: Optional[str] = None) -> str:
        try:
            import whisper
            model = whisper.load_model("base")  # Small, fast model
            result = model.transcribe(wav_path, language=language)
            return result["text"].strip()
        except ImportError:
            raise RuntimeError("Whisper not installed, run: pip install openai-whisper")

    def _call_google_translate(self, text: str) -> str:
        import requests
        
        url = f"https://translation.googleapis.com/language/translate/v2?key={self._cfg.api_key}"
        
        request_data = {
            "q": text,
            "target": "uz",
            "format": "text"
        }
        
        response = requests.post(url, json=request_data, timeout=30)
        response.raise_for_status()
        
        result = response.json()
        if "data" in result and "translations" in result["data"] and len(result["data"]["translations"]) > 0:
            return result["data"]["translations"][0]["translatedText"]
        return text

    def _call_gemini_preview_tts(
        self,
        *,
        text: str,
        voice_name: str,
        speed: float,
        pitch: float,
        instructions: Optional[str],
        emotion: Optional[str],
    ) -> bytes:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self._cfg.api_key)
        prompt = self._build_preview_prompt(
            text=text,
            voice_name=voice_name,
            speed=speed,
            pitch=pitch,
            instructions=instructions,
            emotion=emotion,
        )
        response = client.models.generate_content(
            model=self._cfg.tts_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=voice_name
                        )
                    )
                ),
            ),
        )

        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise RuntimeError("Gemini TTS returned no candidates")
        parts = getattr(candidates[0].content, "parts", None) or []
        if not parts or not getattr(parts[0], "inline_data", None):
            raise RuntimeError("Gemini TTS returned no audio data")
        audio_data = parts[0].inline_data.data
        if not audio_data:
            raise RuntimeError("Gemini TTS audio data is empty")
        return self._pcm_to_wav(audio_data, sample_rate=24000)

    def _build_preview_prompt(
        self,
        *,
        text: str,
        voice_name: str,
        speed: float,
        pitch: float,
        instructions: Optional[str],
        emotion: Optional[str],
    ) -> str:
        emotion_settings = EMOTION_CONFIG.get(emotion, {})
        persona = self._voice_persona(voice_name)
        style_parts = [persona, "anime dubbing", "real human acting", "natural Uzbek delivery"]
        if instructions:
            style_parts.append(instructions)
        elif emotion_settings.get("instructions"):
            style_parts.append(str(emotion_settings["instructions"]))
        style_parts.append(self._speed_hint(speed))
        if pitch >= 2.0:
            style_parts.append("slightly brighter tone")
        elif pitch <= -1.5:
            style_parts.append("slightly deeper tone")
        tag = " | ".join(part for part in style_parts if part)
        return f"[{tag}]\n{text.strip()}"

    def _voice_persona(self, voice_name: str) -> str:
        female_voices = {"Kore", "Aoede", "Leda", "Callirrhoe", "Autonoe"}
        male_voices = {"Puck", "Charon", "Fenrir", "Orus", "Iapetus", "Enceladus"}
        if voice_name in female_voices:
            return "emotional anime girl"
        if voice_name in male_voices:
            return "emotional anime boy"
        return "emotional anime character"

    def _speed_hint(self, speed: float) -> str:
        if speed >= 1.2:
            return "speak fast with clear articulation"
        if speed >= 1.05:
            return "speak slightly faster than normal"
        if speed <= 0.82:
            return "speak slowly and emotionally"
        if speed <= 0.95:
            return "speak slightly slower than normal"
        return "speak at natural conversational speed"

    def _pcm_to_wav(self, audio_data: bytes, *, sample_rate: int) -> bytes:
        wav_bytes = io.BytesIO()
        with wave.open(wav_bytes, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_data)
        wav_bytes.seek(0)
        return wav_bytes.read()
