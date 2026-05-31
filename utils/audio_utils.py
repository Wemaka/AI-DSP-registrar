"""
utils/audio_utils.py
Речевая аналитика (Требование 4):
  STT — openai-whisper (локально)
  TTS — gTTS (синтез речи)
"""

import os
import io
import tempfile
import base64
import whisper

from pathlib import Path
from gtts import gTTS


def transcribe_audio(audio_bytes: bytes, language: str = "ru") -> dict:
    """
    Распознаёт речь из аудиофайла с помощью Whisper.
    Возвращает: {"text": str, "language": str, "error": str|None}
    """
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        model = whisper.load_model("base")
        result = whisper.transcribe(model, tmp_path, language=language, fp16=False)
        os.unlink(tmp_path)

        return {
            "text": result.get("text", "").strip(),
            "language": result.get("language", language),
            "error": None,
        }
    except ImportError:
        # Whisper не установлен — возвращаем заглушку для демо
        return {
            "text": "",
            "language": language,
            "error": "openai-whisper не установлен. Установите: pip install openai-whisper",
        }
    except Exception as e:
        return {"text": "", "language": language, "error": str(e)}


def text_to_speech(text: str, language: str = "ru") -> bytes:
    """
    Синтезирует речь из текста с помощью gTTS.
    Возвращает байты MP3.
    """
    try:
        tts = gTTS(text=text[:500], lang=language, slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except ImportError:
        return b""
    except Exception:
        return b""


def audio_bytes_to_base64(audio_bytes: bytes) -> str:
    return base64.b64encode(audio_bytes).decode("utf-8")


def get_audio_html_player(audio_bytes: bytes, mime: str = "audio/mp3") -> str:
    """Возвращает HTML-тег <audio> для воспроизведения в Streamlit."""
    if not audio_bytes:
        return ""
    b64 = audio_bytes_to_base64(audio_bytes)
    return f'<audio autoplay controls><source src="data:{mime};base64,{b64}" type="{mime}"></audio>'
