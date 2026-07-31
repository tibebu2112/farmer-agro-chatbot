"""
Voice transcription helpers for the farmer chatbot.
Uses Google Speech Recognition (free) via the SpeechRecognition library.
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path
from typing import Optional

import speech_recognition as sr


def transcribe_audio_file(
    audio_bytes: bytes,
    language: str = "en-US",
) -> tuple[Optional[str], Optional[str]]:
    """
    Transcribe audio bytes (wav/webm/ogg from browser) to text.
    Returns (text, error_message).
    """
    if not audio_bytes:
        return None, "No audio received."

    recognizer = sr.Recognizer()

    try:
        from pydub import AudioSegment
    except ImportError:
        return None, "pydub is required for voice input. Install with: pip install pydub"

    temp_dir = Path(tempfile.gettempdir())
    input_path = temp_dir / "farmer_input_audio"
    wav_path = temp_dir / "farmer_input_audio.wav"

    # Try common browser/container formats.
    audio_segment = None
    last_error = None
    for suffix in (".webm", ".wav", ".ogg", ".mp3", ".m4a"):
        try:
            candidate = input_path.with_suffix(suffix)
            candidate.write_bytes(audio_bytes)
            audio_segment = AudioSegment.from_file(str(candidate))
            break
        except Exception as exc:
            last_error = exc

    if audio_segment is None:
        return None, f"Could not read audio file: {last_error}"

    audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
    audio_segment.export(str(wav_path), format="wav")

    with sr.AudioFile(str(wav_path)) as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.3)
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data, language=language)
        return text.strip(), None
    except sr.UnknownValueError:
        return None, "Could not understand the voice. Please speak clearly and try again."
    except sr.RequestError as exc:
        return None, f"Speech service error: {exc}. Check your internet connection."
    finally:
        for path in temp_dir.glob("farmer_input_audio*"):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
