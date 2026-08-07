"""alex-translator — Motor TTS.

Kokoro-82M ONNX (CPU, 0 VRAM) como primario y Piper como fallback para ES/EN.
"""

import struct
import threading
import time
from pathlib import Path

import numpy as np

from config import KOKORO_CONFIG, KOKORO_MODEL_PATH, KOKORO_VOICES_PATH

_kokoro_instance = None
KOKORO_LOCK = threading.Lock()
HAVE_KOKORO = False

# ── Piper TTS Python API (fallback for ES/EN) ─────────────
HAVE_PIPER_PYTHON = False
try:
    from piper import PiperVoice
    HAVE_PIPER_PYTHON = True
except ImportError:
    pass


class PiperTTS:
    """Piper TTS fallback for ES/EN (same architecture as server.py)."""

    def __init__(self):
        self.es_voice = None
        self.en_voice = None
        self.available = False
        self._lock = threading.Lock()
        self._init_voices()

    def _init_voices(self):
        if not HAVE_PIPER_PYTHON:
            return
        es_path = Path.home() / ".local/share/alex/models/piper/es_ES-sharvard-medium.onnx"
        en_path = Path.home() / ".local/share/alex/models/piper/en_US-lessac-medium.onnx"
        if es_path.exists():
            try:
                self.es_voice = PiperVoice.load(str(es_path), use_cuda=False)
            except Exception:
                pass
        if en_path.exists():
            try:
                self.en_voice = PiperVoice.load(str(en_path), use_cuda=False)
            except Exception:
                pass
        self.available = (self.es_voice is not None) or (self.en_voice is not None)

    def synthesize(self, text, lang='es'):
        voice = self.es_voice if lang == 'es' else self.en_voice if lang == 'en' else (self.es_voice or self.en_voice)
        if voice is None:
            return None
        with self._lock:
            try:
                chunks = list(voice.synthesize(text))
                if not chunks:
                    return None
                sr = getattr(chunks[0], 'sample_rate', 22050)
                int16_chunks = []
                for c in chunks:
                    if hasattr(c, 'audio_int16_array') and c.audio_int16_array is not None:
                        int16_chunks.append(c.audio_int16_array)
                    elif hasattr(c, 'audio_int16_bytes') and c.audio_int16_bytes is not None:
                        int16_chunks.append(np.frombuffer(c.audio_int16_bytes, dtype=np.int16))
                if not int16_chunks:
                    return None
                pcm = np.concatenate(int16_chunks)
                return pcm, sr
            except Exception:
                return None


# Se asigna en translator.main():
_piper_tts = None


def _get_kokoro_instance():
    """Lazy-load singleton de Kokoro ONNX."""
    global _kokoro_instance, HAVE_KOKORO
    if _kokoro_instance is not None:
        return _kokoro_instance
    with KOKORO_LOCK:
        if _kokoro_instance is not None:
            return _kokoro_instance
        try:
            from kokoro_onnx import Kokoro
            if not KOKORO_MODEL_PATH.exists() or not KOKORO_VOICES_PATH.exists():
                print(f"[Translator] Kokoro ONNX models not found in {KOKORO_MODEL_PATH.parent}")
                return None
            t0 = time.time()
            _kokoro_instance = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
            elapsed = (time.time() - t0) * 1000
            voices = _kokoro_instance.get_voices()
            print(f"[Translator] Kokoro ONNX loaded in {elapsed:.0f}ms — {len(voices)} voices")
            HAVE_KOKORO = True
            return _kokoro_instance
        except Exception as e:
            print(f"[Translator] Kokoro ONNX error: {e}")
            return None


def kokoro_synthesize(text, lang='es'):
    """Synthesize speech with Kokoro ONNX (CPU). Falls back to Piper for ES/EN."""
    # 1. Try Kokoro (primary)
    k = _get_kokoro_instance()
    if k is not None:
        cfg = KOKORO_CONFIG.get(lang, KOKORO_CONFIG['es'])
        try:
            audio, sr = k.create(text, voice=cfg['voice'], speed=cfg.get('speed', 1.0), lang=cfg['lang'])
            if audio is not None and len(audio) > 0:
                return audio, sr
        except Exception as e:
            print(f"[Translator] Kokoro synthesis error: {e}")

    # 2. Fallback to Piper (ES/EN only)
    if lang in ('es', 'en') and _piper_tts is not None and _piper_tts.available:
        result = _piper_tts.synthesize(text, lang)
        if result is not None:
            return result  # (pcm_int16, sr)

    return None, None


def _make_wav(audio_float32, sample_rate):
    """Convert float32 audio to WAV bytes."""
    audio_float32 = np.clip(audio_float32, -1.0, 1.0)
    int16 = (audio_float32 * 32767).astype(np.int16)
    data_size = len(int16) * 2
    buf = bytearray(44 + data_size)
    buf[0:4] = b'RIFF'
    struct.pack_into('<I', buf, 4, data_size + 36)
    buf[8:12] = b'WAVE'
    buf[12:16] = b'fmt '
    struct.pack_into('<I', buf, 16, 16)
    struct.pack_into('<H', buf, 20, 1)
    struct.pack_into('<H', buf, 22, 1)
    struct.pack_into('<I', buf, 24, sample_rate)
    struct.pack_into('<I', buf, 28, sample_rate * 2)
    struct.pack_into('<H', buf, 32, 2)
    struct.pack_into('<H', buf, 34, 16)
    buf[36:40] = b'data'
    struct.pack_into('<I', buf, 40, data_size)
    buf[44:44 + data_size] = int16.tobytes()
    return bytes(buf)
