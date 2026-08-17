"""alex-translator — Reconocimiento de voz (ASR).

faster-whisper (CTranslate2) en GPU con detección de idioma integrada.
"""

import base64
import os
import struct
import threading

import numpy as np
import torch

from translate import detect_language

HAVE_WHISPER = False
_asr_models = {}
_asr_lock = threading.Lock()

try:
    from faster_whisper import WhisperModel
    HAVE_WHISPER = True
except ImportError:
    pass


def _get_asr(model_name="small"):
    if model_name in _asr_models:
        return _asr_models[model_name]
    with _asr_lock:
        if model_name in _asr_models:
            return _asr_models[model_name]
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            m = WhisperModel(model_name, device=device, compute_type="int8")
            _asr_models[model_name] = m
            return m
        except Exception as e:
            print(f"[Translator] ASR error: {e}")
            return None


def transcribe_audio(audio_b64, language="auto"):
    """Transcribe audio with faster-whisper. Returns: (text, detected_lang, error)."""
    if not HAVE_WHISPER:
        return None, None, "faster-whisper not installed"
    try:
        raw_audio = base64.b64decode(audio_b64)
    except Exception:
        return None, None, "Invalid audio data"
    if len(raw_audio) < 100:
        return None, None, "Audio too small"
    model_name = os.environ.get("TRANSLATOR_ASR_MODEL", "small")
    model = _get_asr(model_name)
    if model is None:
        return None, None, "ASR model not available"
    try:
        sr = 16000
        if len(raw_audio) >= 44:
            sr = struct.unpack_from('<I', raw_audio, 24)[0]
        offset = raw_audio.find(b'data', 12)
        offset = offset + 8 if offset >= 0 else 44
        pcm = raw_audio[offset:]
        pcm_int16 = np.frombuffer(pcm, dtype=np.int16)
        if len(pcm_int16) == 0:
            return None, None, "Empty audio"
        pcm_float32 = pcm_int16.astype(np.float32) / 32768.0
        # Whisper espera 16kHz; el navegador graba a 44.1/48kHz. Resample lineal.
        if sr != 16000 and len(pcm_float32) > 0:
            n_out = int(len(pcm_float32) * 16000 / sr)
            x_old = np.arange(len(pcm_float32))
            x_new = np.linspace(0, len(pcm_float32) - 1, n_out)
            pcm_float32 = np.interp(x_new, x_old, pcm_float32).astype(np.float32)
            sr = 16000
        lang = None if language == "auto" else language
        segments, info = model.transcribe(
            pcm_float32, language=lang, beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200,
                                threshold=0.5, neg_threshold=0.35, min_speech_duration_ms=200),
            condition_on_previous_text=False,
        )
        texts = [seg.text.strip() for seg in segments]
        full_text = " ".join(texts).strip()
        detected_lang = getattr(info, "language", None) or detect_language(full_text)
        return full_text, detected_lang, None
    except Exception as e:
        return None, None, str(e)[:300]
