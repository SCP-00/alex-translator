"""Unit tests for alex-translator tts module.

Tests the pure WAV encoder (_make_wav) and fallback behavior
without Kokoro/Piper models installed.
"""

import struct

import numpy as np
import pytest
from unittest.mock import patch, Mock


class TestMakeWav:
    def test_returns_wav_header(self):
        from tts import _make_wav
        audio = np.zeros(8000, dtype=np.float32)
        wav = _make_wav(audio, 16000)

        assert isinstance(wav, bytes)
        assert wav[:4] == b'RIFF'
        assert wav[8:12] == b'WAVE'
        assert wav[12:16] == b'fmt '

    def test_pcm_data_size(self):
        from tts import _make_wav
        audio = np.zeros(8000, dtype=np.float32)
        wav = _make_wav(audio, 16000)

        # 8000 samples * 2 bytes (int16 mono)
        data_size = len(wav) - 44
        assert data_size == 8000 * 2

    def test_sample_rate_encoded(self):
        from tts import _make_wav
        audio = np.zeros(100, dtype=np.float32)
        wav = _make_wav(audio, 24000)
        sr = struct.unpack_from('<I', wav, 24)[0]
        assert sr == 24000

    def test_clips_out_of_range(self):
        from tts import _make_wav
        audio = np.array([2.0, -2.0, 0.5], dtype=np.float32)
        wav = _make_wav(audio, 16000)
        pcm = np.frombuffer(wav[44:], dtype=np.int16)
        # np.clip(audio, -1.0, 1.0) → -1.0 * 32767 = -32767 (el clip ocurre ANTES)
        assert pcm[0] == 32767
        assert pcm[1] == -32767


class TestKokoroSynthesize:
    @pytest.fixture(autouse=True)
    def _no_models(self):
        """Simula que no hay modelos en disco (paths con exists() → False)."""
        import tts
        tts._kokoro_instance = None
        tts.HAVE_KOKORO = False
        tts._piper_tts = None
        fake_path = Mock()
        fake_path.exists.return_value = False
        with patch.object(tts, "KOKORO_MODEL_PATH", fake_path), \
             patch.object(tts, "KOKORO_VOICES_PATH", fake_path):
            yield

    def test_returns_none_without_models(self):
        """Sin modelo en disco, kokoro_synthesize devuelve None (fallback a Piper)."""
        import tts
        audio, sr = tts.kokoro_synthesize("Hola mundo", "es")
        assert audio is None
        assert sr is None

    def test_get_kokoro_returns_none_without_models(self):
        """Sin modelo en disco, _get_kokoro_instance devuelve None."""
        import tts
        result = tts._get_kokoro_instance()
        assert result is None
