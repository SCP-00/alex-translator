"""Unit tests for alex-translator pipeline.

Tests the async pipeline orchestration with mocked stage functions
(ASR/translation/TTS are external services → mocked, pipeline logic stays real).
"""

import threading
import time

import numpy as np
import pytest


@pytest.fixture
def pipeline(monkeypatch):
    """A TranslationPipeline with stage functions mocked."""
    import pipeline as pipeline_mod
    import asr, translate, tts

    def fake_asr(audio_b64, lang='auto'):
        return "Hola mundo", "es", None

    def fake_translate(text, from_lang, to_lang):
        return "Hello world", None

    def fake_kokoro(text, lang):
        sr = 16000
        audio = np.zeros(sr, dtype=np.float32)  # 1 second of silence
        return audio, sr

    monkeypatch.setattr(asr, 'transcribe_audio', fake_asr)
    monkeypatch.setattr(translate, 'translate', fake_translate)
    monkeypatch.setattr(tts, 'kokoro_synthesize', fake_kokoro)

    p = pipeline_mod.TranslationPipeline()
    p.start()
    yield p
    p.stop()


class TestPipeline:
    def test_start_starts_workers(self, pipeline):
        assert pipeline._running is True
        assert len(pipeline._workers) >= 3

    def test_submit_and_get_result(self, pipeline):
        request_id = pipeline.submit_audio(
            "fake_audio_base64", "auto", "en",
            request_id="test-req-1"
        )
        result = pipeline.get_result(request_id, timeout=10)

        assert result is not None
        assert result['translation'] == "Hello world"
        assert result['original'] == "Hola mundo"
        assert result['to_lang'] == 'en'
        assert 'audio_duration_s' in result

    def test_get_result_timeout(self, pipeline):
        # Request ID that never resolves → timeout returns None
        result = pipeline.get_result("nonexistent-id", timeout=0.5)
        assert result is None

    def test_stop_clears_workers(self, pipeline):
        pipeline.stop()
        assert pipeline._running is False
        assert len(pipeline._workers) == 0

    def test_text_mode_skips_tts(self, monkeypatch):
        import pipeline as pipeline_mod
        import asr, translate, tts

        monkeypatch.setattr(asr, 'transcribe_audio', lambda a, l='auto': ("Hola", "es", None))
        monkeypatch.setattr(translate, 'translate', lambda t, f, to: ("Hello", None))
        # TTS should NOT be called in text mode
        called = []
        def fake_kokoro(text, lang):
            called.append(text)
            raise AssertionError("TTS should not run in text mode")
        monkeypatch.setattr(tts, 'kokoro_synthesize', fake_kokoro)

        p = pipeline_mod.TranslationPipeline()
        p.start()
        try:
            request_id = p.submit_audio("fake", "auto", "en", request_id="text-mode", text_mode=True)
            result = p.get_result(request_id, timeout=10)
            assert result is not None
            assert result['translation'] == "Hello"
            assert called == []
        finally:
            p.stop()


class TestPipelineErrorHandling:
    def test_asr_error_returns_error_result(self, monkeypatch):
        import pipeline as pipeline_mod
        import asr, translate, tts
        monkeypatch.setattr(asr, 'transcribe_audio', lambda a, l='auto': (None, None, "ASR failed"))
        monkeypatch.setattr(translate, 'translate', lambda t, f, to: (None, None))
        monkeypatch.setattr(tts, 'kokoro_synthesize', lambda t, l: (None, None))

        p = pipeline_mod.TranslationPipeline()
        p.start()
        try:
            request_id = p.submit_audio("fake", "auto", "en", request_id="asr-err")
            result = p.get_result(request_id, timeout=10)
            assert result is not None
            assert 'error' in result
            assert result['stage'] == 'asr'
        finally:
            p.stop()
