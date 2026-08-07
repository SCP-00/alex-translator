#!/usr/bin/env python3
"""Alex Voice — Translator Server v3 (Async Pipeline).

Servidor modular. Este archivo es solo el entrypoint: conecta los módulos
del proyecto y arranca el servidor HTTP.

Módulos:
    config    — configuración central (paths, idiomas, pares, idioms)
    translate — MarianMT + detección de idioma + idioms
    tts       — Kokoro-82M ONNX (primario) + Piper (fallback)
    asr       — faster-whisper (GPU)
    pipeline  — workers asíncronos ASR → Traducción → TTS
    handler   — Handler HTTP

Pipeline: Speech → STT (faster-whisper GPU) → Translation (MarianMT GPU) → TTS (Kokoro ONNX CPU)
- ASR: faster-whisper small INT8 (GPU, ~1.5GB)
- Translation: Helsinki-NLP Opus-MT via transformers (GPU) — ~100ms
- TTS: Kokoro-82M ONNX (CPU, 0MB VRAM) — 54 voices, 5 languages
"""

import sys
import threading
import time
from pathlib import Path

import asr
import pipeline
import translate
import tts
from config import PORT
from handler import ThreadingHTTPServer, TranslatorHandler

sys.stdout.reconfigure(encoding='utf-8', errors='replace')


def main():
    print(f"\n{'='*50}")
    print(f"  >> Alex Voice — Translator Server v3 (Async Pipeline)")
    print(f"  >> Port: {PORT}")
    print(f"  >> STT:   faster-whisper small (GPU, ~1.5GB)")
    print(f"  >> TRANS: Helsinki-NLP Opus-MT (transformers, ~100ms)")
    print(f"  >> TTS:   Kokoro-82M ONNX (CPU, 0MB VRAM)")
    print(f"  >> Pipeline: ASR→Trans→TTS (async threading)")
    print(f"{'='*50}\n")

    # Start HTTP server FIRST (serves HTML immediately while models load)
    httpd = ThreadingHTTPServer(("0.0.0.0", PORT), TranslatorHandler)
    print(f"[Translator] HTTP server ready at http://localhost:{PORT}")

    # Start async pipeline workers
    pipeline._pipeline.start()

    # Load Piper TTS fallback (immediate, CPU, fast)
    es_model = Path.home() / ".local/share/alex/models/piper/es_ES-sharvard-medium.onnx"
    en_model = Path.home() / ".local/share/alex/models/piper/en_US-lessac-medium.onnx"
    if es_model.exists() or en_model.exists():
        tts._piper_tts = tts.PiperTTS()
        if tts._piper_tts.available:
            print(f"[Translator] Piper TTS fallback ready (ES/EN)")

    # Load models in background (so HTML is served immediately)
    def _load_models():
        if asr.HAVE_WHISPER:
            print("[Translator] Loading faster-whisper small...")
            t0 = time.time()
            asr._get_asr("small")
            print(f"[Translator] ASR loaded in {time.time()-t0:.1f}s")

        core_pairs = [('en', 'es'), ('es', 'en'), ('en', 'ja'), ('ja', 'en'), ('ja', 'es')]
        loaded = 0
        for fc, tc in core_pairs:
            if translate._get_translation_model(fc, tc):
                loaded += 1
        print(f"[Translator] Translation models loaded: {loaded}/{len(core_pairs)}")

        # Kokoro: now models exist in models/onnx/
        k = tts._get_kokoro_instance()
        if k is not None:
            print("[Translator] Kokoro TTS loaded (CPU, 0 VRAM)")
        else:
            print("[Translator] Kokoro TTS not available — using Piper fallback for ES/EN")
        print("[Translator] All models loaded — pipeline ready")

    threading.Thread(target=_load_models, daemon=True, name="model-loader").start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Translator] Shutting down...")
    finally:
        pipeline._pipeline.stop()
        httpd.server_close()
        print("[Translator] Stopped.")


if __name__ == "__main__":
    main()
