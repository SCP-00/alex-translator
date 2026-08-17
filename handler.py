"""alex-translator — Handler HTTP.

Rutas: /api/translate (texto), /api/tts, /api/asr, /api/pipeline (async),
/api/load y /api/unload (gestión de VRAM). Estado mutable (translate,
tts, asr, pipeline) accedido vía módulos.
"""

import base64
import json
import sys
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from socketserver import ThreadingMixIn

import torch

import asr
import pipeline
import translate
import tts
from config import FRONTEND_DIR, LANG_MAP

# ── Logging SDK compartido (ALEX) ────────────────────────────
_SHARED_DIR = Path(__file__).parent.parent / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))
try:
    from logging_sdk import request_context
    _HAVE_SDK = True
except Exception:
    _HAVE_SDK = False


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server — prevents blocking on pipeline waits."""
    daemon_threads = True


class TranslatorHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(FRONTEND_DIR), **kwargs)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        if self.path == "/api/status":
            self._json({
                "whisper_loaded": asr.HAVE_WHISPER,
                "transformers_loaded": translate.HAVE_TRANSFORMERS,
                "kokoro_loaded": tts.HAVE_KOKORO,
                "languages": list(LANG_MAP.keys()),
                "translators_loaded": list(f"{k[0]}→{k[1]}" for k in translate._translation_models.keys()),
                "pipeline_running": pipeline._pipeline._running,
            })
        elif self.path == "/api/pipeline/stats":
            self._json({
                "asr_queue_size": pipeline._pipeline._asr_queue.qsize(),
                "trans_queue_size": pipeline._pipeline._trans_queue.qsize(),
                "tts_queue_size": pipeline._pipeline._tts_queue.qsize(),
            })
        elif self.path in ("/", "/index.html"):
            self.path = "/index.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"

        if _HAVE_SDK:
            with request_context("translator.http", event=f"http.{self.path}"):
                self._route_post(body)
        else:
            self._route_post(body)

    def _route_post(self, body):
        if self.path == "/api/translate":
            self._handle_translate(body)
        elif self.path == "/api/tts":
            self._handle_tts(body)
        elif self.path == "/api/asr":
            self._handle_asr(body)
        elif self.path == "/api/pipeline":
            self._handle_pipeline(body)
        elif self.path == "/api/load":
            self._handle_load(body)
        elif self.path == "/api/unload":
            self._handle_unload()
        else:
            self.send_response(404)
            self.end_headers()

    def _handle_load(self, body):
        """Pre-load translation models and Kokoro TTS."""
        t0 = time.time()
        try:
            data = json.loads(body)
        except Exception:
            data = {}
        pairs = data.get("pairs", [("en", "es"), ("es", "en"), ("en", "ja"), ("ja", "en")])
        loaded = 0
        for fc, tc in pairs:
            if translate._get_translation_model(fc, tc):
                loaded += 1
        tts._get_kokoro_instance()
        elapsed = int((time.time() - t0) * 1000)
        self._json({"ok": True, "translators_loaded": loaded, "time_ms": elapsed})

    def _handle_unload(self):
        """Unload models to free VRAM."""
        with translate._translation_lock:
            translate._translation_models.clear()
        with tts.KOKORO_LOCK:
            tts._kokoro_instance = None
            tts.HAVE_KOKORO = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self._json({"ok": True, "message": "All models unloaded"})

    def _handle_translate(self, body):
        """Sync text translation (no pipeline)."""
        t_start = time.time()
        try:
            data = json.loads(body)
            text = data.get("text", "").strip()
            from_lang = data.get("from_lang", "auto")
            to_lang = data.get("to_lang", "es")
            if not text:
                self._json({"error": "Empty text"})
                return
            if from_lang == "auto":
                from_lang = translate.detect_language(text)
            t_trans_start = time.time()
            translation, error = translate.translate(text, from_lang, to_lang)
            trans_time = int((time.time() - t_trans_start) * 1000)
            if error:
                self._json({"error": error})
                return
            total_time = int((time.time() - t_start) * 1000)
            self._json({
                "original": text,
                "translation": translation,
                "from_lang": from_lang,
                "to_lang": to_lang,
                "from_lang_name": LANG_MAP.get(from_lang, from_lang),
                "to_lang_name": LANG_MAP.get(to_lang, to_lang),
                "available_languages": list(LANG_MAP.keys()),
                "translation_time_ms": trans_time,
                "total_time_ms": total_time,
            })
        except Exception as e:
            self._json({"error": str(e)[:200]})

    def _handle_tts(self, body):
        """Generate audio with Kokoro ONNX (CPU, 0 VRAM)."""
        t0 = time.time()
        try:
            data = json.loads(body)
            text = data.get("text", "").strip()
            lang = data.get("language", "Spanish")
            if not text:
                self._json({"error": "Empty text"})
                return
            lang_code_map = {
                'Spanish': 'es', 'English': 'en', 'Japanese': 'ja',
                'French': 'fr', 'German': 'de',
            }
            lang_code = lang_code_map.get(lang, 'es')
            audio, sr = tts.kokoro_synthesize(text, lang_code)
            if audio is None:
                self._json({"error": "Kokoro synthesis failed"})
                return
            wav_data = tts._make_wav(audio, sr)
            gen_time = int((time.time() - t0) * 1000)
            audio_secs = len(audio) / sr if sr else 0
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav_data)))
            self.send_header("X-Generation-Time-Ms", str(gen_time))
            self.send_header("X-Audio-Duration-S", f"{audio_secs:.1f}")
            self.send_header("X-TTS-Engine", "kokoro_onnx")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(wav_data)
        except Exception as e:
            self._json({"error": str(e)[:200]})

    def _handle_asr(self, body):
        """Sync ASR (no pipeline)."""
        t0 = time.time()
        try:
            data = json.loads(body)
            audio_b64 = data.get("audio", "")
            lang = data.get("lang", "auto")
            if not audio_b64:
                self._json({"error": "No audio"})
                return
            text, whisper_lang, error = asr.transcribe_audio(audio_b64, lang)
            if error:
                self._json({"error": error})
                return
            detected_lang = whisper_lang or translate.detect_language(text)
            self._json({
                "text": text,
                "detected_lang": detected_lang,
                "detected_lang_name": LANG_MAP.get(detected_lang, detected_lang),
                "time_ms": int((time.time() - t0) * 1000),
            })
        except Exception as e:
            self._json({"error": str(e)[:200]})

    def _handle_pipeline(self, body):
        """Async pipeline: submit audio → get translated audio back.

        Request:
            {"audio_b64": "...", "from_lang": "auto", "to_lang": "ja"}

        Response (waits up to 10s for result):
            {"original": "...", "translation": "...", "tts_time_ms": ...,
             "total_time_ms": ..., "audio": "<base64 wav>"}
        """
        try:
            data = json.loads(body)
            audio_b64 = data.get("audio", "") or data.get("audio_b64", "")
            from_lang = data.get("from_lang", "auto")
            to_lang = data.get("to_lang", "es")

            if not audio_b64:
                self._json({"error": "No audio"})
                return

            # Submit to async pipeline
            request_id = pipeline._pipeline.submit_audio(
                audio_b64, from_lang, to_lang,
                request_id=f"req_{int(time.time()*1000)}"
            )

            # Wait for result (blocking but pipeline runs async)
            result = pipeline._pipeline.get_result(request_id, timeout=10)

            if result is None:
                self._json({"error": "Pipeline timeout (10s)"})
                return

            if 'error' in result:
                self._json(result)
                return

            # Encode audio for response
            wav_b64 = base64.b64encode(result['wav_data']).decode() if result.get('wav_data') else None
            self._json({
                "original": result['original'],
                "translation": result['translation'],
                "from_lang": result['from_lang'],
                "to_lang": result['to_lang'],
                "from_lang_name": result['from_lang_name'],
                "to_lang_name": result['to_lang_name'],
                "asr_time_ms": result['asr_time_ms'],
                "translation_time_ms": result['translation_time_ms'],
                "tts_time_ms": result['tts_time_ms'],
                "total_time_ms": result['total_time_ms'],
                "audio_duration_s": result['audio_duration_s'],
                "audio": wav_b64,
            })

        except Exception as e:
            self._json({"error": str(e)[:200]})

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "/api/" in msg:
            print(f"[Translator] {msg}")
