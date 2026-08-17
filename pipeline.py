"""alex-translator — Pipeline asíncrono ASR → Traducción → TTS.

Mientras TTS reproduce la frase N, la GPU ya transcribe/traduce la frase N+1.
Esto reduce la latencia percibida a solo la etapa más lenta.
"""

import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

import asr
import translate
import tts
from config import LANG_MAP

# ── Logging SDK compartido (ALEX) ────────────────────────────
_SHARED_DIR = Path(__file__).parent.parent / "shared"
if str(_SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(_SHARED_DIR))
try:
    from logging_sdk import get_handler
    log = get_handler("translator.pipeline")
    _HAVE_SDK = True
except Exception:
    _HAVE_SDK = False


def _log(event, msg, **data):
    """Log vía SDK si está disponible, si no a consola."""
    if _HAVE_SDK:
        log.info(event, msg, **data)
    print(f"[Pipeline] {msg}")


class TranslationPipeline:
    """Async pipeline: ASR Queue → Translation Queue → TTS Queue

    While TTS plays sentence N, GPU is already transcribing/translating sentence N+1.
    This cuts perceived latency to just the slowest stage (~150ms for TTS).
    """

    def __init__(self):
        # Queues connecting pipeline stages
        self._asr_queue = queue.Queue(maxsize=50)
        self._trans_queue = queue.Queue(maxsize=50)
        self._tts_queue = queue.Queue(maxsize=50)

        # Result tracking for sync requests
        self._results = {}  # request_id -> result
        self._results_lock = threading.Lock()

        # Pipeline state
        self._running = False
        self._workers = []

    def start(self):
        """Start all pipeline worker threads."""
        if self._running:
            return
        self._running = True

        # ASR Worker: audio → text (GPU)
        t = threading.Thread(target=self._asr_worker, daemon=True, name="pipeline-asr")
        t.start()
        self._workers.append(t)

        # Translation Worker: text_es → text_target (GPU)
        t = threading.Thread(target=self._translation_worker, daemon=True, name="pipeline-trans")
        t.start()
        self._workers.append(t)

        # TTS Worker: text → audio (CPU)
        t = threading.Thread(target=self._tts_worker, daemon=True, name="pipeline-tts")
        t.start()
        self._workers.append(t)

        _log("pipeline.start", "Async workers started: ASR → Translation → TTS",
             workers=len(self._workers))

    def stop(self):
        """Stop all pipeline workers."""
        self._running = False
        # Put poison pills to unblock workers
        for q in [self._asr_queue, self._trans_queue, self._tts_queue]:
            try:
                q.put(None, timeout=0.1)
            except queue.Full:
                pass
        for t in self._workers:
            t.join(timeout=3)
        self._workers.clear()
        _log("pipeline.stop", "Workers stopped")

    def submit_audio(self, audio_b64, from_lang="auto", to_lang="es",
                     request_id=None, text_mode=False):
        """Submit audio for async pipeline processing."""
        if request_id is None:
            request_id = str(time.time())
        item = {
            'audio_b64': audio_b64,
            'from_lang': from_lang,
            'to_lang': to_lang,
            'request_id': request_id,
            'text_mode': text_mode,
            't0': time.time(),
        }
        self._asr_queue.put(item)
        return request_id

    def get_result(self, request_id, timeout=30):
        """Wait for and return the result of an async request."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._results_lock:
                if request_id in self._results:
                    result = self._results.pop(request_id)
                    return result
            time.sleep(0.05)
        return None

    def _asr_worker(self):
        """ASR Worker: pulls audio from queue, transcribes, pushes text to translation queue."""
        print("[Pipeline] ASR worker ready")
        while self._running:
            try:
                item = self._asr_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break

            try:
                t0 = time.time()
                text, detected_lang, error = asr.transcribe_audio(
                    item['audio_b64'], item.get('from_lang', 'auto')
                )
                asr_time = (time.time() - t0) * 1000

                if error or not text:
                    _log("pipeline.asr_error", "ASR failed",
                         error=error or "ASR produced no text",
                         request_id=item['request_id'])
                    with self._results_lock:
                        self._results[item['request_id']] = {
                            'error': error or "ASR produced no text",
                            'stage': 'asr',
                        }
                    continue

                # Auto-detect source language
                from_lang = item.get('from_lang', 'auto')
                if from_lang == 'auto':
                    from_lang = detected_lang

                # Pass to translation queue
                trans_item = {
                    'text': text,
                    'from_lang': from_lang,
                    'to_lang': item['to_lang'],
                    'request_id': item['request_id'],
                    'text_mode': item.get('text_mode', False),
                    't0': item['t0'],
                    'asr_time': asr_time,
                    'detected_lang': detected_lang,
                }
                self._trans_queue.put(trans_item)
                _log("pipeline.asr", "ASR transcribed",
                     text=text[:80], asr_time_ms=int(asr_time), lang=detected_lang,
                     request_id=item['request_id'])

            except Exception as e:
                print(f"[Pipeline] ASR worker error: {e}")
                with self._results_lock:
                    self._results[item['request_id']] = {
                        'error': str(e)[:200],
                        'stage': 'asr',
                    }

    def _translation_worker(self):
        """Translation Worker: pulls text, translates, pushes to TTS queue."""
        print("[Pipeline] Translation worker ready")
        while self._running:
            try:
                item = self._trans_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break

            try:
                t0 = time.time()
                translated, error = translate.translate(
                    item['text'], item['from_lang'], item['to_lang']
                )
                trans_time = (time.time() - t0) * 1000

                if error or not translated:
                    with self._results_lock:
                        self._results[item['request_id']] = {
                            'error': error or "Translation produced no text",
                            'stage': 'translation',
                            'original': item['text'],
                        }
                    continue

                # Pass to TTS queue
                tts_item = {
                    'text': translated,
                    'lang': item['to_lang'],
                    'request_id': item['request_id'],
                    'text_mode': item.get('text_mode', False),
                    't0': item['t0'],
                    'asr_time': item.get('asr_time', 0),
                    'trans_time': trans_time,
                    'original': item['text'],
                    'detected_lang': item.get('detected_lang', 'auto'),
                }
                self._tts_queue.put(tts_item)
                _log("pipeline.translate", "Translated",
                     from_lang=item['from_lang'], to_lang=item['to_lang'],
                     trans_time_ms=int(trans_time), request_id=item['request_id'])

            except Exception as e:
                print(f"[Pipeline] Translation worker error: {e}")
                with self._results_lock:
                    self._results[item['request_id']] = {
                        'error': str(e)[:200],
                        'stage': 'translation',
                    }

    def _tts_worker(self):
        """TTS Worker: pulls translated text, synthesizes audio, stores result."""
        print("[Pipeline] TTS worker ready")
        while self._running:
            try:
                item = self._tts_queue.get(timeout=1)
            except queue.Empty:
                continue
            if item is None:
                break

            try:
                t0 = time.time()
                text_mode = item.get('text_mode', False)

                if text_mode:
                    # Text mode: skip TTS, return text only
                    tts_time = 0
                    wav_data = None
                    audio_secs = 0
                else:
                    audio, sr = tts.kokoro_synthesize(item['text'], item['lang'])
                    tts_time = (time.time() - t0) * 1000
                    wav_data = None
                    audio_secs = 0
                    if audio is not None and len(audio) > 0:
                        # Normalize: Piper returns int16, Kokoro returns float32
                        if audio.dtype == np.int16:
                            audio = audio.astype(np.float32) / 32767.0
                        wav_data = tts._make_wav(audio, sr)
                        audio_secs = len(audio) / sr if sr else 0

                total_time = (time.time() - item['t0']) * 1000

                result = {
                    'original': item.get('original', ''),
                    'translation': item['text'],
                    'from_lang': item.get('detected_lang', 'auto'),
                    'to_lang': item['lang'],
                    'from_lang_name': LANG_MAP.get(item.get('detected_lang', 'auto'), 'auto'),
                    'to_lang_name': LANG_MAP.get(item['lang'], item['lang']),
                    'available_languages': list(LANG_MAP.keys()),
                    'asr_time_ms': int(item.get('asr_time', 0)),
                    'translation_time_ms': int(item.get('trans_time', 0)),
                    'tts_time_ms': int(tts_time),
                    'total_time_ms': int(total_time),
                    'audio_duration_s': round(audio_secs, 1),
                    'wav_data': wav_data,
                }
                with self._results_lock:
                    self._results[item['request_id']] = result

                _log("pipeline.tts", "TTS synthesized",
                     tts_time_ms=int(tts_time), total_time_ms=int(total_time),
                     audio_secs=round(audio_secs, 2), request_id=item['request_id'])

            except Exception as e:
                print(f"[Pipeline] TTS worker error: {e}")
                with self._results_lock:
                    self._results[item['request_id']] = {
                        'error': str(e)[:200],
                        'stage': 'tts',
                    }


# ── Global pipeline instance ───────────────────────────────
_pipeline = TranslationPipeline()
