#!/usr/bin/env python3
"""
Alex Voice — Translator Server v3 (Async Pipeline)
====================================================
Asynchronous Sentence-Level Pipelining for real-time speech translation.

Architecture:
  Audio → VAD → [ASR Queue] → ASR Worker (GPU) → [Trans Queue] → Translation Worker (GPU) → [TTS Queue] → TTS Worker (CPU) → Audio Out

Key improvement: While TTS plays sentence N, GPU already transcribes sentence N+1.
This cuts perceived latency from sequential (ASR+Trans+TTS) to just the slowest stage.

Pipeline: Speech → STT (faster-whisper GPU) → Translation (MarianMT GPU) → TTS (Kokoro ONNX CPU)
- ASR: faster-whisper small INT8 (GPU, ~1.5GB)
- Translation: Helsinki-NLP Opus-MT via transformers (GPU) — ~100ms
- TTS: Kokoro-82M ONNX (CPU, 0MB VRAM) — 54 voices, 5 languages
"""

import json, os, sys, time, base64, struct, threading, re, queue
from pathlib import Path
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn

import numpy as np
import torch

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PROJECT_ROOT = Path(__file__).parent.resolve()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
PORT = int(os.environ.get("TRANSLATOR_PORT", "3003"))

# ── Language maps ──────────────────────────────────────────
LANG_MAP = {
    'en': 'English', 'es': 'Spanish', 'ja': 'Japanese',
    'fr': 'French', 'ko': 'Korean', 'zh': 'Chinese',
    'de': 'German', 'pt': 'Portuguese',
}

# ── MarianMT Translation (transformers) ───────────────────
HAVE_TRANSFORMERS = False
_translation_models = {}  # (from_lang, to_lang) -> (model, tokenizer)
_translation_lock = threading.Lock()

PAIR_MODEL_MAP = {
    ('en', 'es'): 'Helsinki-NLP/opus-mt-en-es',
    ('es', 'en'): 'Helsinki-NLP/opus-mt-es-en',
    ('en', 'ja'): 'Helsinki-NLP/opus-mt-en-jap',
    ('ja', 'en'): 'Helsinki-NLP/opus-mt-ja-en',
    ('ja', 'es'): 'Helsinki-NLP/opus-mt-ja-es',
}

PIVOT_PAIRS = {
    ('es', 'ja'): ('es', 'en', 'ja'),
    ('ja', 'es'): ('ja', 'en', 'es'),
}

try:
    from transformers import MarianMTModel, MarianTokenizer
    HAVE_TRANSFORMERS = True
except ImportError:
    print("[Translator] transformers not available. pip install transformers")


def _get_translation_model(from_lang, to_lang):
    """Lazy-load a MarianMT model+tokenizer pair."""
    if not HAVE_TRANSFORMERS:
        return None
    key = (from_lang, to_lang)
    if key in _translation_models:
        return _translation_models[key]
    with _translation_lock:
        if key in _translation_models:
            return _translation_models[key]
        hf_model_name = PAIR_MODEL_MAP.get(key)
        if not hf_model_name:
            return None
        try:
            t0 = time.time()
            device = "cuda" if torch.cuda.is_available() else "cpu"
            tokenizer = MarianTokenizer.from_pretrained(hf_model_name)
            model = MarianMTModel.from_pretrained(hf_model_name).to(device)
            model.eval()
            elapsed = (time.time() - t0) * 1000
            print(f"[Translator] {from_lang}→{to_lang} loaded on {device} in {elapsed:.0f}ms")
            result = (model, tokenizer)
            _translation_models[key] = result
            return result
        except Exception as e:
            print(f"[Translator] Error loading {from_lang}→{to_lang}: {e}")
            return None


# ── Idiom Dictionary (pre-processing) ─────────────────────
# Reemplaza idioms en el texto ORIGEN por equivalentes neutros
# en el mismo idioma ANTES de enviarlos a MarianMT.
#
# Estrategia:
# 1. Detectar idiom en el texto fuente (ej: "break a leg")
# 2. Reemplazar por su equivalente neutro (ej: "good luck")
# 3. MarianMT traduce el texto neutro correctamente a cualquier idioma
#
# Esto funciona para TODOS los pares de idiomas, no solo EN↔ES.
# Ordenado por longitud descendente para matchear frase más específica primero.
IDIOM_MAP = {
    'en': [  # English idioms → plain English
        ("it's raining cats and dogs", "it's raining heavily"),
        ("raining cats and dogs", "raining heavily"),
        ("break a leg", "good luck"),
        ("a piece of cake", "very easy"),
        ("piece of cake", "very easy"),
        ("cost an arm and a leg", "be very expensive"),
        ("costs an arm and a leg", "is very expensive"),
        ("once in a blue moon", "rarely"),
        ("when pigs fly", "never"),
        ("hit the nail on the head", "be exactly right"),
        ("hits the nail on the head", "is exactly right"),
        ("the ball is in your court", "it's your turn to act"),
        ("bite the bullet", "face the difficulty bravely"),
        ("bites the bullet", "faces the difficulty bravely"),
        ("let the cat out of the bag", "reveal the secret"),
        ("lets the cat out of the bag", "reveals the secret"),
        ("feeling under the weather", "feeling unwell"),
        ("under the weather", "unwell"),
        ("feel under the weather", "feel unwell"),
        ("add insult to injury", "make a bad situation worse"),
        ("cry over spilt milk", "worry about past mistakes"),
        ("don't cry over spilt milk", "don't worry about the past"),
    ],
    'es': [  # Spanish idioms → plain Spanish
        ("lloviendo a cántaros", "lloviendo mucho"),
        ("llueve a cántaros", "llueve mucho"),
        ("pan comido", "muy fácil"),
        ("costar un ojo de la cara", "ser muy caro"),
        ("cuesta un ojo de la cara", "es muy caro"),
        ("costó un ojo de la cara", "fue muy caro"),
        ("dar en el clavo", "tener toda la razón"),
        ("da en el clavo", "tiene toda la razón"),
        ("dio en el clavo", "tuvo toda la razón"),
        ("más vale tarde que nunca", "es mejor tarde que nunca"),
        ("en menos que canta un gallo", "muy rápidamente"),
        ("no hay mal que por bien no venga", "todo pasa por algo"),
        ("al mal tiempo, buena cara", "hay que ser positivo"),
        ("cuando las ranas críen pelo", "nunca"),
        ("descubrir el pastel", "revelar un secreto"),
        ("hacer de tripas corazón", "enfrentar la dificultad con valor"),
        ("tirar la toalla", "rendirse"),
        ("echar leña al fuego", "empeorar la situación"),
    ],
}


def _replace_idioms(text, lang):
    """Reemplaza idioms en el texto ORIGEN por equivalentes neutros.
    
    Se ejecuta ANTES de enviar a MarianMT para que el modelo
traduzca el significado real del idiom en vez de sus palabras literales.
    
    Protección de word boundaries: verifica que el carácter antes y
después del match no sea alfanumérico, evitando falsos positivos
como "once in a blue moon" dentro de "once in a blue moonlight".
    """
    idioms = IDIOM_MAP.get(lang)
    if not idioms:
        return text
    result = text
    result_lower = result.lower()
    for idiom, replacement in idioms:
        idx = result_lower.find(idiom.lower())
        if idx == -1:
            continue
        # Word boundary check: char before/after must be non-alphanumeric
        before_char = result[idx - 1] if idx > 0 else ' '
        after_idx = idx + len(idiom)
        after_char = result[after_idx] if after_idx < len(result) else ' '
        if before_char.isalnum() or after_char.isalnum():
            continue  # Partial word match, skip
        # Preserve case: if the actual text has uppercase start, capitalize replacement
        if result[idx].isupper():
            replacement = replacement[0].upper() + replacement[1:]
        result = result[:idx] + replacement + result[after_idx:]
        result_lower = result.lower()
    return result


def translate(text, from_lang, to_lang):
    """Translate text using Helsinki-NLP Opus-MT via transformers."""
    if not HAVE_TRANSFORMERS:
        return None, "transformers not installed"
    if from_lang == to_lang:
        return text, None
    
    # Step 0: Pre-process — replace idioms with neutral equivalents
    processed_text = _replace_idioms(text, from_lang)
    
    # Step 1: Try MarianMT translation
    result = _get_translation_model(from_lang, to_lang)
    if result is not None:
        try:
            model, tokenizer = result
            device = next(model.parameters()).device
            inputs = tokenizer(processed_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=512)
            translated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return translated, None
        except Exception as e:
            print(f"[Translator] Translation error {from_lang}→{to_lang}: {e}")
    
    # Fallback: pivot through English
    pivot = PIVOT_PAIRS.get((from_lang, to_lang))
    if pivot:
        t_en, _ = translate(processed_text, from_lang, 'en')
        if t_en:
            t_final, err = translate(t_en, 'en', to_lang)
            if t_final:
                return t_final, None
            return None, err
    return None, f"No translation path for {from_lang}→{to_lang}"


# ── Kokoro-82M ONNX TTS (CPU, 0 VRAM) ────────────────────
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

_piper_tts = None

KOKORO_CONFIG = {
    'es': {'lang': 'es', 'voice': 'em_alex', 'speed': 0.9},
    'en': {'lang': 'en-us', 'voice': 'af_heart', 'speed': 1.0},
    'ja': {'lang': 'ja', 'voice': 'jf_alpha', 'speed': 0.9},
    'fr': {'lang': 'fr-fr', 'voice': 'ff_siwis', 'speed': 1.0},
    'de': {'lang': 'de', 'voice': 'bf_emma', 'speed': 1.0},
}

KOKORO_MODEL_PATH = Path.home() / ".local/share/alex/models/onnx/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = Path.home() / ".local/share/alex/models/onnx/voices-v1.0.bin"


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


# ── ASR (faster-whisper, GPU) ─────────────────────────────
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


# ── Language Detection ─────────────────────────────────────
def detect_language(text):
    """Detect es/en/ja."""
    if not text.strip():
        return 'en'
    if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in text):
        return 'ja'
    es_words = {'hola', 'gracias', 'como', 'estas', 'muy', 'bien', 'que', 'el', 'la', 'los', 'las',
                'por', 'para', 'con', 'sin', 'es', 'son', 'del', 'todo', 'casa', 'agua', 'vida'}
    words = [w.strip('.,!?;:\'\"()[]{}') for w in text.lower().split() if w.strip('.,!?;:\'\"()[]{}')]
    es_chars = sum(1 for c in text if '\u00e1' <= c <= '\u00fa' or c in 'ñçüöéèêëàâîôùû¿¡')
    if not words:
        return 'en'
    if sum(1 for w in words if w in es_words) > 0 or es_chars > 0:
        return 'es'
    return 'en'


# ══════════════════════════════════════════════════════════════
#  ASYNC PIPELINE — Queue-based workers for real-time fluency
# ══════════════════════════════════════════════════════════════

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

        print("[Pipeline] Async workers started: ASR → Translation → TTS")

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
        print("[Pipeline] Workers stopped")

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
                text, detected_lang, error = transcribe_audio(
                    item['audio_b64'], item.get('from_lang', 'auto')
                )
                asr_time = (time.time() - t0) * 1000

                if error or not text:
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
                print(f"[Pipeline] ASR: \"{text[:50]}...\" ({asr_time:.0f}ms, lang={detected_lang})")

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
                translated, error = translate(
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
                print(f"[Pipeline] Translation: \"{item['text'][:30]}\" → \"{translated[:30]}\" ({trans_time:.0f}ms)")

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
                    audio, sr = kokoro_synthesize(item['text'], item['lang'])
                    tts_time = (time.time() - t0) * 1000
                    wav_data = None
                    audio_secs = 0
                    if audio is not None and len(audio) > 0:
                        # Normalize: Piper returns int16, Kokoro returns float32
                        if audio.dtype == np.int16:
                            audio = audio.astype(np.float32) / 32767.0
                        wav_data = _make_wav(audio, sr)
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

                print(f"[Pipeline] TTS: {tts_time:.0f}ms, total pipeline: {total_time:.0f}ms")

            except Exception as e:
                print(f"[Pipeline] TTS worker error: {e}")
                with self._results_lock:
                    self._results[item['request_id']] = {
                        'error': str(e)[:200],
                        'stage': 'tts',
                    }


# ── Global pipeline instance ───────────────────────────────
_pipeline = TranslationPipeline()


# ── Threading HTTP Server ─────────────────────────────────
class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    """Multi-threaded HTTP server — prevents blocking on pipeline waits."""
    daemon_threads = True


# ── HTTP Handler ───────────────────────────────────────────
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
                "whisper_loaded": HAVE_WHISPER,
                "transformers_loaded": HAVE_TRANSFORMERS,
                "kokoro_loaded": HAVE_KOKORO,
                "languages": list(LANG_MAP.keys()),
                "translators_loaded": list(f"{k[0]}→{k[1]}" for k in _translation_models.keys()),
                "pipeline_running": _pipeline._running,
            })
        elif self.path == "/api/pipeline/stats":
            self._json({
                "asr_queue_size": _pipeline._asr_queue.qsize(),
                "trans_queue_size": _pipeline._trans_queue.qsize(),
                "tts_queue_size": _pipeline._tts_queue.qsize(),
            })
        elif self.path in ("/", "/index.html"):
            self.path = "/translator.html"
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length) if length > 0 else b"{}"

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
            if _get_translation_model(fc, tc):
                loaded += 1
        _get_kokoro_instance()
        elapsed = int((time.time() - t0) * 1000)
        self._json({"ok": True, "translators_loaded": loaded, "time_ms": elapsed})

    def _handle_unload(self):
        """Unload models to free VRAM."""
        global _translation_models, _kokoro_instance, HAVE_KOKORO
        with _translation_lock:
            _translation_models.clear()
        with KOKORO_LOCK:
            _kokoro_instance = None
            HAVE_KOKORO = False
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
                from_lang = detect_language(text)
            t_trans_start = time.time()
            translation, error = translate(text, from_lang, to_lang)
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
            audio, sr = kokoro_synthesize(text, lang_code)
            if audio is None:
                self._json({"error": "Kokoro synthesis failed"})
                return
            wav_data = _make_wav(audio, sr)
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
            text, whisper_lang, error = transcribe_audio(audio_b64, lang)
            if error:
                self._json({"error": error})
                return
            detected_lang = whisper_lang or detect_language(text)
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
            request_id = _pipeline.submit_audio(
                audio_b64, from_lang, to_lang,
                request_id=f"req_{int(time.time()*1000)}"
            )

            # Wait for result (blocking but pipeline runs async)
            result = _pipeline.get_result(request_id, timeout=10)

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


# ── MAIN ───────────────────────────────────────────────────
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
    _pipeline.start()

    # Load Piper TTS fallback (immediate, CPU, fast)
    global _piper_tts
    es_model = Path.home() / ".local/share/alex/models/piper/es_ES-sharvard-medium.onnx"
    en_model = Path.home() / ".local/share/alex/models/piper/en_US-lessac-medium.onnx"
    if es_model.exists() or en_model.exists():
        _piper_tts = PiperTTS()
        if _piper_tts.available:
            print(f"[Translator] Piper TTS fallback ready (ES/EN)")

    # Load models in background (so HTML is served immediately)
    def _load_models():
        if HAVE_WHISPER:
            print("[Translator] Loading faster-whisper small...")
            t0 = time.time()
            _get_asr("small")
            print(f"[Translator] ASR loaded in {time.time()-t0:.1f}s")

        core_pairs = [('en', 'es'), ('es', 'en'), ('en', 'ja'), ('ja', 'en'), ('ja', 'es')]
        loaded = 0
        for fc, tc in core_pairs:
            if _get_translation_model(fc, tc):
                loaded += 1
        print(f"[Translator] Translation models loaded: {loaded}/{len(core_pairs)}")

        # Kokoro: now models exist in models/onnx/
        k = _get_kokoro_instance()
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
        _pipeline.stop()
        httpd.server_close()
        print("[Translator] Stopped.")


if __name__ == "__main__":
    main()
