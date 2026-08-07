"""alex-translator — Motor de traducción.

MarianMT (Helsinki-NLP via transformers) en GPU con caché lazy de pares,
pre-procesado de idioms y detección de idioma.
"""

import threading
import time

import torch

from config import IDIOM_MAP, PAIR_MODEL_MAP, PIVOT_PAIRS

HAVE_TRANSFORMERS = False
_translation_models = {}  # (from_lang, to_lang) -> (model, tokenizer)
_translation_lock = threading.Lock()

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


# ── Language Detection ─────────────────────────────────────
def detect_language(text):
    """Detect es/en/ja."""
    if not text.strip():
        return 'en'
    if any('\u3040' <= c <= '\u309f' or '\u30a0' <= c <= '\u30ff' or '\u4e00' <= c <= '\u9fff' for c in text):
        return 'ja'
    es_words = {'hola', 'gracias', 'como', 'estas', 'muy', 'bien', 'que', 'el', 'la', 'los', 'las',
                'por', 'para', 'con', 'sin', 'es', 'son', 'del', 'todo', 'casa', 'agua', 'vida'}
    words = [w.strip('.,!?;:\'"()[]{}') for w in text.lower().split() if w.strip('.,!?;:\'"()[]{}')]
    es_chars = sum(1 for c in text if '\u00e1' <= c <= '\u00fa' or c in 'ñçüöéèêëàâîôùû¿¡')
    if not words:
        return 'en'
    if sum(1 for w in words if w in es_words) > 0 or es_chars > 0:
        return 'es'
    return 'en'
