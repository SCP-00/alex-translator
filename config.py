"""alex-translator — Configuración central.

Paths, puerto, idiomas, pares de modelos MarianMT, diccionario de idioms
y configuración TTS. Todo lo configurable vive aquí.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
FRONTEND_DIR = PROJECT_ROOT / "frontend"
# PLAN_B_PORT lo fija el CLI `alex start translator` (puerto canónico 3003);
# TRANSLATOR_PORT es el fallback si se lanza a mano.
PORT = int(os.environ.get("PLAN_B_PORT") or os.environ.get("TRANSLATOR_PORT") or "3003")


# ── Language maps ──────────────────────────────────────────
LANG_MAP = {
    'en': 'English', 'es': 'Spanish', 'ja': 'Japanese',
    'fr': 'French', 'ko': 'Korean', 'zh': 'Chinese',
    'de': 'German', 'pt': 'Portuguese',
}


# ── MarianMT Translation (transformers) ───────────────────
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


# ── Kokoro-82M ONNX TTS (CPU, 0 VRAM) ────────────────────
KOKORO_CONFIG = {
    'es': {'lang': 'es', 'voice': 'em_alex', 'speed': 0.9},
    'en': {'lang': 'en-us', 'voice': 'af_heart', 'speed': 1.0},
    'ja': {'lang': 'ja', 'voice': 'jf_alpha', 'speed': 0.9},
    'fr': {'lang': 'fr-fr', 'voice': 'ff_siwis', 'speed': 1.0},
    'de': {'lang': 'de', 'voice': 'bf_emma', 'speed': 1.0},
}

KOKORO_MODEL_PATH = Path.home() / ".local/share/alex/models/onnx/kokoro-v1.0.onnx"
KOKORO_VOICES_PATH = Path.home() / ".local/share/alex/models/onnx/voices-v1.0.bin"


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
