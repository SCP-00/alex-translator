# Alex Translator — Real-time Speech Translation

> **Part of the Alex Ecosystem** — Traducción de voz con pipeline asíncrono.

## Architecture

```
Audio → VAD → [ASR Queue] → ASR Worker (GPU) → [Trans Queue] → Translation (GPU) → [TTS Queue] → TTS (CPU) → Audio
```

Pipeline asíncrono: mientras TTS reproduce la oración N, la GPU ya transcribe N+1.

## Features
- 🌍 **8 idiomas**: ES, EN, JA, FR, KO, ZH, DE, PT
- 🔄 **Pipeline async**: ASR → Translation → TTS
- 🧠 **MarianMT** via HuggingFace Transformers
- 🔊 **Kokoro-82M** TTS en CPU (0 VRAM)
- 📖 **Idiom-aware**: reemplaza idioms antes de traducir

## Quick Start

Desde el monorepo S2S (recomendado — gestiona venv, puertos y el motor LLM):

```bash
./alex start translator   # → http://localhost:3003
```

Lanzamiento manual (puerto real **3003**; `translator.py` no acepta `--port`):

```bash
cd alex-translator
source .venv/bin/activate   # o el venv compartido del ecosistema
python3 translator.py       # → http://localhost:3003
```

