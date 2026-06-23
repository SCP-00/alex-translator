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

```bash
cd alex-translator
python3 translator.py --port 3002
```

