# 📖 Alex Voice — Guía de Instalación Detallada

> **Versión:** v4.0 · Julio 2026  
> **Repositorio:** [github.com/SCP-00/ALEX_voice](https://github.com/SCP-00/ALEX_voice)  
> **Compatibilidad:** Linux (Ubuntu 22.04+, Debian 12+, Arch, Kali, Fedora), Windows (WSL2)

---

## ⚡ Índice

- [Requisitos por Componente](#-requisitos-por-componente)
- [Instalación Rápida (1 comando)](#-instalación-rápida-1-comando)
- [Instalación Manual (Linux)](#-instalación-manual-linux)
- [Instalación en Windows (WSL2)](#-instalación-en-windows-wsl2)
- [Instalación Selectiva (por componentes)](#-instalación-selectiva-por-componentes)
- [Verificación](#-verificación)
- [Solución de Problemas](#-solución-de-problemas)

---

## 📊 Requisitos por Componente

### Mínimos vs Recomendados

| Componente | Requisito | Mínimo | Recomendado |
|:-----------|:----------|:------:|:-----------:|
| **🎓 Teacher** | GPU VRAM | 4 GB | **6 GB** |
| | RAM | 8 GB | **16 GB** |
| | Disco | 10 GB | 15 GB |
| | SO | Linux | Linux (Ubuntu 22.04+) |
| **💬 Conversation** | GPU VRAM | 4 GB | **6 GB** |
| | RAM | 8 GB | **16 GB** |
| | Disco | 10 GB | 15 GB |
| | SO | Linux | Linux |
| **🌍 Translator** | GPU VRAM | 2 GB | **4 GB** |
| | RAM | 4 GB | 8 GB |
| | Disco | 5 GB | 10 GB |
| | SO | Linux / Windows | Linux |

### GPU Compatibles

| GPU | VRAM | Teacher | Conv | Transl. | Experiencia |
|:----|:----:|:-------:|:----:|:-------:|:------------|
| **RTX 3050 6GB** 🏆 | 6 GB | ✅ | ✅ | ✅ | **Recomendada — la que usamos** |
| RTX 3060 12GB | 12 GB | ✅ | ✅ | ✅ | Sobrado, 128K contexto |
| RTX 4060 8GB | 8 GB | ✅ | ✅ | ✅ | Más rápido |
| RTX 4070+ 12GB+ | 12 GB | ✅ | ✅ | ✅ | Experiencia premium |
| GTX 1650 4GB | 4 GB | ❌ | ❌ | ✅ | Solo Translator |
| GTX 1660 Ti 6GB | 6 GB | ⚠️ | ⚠️ | ✅ | Teacher lento (10-15 tok/s) |
| RTX 3090/4090 24GB | 24 GB | ✅🚀 | ✅🚀 | ✅ | Máxima velocidad |
| **CPU (sin GPU)** | 0 GB | ❌ | ❌ | ✅ | Solo Translator |

---

## 🚀 Instalación Rápida (1 comando)

### Linux (Ubuntu/Debian/Kali/Arch)

```bash
curl -fsSL https://raw.githubusercontent.com/SCP-00/ALEX_voice/main/install.sh | sh
```

Esto hace automáticamente:
1. ✅ Detecta tu GPU y sistema operativo
2. ✅ Instala Python, pip, build-essential, cmake, espeak-ng
3. ✅ Crea virtual environment + pip install (PyTorch CUDA, Whisper, Kokoro, MarianMT, Flask)
4. ✅ Descarga e instala Ollama + modelo prometheus-orchestrator (2.9 GB)
5. ✅ Descarga modelos Kokoro ONNX (311MB + 27MB)
6. ✅ Crea atajo de escritorio
7. ✅ Abre http://localhost:5000

> ⏱️ **Tiempo estimado:** 15-30 minutos (depende de tu velocidad de internet)

---

## 📋 Instalación Manual (Linux)

### Paso 1: Clonar

```bash
git clone https://github.com/SCP-00/ALEX_voice.git
cd ALEX_voice
```

### Paso 2: Setup interactivo

```bash
chmod +x setup.sh
./setup.sh
```

Te aparecerá un menú:

```
  1) Full install (everything, ~6.5GB download)
  2) Quick start (Teacher + Conversation + Grammar + models, ~4GB)
  3) Minimal (Teacher only + models, ~3.5GB)
  4) Datasets only (full language data, ~500MB)
  5) Custom selection
  q) Quit

  Select [1]: _
```

**Recomendado:** Opción 2 (Quick start) para empezar rápido.

### Paso 3: Iniciar

```bash
./alex_voice_app.sh
# → Abre http://localhost:5000
```

### Instalación manual paso a paso (si el setup.sh falla)

```bash
# 1. Python virtual env
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip

# 2. PyTorch CUDA 12.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124

# 3. Core ML packages
pip install faster-whisper onnxruntime silero-vad psutil pynvml numpy

# 4. TTS
pip install kokoro-onnx piper-tts loguru scipy

# 5. Translation
pip install transformers sentencepiece protobuf

# 6. Japanese
pip install cutlet unidic-lite

# 7. Web
pip install flask flask-cors requests

# 8. Ollama model
curl -fsSL https://ollama.com/install.sh | sh
ollama pull prometheus-orchestrator

# 9. Kokoro models
mkdir -p models/onnx
curl -L -o models/onnx/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/onnx/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin

# 10. Iniciar
chmod +x alex_voice_app.sh
./alex_voice_app.sh
```

---

## 🪟 Instalación en Windows (WSL2)

### Requisitos Windows

- **WSL2** con distribución Ubuntu 22.04+
- **NVIDIA Driver** para WSL2 (instalado en Windows, no en WSL)
- **CUDA 12.4+** (Windows driver)
- Al menos 20 GB libres en el disco de WSL

### Pasos

```powershell
# 1. Abrir PowerShell como Administrador
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2

# 2. Dentro de WSL (Ubuntu)
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv build-essential cmake \
  curl wget git unzip libportaudio2 libsndfile1 espeak-ng

# 3. Verificar CUDA desde WSL2
nvidia-smi  # Debería mostrar tu GPU NVIDIA

# 4. Clonar e instalar
cd ~
git clone https://github.com/SCP-00/ALEX_voice.git
cd ALEX_voice
chmod +x setup.sh && ./setup.sh --quick

# 5. Iniciar
./alex_voice_app.sh
```

> **Nota:** Para acceder al menú desde Windows, abre `http://localhost:5000` en tu navegador de Windows.

---

## 🎯 Instalación Selectiva (por componentes)

### Solo Teacher (mínimo viable)

```bash
./setup.sh --minimal
```

- Descarga: ~3.5 GB
- Disco: ~4.0 GB
- VRAM: ~4.0 GB

### Teacher + Conversation (recomendado)

```bash
./setup.sh --quick
```

- Descarga: ~4.0 GB
- Disco: ~5.0 GB
- VRAM: ~4.0 GB

### Teacher + Conversation + Translator + Grammar

```bash
./setup.sh --all
```

- Descarga: ~6.5 GB
- Disco: ~10.0 GB
- VRAM: ~4.5 GB (Teacher máximo)

### Solo Translator

```bash
# Instalación manual mínima
python3 -m venv venv
source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install faster-whisper transformers sentencepiece kokoro-onnx numpy
python3 translator.py
```

---

## ✅ Verificación

Después de la instalación, verifica que todo funciona:

```bash
python3 -c "
import torch
print(f'CUDA: {torch.cuda.is_available()}')        # Debería ser True
print(f'GPU: {torch.cuda.get_device_name(0)}')     # Tu GPU
print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

from faster_whisper import WhisperModel
print('Whisper: OK')

from kokoro_onnx import Kokoro
print('Kokoro: OK')

from transformers import MarianMTModel
print('MarianMT: OK')
"
```

---

## 🐛 Solución de Problemas

### Error: "CUDA out of memory"

```bash
# Cerrar otros programas que usen GPU
# Verificar qué modelos tiene cargados Ollama
ollama ps

# Liberar modelos no usados
curl -s http://localhost:11434/api/generate \
  -d '{"model":"prometheus-orchestrator","keep_alive":"0m"}'
```

### Error: "Ollama is not running"

```bash
# Iniciar Ollama
ollama serve &

# Verificar
curl http://localhost:11434/api/tags
```

### Error: "Port already in use"

```bash
# Puerto 5000 (menú)
fuser -k 5000/tcp

# Puerto 3000 (Teacher)
kill $(lsof -ti:3000)

# Matar todo
pkill -f 'python.*server.py'
pkill -f 'python.*translator.py'
```

### Error: "ModuleNotFoundError: No module named 'torch'"

```bash
# PyTorch CUDA no está instalado
source venv/bin/activate
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### Error: "Translation fails / models not downloading"

```bash
# Limpiar cache de HuggingFace
rm -rf ~/.cache/huggingface/hub/

# Re-ejecutar
python3 translator.py
```

### Error: "Kokoro model not found"

```bash
# Los modelos ONNX no se descargaron
mkdir -p models/onnx
curl -L -o models/onnx/kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -L -o models/onnx/voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

### Problemas de Rendimiento

| Síntoma | Causa Probable | Solución |
|:--------|:---------------|:---------|
| Teacher tarda >60s | Poca VRAM o cold start | Usar `NUM_CTX=32768` en server.py |
| TTS suena robótico | Velocidad muy alta | Ajustar a 0.9x en la UI |
| Whisper no transcribe | VAD muy agresivo | Hablar más cerca del micrófono |
| Conversación sin memoria | No se envían mensajes anteriores | El cliente debe enviar el array `messages` completo |
| Traducción ES→JA no funciona | No hay modelo directo | Se usa pivot ES→EN→JA (automático) |

---

## 🔗 Enlaces

| Recurso | URL |
|:--------|:----|
| **Repositorio** | https://github.com/SCP-00/ALEX_voice |
| **Alex Grammar (app hermana)** | https://github.com/SCP-00/ALEX_grammar |
| **Reportar bug** | https://github.com/SCP-00/ALEX_voice/issues |
| **Ollama** | https://ollama.com |
| **Kokoro TTS** | https://github.com/thewh1teagle/kokoro-onnx |
| **MarianMT** | https://huggingface.co/Helsinki-NLP |

---

<div align="center">

*Alex Voice v4.0 — Julio 2026 · Hecho con ❤️ para aprendices de idiomas*

</div>
