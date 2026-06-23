#!/usr/bin/env bash
set -e

# ═══════════════════════════════════════════════════════════════
#  Alex Voice — Interactive Setup (v4.0)
# ═══════════════════════════════════════════════════════════════
#  Usage: chmod +x setup.sh && ./setup.sh
#
#  Interactive menu lets you choose which components to install:
#    [1] 🎓 Teacher      — LLM + TTS + ASR (Ollama, Kokoro, Whisper)
#    [2] 💬 Conversation — Chat + Voice round-trip
#    [3] 🌍 Translator   — MarianMT + TTS pipeline
#    [4] 📝 Grammar App  — Full vocabulary datasets + exercises
#    [5] 🌐 All datasets  — Tatoeba full + JMdict + KANJIDIC + wordfreq
#
#  Each component shows download size before installing.
# ═══════════════════════════════════════════════════════════════

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

# ── Colors ──
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

info()  { echo -e "${BLUE}[SETUP]${NC} $1"; }
ok()    { echo -e "${GREEN}[✔]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
err()   { echo -e "${RED}[✘]${NC} $1"; }
header(){ echo -e "\n${CYAN}═══════════════════════════════════════════${NC}"
          echo -e "${BOLD}  $1${NC}"
          echo -e "${CYAN}═══════════════════════════════════════════${NC}"; }

# ── Exact-match component check ──
contains() {
  for e in "${SELECTED[@]}"; do [[ "$e" == "$1" ]] && return 0; done
  return 1
}

# ── Parse args for non-interactive mode ──
parse_args() {
    for arg in "$@"; do
        case "$arg" in
            --all) SELECTED=("teacher" "conversation" "translator" "grammar" "datasets_en" "datasets_es" "datasets_ja" "models");;
            --quick) SELECTED=("teacher" "conversation" "grammar" "models");;
            --minimal) SELECTED=("teacher" "models");;
            --datasets) SELECTED=("datasets_en" "datasets_es" "datasets_ja");;
            --help)
                echo "Usage: ./setup.sh [OPTIONS]"
                echo "  --all       Install everything (full ~8GB download)"
                echo "  --quick     Teacher + Conversation + Grammar + models"
                echo "  --minimal   Teacher only + models"
                echo "  --datasets  Download full language datasets only"
                echo "  (no args)   Interactive menu"
                exit 0;;
        esac
    done
}

# ── Installation steps ──

install_system_packages() {
    header "[1/6] System Packages"
    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq && sudo apt-get install -y -qq \
            python3 python3-pip python3-venv build-essential cmake \
            curl wget git unzip libportaudio2 libsndfile1 \
            espeak-ng espeak-ng-data libespeak-ng-dev 2>&1 | tail -1
        ok "System packages (apt)"
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm python python-pip base-devel cmake \
            curl wget git unzip portaudio libsndfile espeak-ng
        ok "System packages (pacman)"
    else
        warn "Unknown package manager. Install: python3, pip, build-essential, cmake, espeak-ng"
    fi
}

install_venv() {
    header "[2/6] Python Virtual Environment"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        ok "venv created"
    else
        ok "venv already exists"
    fi
    source venv/bin/activate
    pip install --upgrade pip -q
}

install_core_packages() {
    header "[3/6] Python Packages"

    # PyTorch CUDA
    python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null && \
        ok "PyTorch CUDA already installed" || {
        info "Installing PyTorch CUDA 12.4 (~2.5GB)..."
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 -q
        ok "PyTorch CUDA installed"
    }

    # Core ML
    pip install faster-whisper onnxruntime silero-vad psutil pynvml numpy -q
    ok "Core ML packages"

    # TTS
    pip install kokoro-onnx piper-tts loguru scipy -q
    ok "TTS packages"

    # Translation
    pip install transformers sentencepiece protobuf -q
    ok "Translation packages"

    # Japanese
    pip install cutlet unidic-lite -q
    ok "Japanese packages"

    # Grammar app
    pip install flask flask-cors -q
    ok "Grammar app packages"
}

install_models() {
    header "[4/6] Model Files"

    if contains "models" || contains "teacher" || contains "conversation"; then
        mkdir -p models/onnx
        # Kokoro ONNX
        if [ ! -f "models/onnx/kokoro-v1.0.onnx" ]; then
            info "Downloading Kokoro ONNX (311MB)..."
            curl -L -o models/onnx/kokoro-v1.0.onnx \
                https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
            ok "Kokoro ONNX downloaded"
        else
            ok "Kokoro ONNX already exists"
        fi

        if [ ! -f "models/onnx/voices-v1.0.bin" ]; then
            info "Downloading Kokoro voices (27MB)..."
            curl -L -o models/onnx/voices-v1.0.bin \
                https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
            ok "Kokoro voices downloaded"
        else
            ok "Kokoro voices already exist"
        fi

        # Piper TTS models
        if [ ! -f "models/es_ES-sharvard-medium.onnx" ]; then
            info "Downloading Piper ES (77MB)..."
            curl -L -o models/es_ES-sharvard-medium.onnx \
                https://huggingface.co/rhasspy/piper-voices/resolve/main/es/es_ES/sharvard/medium/es_ES-sharvard-medium.onnx
            ok "Piper ES downloaded"
        else
            ok "Piper ES already exists"
        fi

        if [ ! -f "models/en_US-lessac-medium.onnx" ]; then
            info "Downloading Piper EN (63MB)..."
            curl -L -o models/en_US-lessac-medium.onnx \
                https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
            ok "Piper EN downloaded"
        else
            ok "Piper EN already exists"
        fi
    fi

    # Ollama model
    if contains "teacher" || contains "conversation"; then
        if command -v ollama &>/dev/null; then
            if ! ollama list 2>/dev/null | grep -q prometheus-orchestrator; then
                info "Downloading prometheus-orchestrator (Qwen3.5 4B, ~3.0GB)..."
                ollama pull prometheus-orchestrator
                ok "LLM model downloaded"
            else
                ok "LLM model already exists"
            fi
        else
            warn "Ollama not installed. Install: curl -fsSL https://ollama.com/install.sh | sh"
            warn "Then: ollama pull prometheus-orchestrator"
        fi
    fi
}

install_datasets() {
    header "[5/6] Language Datasets"

    # Count how many dataset types were selected
    DS_COUNT=0
    contains "datasets_en" && DS_COUNT=$((DS_COUNT + 1))
    contains "datasets_es" && DS_COUNT=$((DS_COUNT + 1))
    contains "datasets_ja" && DS_COUNT=$((DS_COUNT + 1))

    TATOEBA_LIMIT=$((DS_COUNT * 10000))
    [ "$TATOEBA_LIMIT" -gt 30000 ] && TATOEBA_LIMIT=100000

    if [ "$TATOEBA_LIMIT" -gt 0 ]; then
        info "Importing datasets (limit: ${TATOEBA_LIMIT} sentences)..."
        mkdir -p grammar_app/data/cache
        cd grammar_app/backend
        python3 seed_open_data.py --download --import-all --limit "$TATOEBA_LIMIT" 2>&1
        cd "$ROOT"
        ok "Datasets imported"
    else
        info "Skipping datasets (not selected)"
    fi

    # Fallback: seed minimal grammar data
    if contains "grammar" && [ "$TATOEBA_LIMIT" -eq 0 ]; then
        info "Seeding minimal grammar data..."
        cd grammar_app/backend
        python3 -c "from database import init_db, seed_default_data; init_db(); seed_default_data()"
        cd "$ROOT"
        ok "Minimal grammar data seeded"
    fi
}

verify_installation() {
    header "[6/6] Verification"

    python3 -c "
import sys
print(f'Python: {sys.version.split()[0]}')
checks = []
try:
    import torch; checks.append(f'PyTorch: {\"CUDA\" if torch.cuda.is_available() else \"CPU\"}')
except: checks.append('PyTorch: MISSING')
try:
    import faster_whisper; checks.append('Whisper: OK')
except: checks.append('Whisper: MISSING')
try:
    from kokoro_onnx import Kokoro; checks.append('Kokoro: OK')
except: checks.append('Kokoro: MISSING')
try:
    import piper; checks.append('Piper: OK')
except: checks.append('Piper: MISSING')
try:
    from transformers import MarianMTModel; checks.append('MarianMT: OK')
except: checks.append('MarianMT: MISSING')
try:
    import cutlet; checks.append('Cutlet: OK')
except: checks.append('Cutlet: MISSING')
for c in checks: print(f'  {c}')
"

    echo ""
    info "Model files:"
    for f in models/onnx/kokoro-v1.0.onnx models/onnx/voices-v1.0.bin \
             models/es_ES-sharvard-medium.onnx models/en_US-lessac-medium.onnx; do
        if [ -f "$f" ]; then
            ok "$f ($(du -h "$f" | cut -f1))"
        else
            warn "$f NOT FOUND"
        fi
    done

    if [ -f grammar_app/data/grammar.db ]; then
        ok "Grammar DB: $(du -h grammar_app/data/grammar.db | cut -f1)"
    fi

    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo -e "  ${BOLD}✅ Setup Complete${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════${NC}"
    echo ""
    echo "  Start with:"
    echo "    ./alex_voice_app.sh                   # Launcher (port 5000)"
    echo "    python3 server.py --port 3000         # Teacher"
    echo "    python3 translator.py                 # Translator (port 3003)"
    echo "    cd grammar_app/backend && python3 app.py  # Grammar (port 3004)"
    echo ""
    echo -e "  ${YELLOW}Disk usage:${NC}"
    du -sh models/ 2>/dev/null || echo "  models/: (empty)"
    du -sh grammar_app/data/ 2>/dev/null || echo "  grammar data/: (empty)"
}

# ── Main ──
main() {
    # Header
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║        ⚡ ALEX VOICE — Interactive Setup v4.0       ║"
    echo "║  Architecture: Ollama + Kokoro + Whisper + MarianMT  ║"
    echo "║  License: MIT + CC-BY datasets (Tatoeba, EDRDG)      ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"

    if [ $# -gt 0 ]; then
        parse_args "$@"
    else
        echo ""
        echo -e "  ${BOLD}Installation Options:${NC}"
        echo "  1) Full install (everything, ~6.5GB download)"
        echo "  2) Quick start (Teacher + Conversation + Grammar + models, ~4GB)"
        echo "  3) Minimal (Teacher only + models, ~3.5GB)"
        echo "  4) Datasets only (full language data, ~500MB)"
        echo "  5) Custom selection"
        echo "  q) Quit"
        echo ""
        read -p "  Select [1]: " choice
        choice=${choice:-1}
        case "$choice" in
            1) SELECTED=("teacher" "conversation" "translator" "grammar" "datasets_en" "datasets_es" "datasets_ja" "models");;
            2) SELECTED=("teacher" "conversation" "grammar" "models");;
            3) SELECTED=("teacher" "models");;
            4) SELECTED=("datasets_en" "datasets_es" "datasets_ja");;
            5)
                echo ""
                echo "  Select components (space-separated numbers):"
                echo "    1) 🎓 Teacher"
                echo "    2) 💬 Conversation"
                echo "    3) 🌍 Translator"
                echo "    4) 📝 Grammar App"
                echo "    5) 🔤 EN Datasets"
                echo "    6) 🟠 ES Datasets"
                echo "    7) 🗾 JA Datasets"
                echo "    8) 📦 Model files"
                read -p "  Numbers: " nums
                SELECTED=()
                for n in $nums; do
                    case $n in
                        1) SELECTED+=("teacher");;
                        2) SELECTED+=("conversation");;
                        3) SELECTED+=("translator");;
                        4) SELECTED+=("grammar");;
                        5) SELECTED+=("datasets_en");;
                        6) SELECTED+=("datasets_es");;
                        7) SELECTED+=("datasets_ja");;
                        8) SELECTED+=("models");;
                    esac
                done
                ;;
            q) echo "Aborted."; exit 0;;
        esac
    fi

    # Show selected
    echo ""
    echo -e "  ${BOLD}Selected:${NC}"
    for s in "${SELECTED[@]}"; do
        echo -e "    ${GREEN}●${NC} $s"
    done
    echo ""
    read -p "  Proceed with installation? [Y/n]: " confirm
    confirm=${confirm:-Y}
    if [[ "$confirm" =~ ^[Nn] ]]; then
        echo "Aborted."
        exit 0
    fi

    install_system_packages
    install_venv
    install_core_packages
    install_models
    install_datasets
    verify_installation
}

main "$@"
