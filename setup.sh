#!/usr/bin/env bash
# ═══════════════════════════════════════════════════════════════
#  Alex Translator — Setup (v5: unificado, llama.cpp)
#
#  Este proyecto forma parte del ecosistema S2S: comparte venv y
#  modelos con teacher/conversation/grammar y usa llama.cpp (no Ollama).
#
#  - Si estás dentro del monorepo S2S → delega en ../setup.sh
#  - Si clonaste este repo solo → instala de forma autónoma (requirements.txt)
#
#  Uso:  ./setup.sh [--apps|--model-llm|--check]
# ═══════════════════════════════════════════════════════════════
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Modo monorepo: delegar al setup unificado ─────────────────
if [ -f "$ROOT/../setup.sh" ] && [ -x "$ROOT/../alex" ]; then
    echo "→ Ecosistema S2S detectado: delegando al setup unificado (../setup.sh)."
    echo "  (venv compartido, modelos en ~/.local/share/alex/, motor llama.cpp)"
    echo ""
    exec bash "$ROOT/../setup.sh" "$@"
fi

# ── Modo autónomo (repo clonado solo) ─────────────────────────
echo "╔══════════════════════════════════════════════╗"
echo "║  Alex Translator — Setup autónomo            ║"
echo "╚══════════════════════════════════════════════╝"

if [ ! -d "$ROOT/.venv" ]; then
    echo "  Creando .venv..."
    python3 -m venv "$ROOT/.venv"
fi
PY="$ROOT/.venv/bin/python"
"$PY" -m pip install --quiet --upgrade pip
"$PY" -m pip install --quiet -r "$ROOT/requirements.txt"

echo ""
echo "✅ Listo. Para lanzar:"
echo "    source .venv/bin/activate && python3 translator.py   # puerto 3003"
echo ""
echo "⚠️  El chat necesita el motor LLM local (llama.cpp); para el ecosistema"
echo "    completo (hub, modelos, doctor) usa el monorepo S2S."
echo ""
