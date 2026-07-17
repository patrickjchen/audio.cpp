#!/usr/bin/env bash
# Launch the audio.cpp WebUI on Linux/macOS (POSIX counterpart of run_webui.bat).
#
# The WebUI starts/switches audiocpp_server on demand — pick a model in the UI and
# click load; no need to start a server separately. Backend (cuda|cpu) is auto-detected
# by webui.py from nvidia-smi and the available build; override with AUDIOCPP_BACKEND=gpu|cpu.
# UI language defaults to English; set AUDIOCPP_LANG=zh for Chinese.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEBUI_DIR="$ROOT/webui"

# Locate a Python that has the deps (gradio/requests/torch/safetensors/...).
PY=""
for cand in \
    "${AUDIOCPP_PYTHON:-}" \
    "$ROOT/venv/bin/python" \
    "$ROOT/.venv/bin/python" \
    "$(command -v python3 || true)" \
    "$(command -v python || true)"; do
    if [ -n "$cand" ] && [ -x "$cand" ]; then PY="$cand"; break; fi
done

if [ -z "$PY" ]; then
    echo "[run_webui] no Python found. Create a venv and install deps:" >&2
    echo "  python3 -m venv venv && ./venv/bin/pip install gradio requests torch safetensors pyyaml huggingface_hub" >&2
    exit 1
fi

echo "[run_webui] python: $PY"
echo "[run_webui] backend: ${AUDIOCPP_BACKEND:-auto}  language: ${AUDIOCPP_LANG:-en}"
echo "[run_webui] UI -> http://127.0.0.1:7860"
exec "$PY" "$WEBUI_DIR/webui.py"
