#!/usr/bin/env bash
# Start the E2B llama-server from the native CUDA build of llama.cpp (no docker, no sudo).
# Same native binary as start_e4b_server.sh; see that script's header for the build step.
#
# Stop: Ctrl-C, or  pkill -f 'build-cuda/bin/llama-server'
set -e

ROOT=~/yuan_vlm
BIN_DIR="$ROOT/build/llama.cpp/build-cuda/bin"

if [ ! -x "$BIN_DIR/llama-server" ]; then
  echo "error: $BIN_DIR/llama-server not found — build it first (see start_e4b_server.sh)." >&2
  exit 1
fi

export LD_LIBRARY_PATH="$BIN_DIR:$LD_LIBRARY_PATH"

exec "$BIN_DIR/llama-server" \
  -m "$ROOT/models/google_gemma-4-E2B-it-Q4_K_M.gguf" \
  --mmproj "$ROOT/models/mmproj-google_gemma-4-E2B-it-f16.gguf" \
  -c 4096 \
  -n 2048 \
  -ngl 99 \
  -fit off \
  --host 0.0.0.0 \
  --port 8080 \
  --parallel 1 \
  --reasoning off
