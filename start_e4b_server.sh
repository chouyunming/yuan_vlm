#!/usr/bin/env bash
# Start the E4B llama-server from the native CUDA build of llama.cpp (no docker, no sudo).
#
# Build (once):
#   cd ~/yuan_vlm/build/llama.cpp
#   cmake -B build-cuda -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=87 \
#         -DLLAMA_CURL=OFF -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
#   cmake --build build-cuda --target llama-server -j"$(nproc)"
#
# Stop: Ctrl-C, or  pkill -f 'build-cuda/bin/llama-server'
set -e

ROOT=~/yuan_vlm
BIN_DIR="$ROOT/build/llama.cpp/build-cuda/bin"

if [ ! -x "$BIN_DIR/llama-server" ]; then
  echo "error: $BIN_DIR/llama-server not found — build it first (see header)." >&2
  exit 1
fi

export LD_LIBRARY_PATH="$BIN_DIR:$LD_LIBRARY_PATH"

exec "$BIN_DIR/llama-server" \
  -m "$ROOT/models/gemma-4-E4B-it-Q4_K_M.gguf" \
  --mmproj "$ROOT/models/mmproj-gemma-4-E4B-it-Q8_0.gguf" \
  -c 2048 \
  -n 1024 \
  -ngl 99 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --host 0.0.0.0 \
  --port 8080 \
  --parallel 1 \
  --temp 0 \
  --seed 1 \
  --reasoning off
