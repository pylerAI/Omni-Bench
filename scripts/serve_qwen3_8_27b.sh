#!/usr/bin/env bash
set -euo pipefail

# Qwen3.8-27B (dense, vision+text). Audio is handled out-of-band by Whisper,
# so this server never decodes an audio track.

# Use the repo venv's vllm directly (no reliance on PATH / an activated env).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM="${VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}"
[ -x "$VLLM" ] || VLLM="vllm"

# FlashInfer JIT-compiles some kernels at load, needing ninja + nvcc on PATH.
# Derive the bin dir from the vllm we actually run: the venv is not always the
# repo's own .venv (on an NFS home it must live elsewhere), and hardcoding
# $REPO_ROOT/.venv left ninja off PATH, which fails engine init with
# "FileNotFoundError: 'ninja'" only once the first sampling kernel is built.
VENV_BIN="$(cd "$(dirname "$VLLM")" 2>/dev/null && pwd)"
[ -n "$VENV_BIN" ] && export PATH="$VENV_BIN${PATH:+:$PATH}"
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
[ -d /usr/local/cuda ] && export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

MODEL_PATH="${MODEL_PATH:-/gpfs/public/artifacts/models/Qwen/Qwen3.8-27B/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3.8-27B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/gpfs/public/datasets}"

if [ -z "${DATA_PARALLEL_SIZE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "-1" ]; then
    DATA_PARALLEL_SIZE=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '[0-9]')
  else
    DATA_PARALLEL_SIZE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  fi
  [ "${DATA_PARALLEL_SIZE:-0}" -ge 1 ] 2>/dev/null || DATA_PARALLEL_SIZE=1
fi

ARGS=(
  serve "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --host "${HOST}"
  --port "${PORT}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --data-parallel-size "${DATA_PARALLEL_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --trust-remote-code
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
)
if [ -n "${MAX_MODEL_LEN:-}" ]; then
  ARGS+=(--max-model-len "${MAX_MODEL_LEN}")
fi

exec "$VLLM" "${ARGS[@]}"
