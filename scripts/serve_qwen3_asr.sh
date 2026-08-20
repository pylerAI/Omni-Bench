#!/usr/bin/env bash
set -euo pipefail

# Qwen3-ASR-1.7B on its own port. Serve it *before or after* the model under
# test, never alongside: sharing GPUs with a vLLM instance at 0.9 utilisation
# starves its workers and kills the engine.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM="${VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}"
[ -x "$VLLM" ] || VLLM="vllm"

# FlashInfer JIT needs ninja + nvcc on PATH; derive from the vllm we run.
VENV_BIN="$(cd "$(dirname "$VLLM")" 2>/dev/null && pwd)"
[ -n "$VENV_BIN" ] && export PATH="$VENV_BIN${PATH:+:$PATH}"
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
[ -d /usr/local/cuda ] && export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

MODEL_PATH="${MODEL_PATH:-/gpfs/public/artifacts/models/Qwen/Qwen3-ASR-1.7B/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-ASR-1.7B}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8002}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

if [ -z "${DATA_PARALLEL_SIZE:-}" ]; then
  if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES}" != "-1" ]; then
    DATA_PARALLEL_SIZE=$(printf '%s' "${CUDA_VISIBLE_DEVICES}" | tr ',' '\n' | grep -c '[0-9]')
  else
    DATA_PARALLEL_SIZE=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
  fi
  [ "${DATA_PARALLEL_SIZE:-0}" -ge 1 ] 2>/dev/null || DATA_PARALLEL_SIZE=1
fi

exec "$VLLM" serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code
