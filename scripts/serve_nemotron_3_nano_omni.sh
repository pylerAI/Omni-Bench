#!/usr/bin/env bash
set -euo pipefail

# Use the repo venv's vllm directly (no reliance on PATH / an activated env).
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VLLM="${VLLM_BIN:-$REPO_ROOT/.venv/bin/vllm}"
[ -x "$VLLM" ] || VLLM="vllm"

# FlashInfer JIT-compiles some kernels at load, needing ninja + nvcc on PATH.
export PATH="$REPO_ROOT/.venv/bin${PATH:+:$PATH}"
[ -d /usr/local/cuda/bin ] && export PATH="/usr/local/cuda/bin:$PATH"
[ -d /usr/local/cuda ] && export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"

MODEL_PATH="${MODEL_PATH:-/gpfs/public/artifacts/models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/gpfs/public/datasets}"
# MoE backend: "triton" is stable for FP8/BF16 experts; NVFP4 experts need the
# auto-selected FP4 kernels, so set MOE_BACKEND=auto (or empty) for that variant.
MOE_BACKEND="${MOE_BACKEND:-triton}"
export VLLM_MAX_AUDIO_DECODE_DURATION_S="${VLLM_MAX_AUDIO_DECODE_DURATION_S:-3600}"
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
  --no-enable-flashinfer-autotune
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
)
if [ -n "${MOE_BACKEND}" ] && [ "${MOE_BACKEND}" != "auto" ]; then
  ARGS+=(--moe-backend "${MOE_BACKEND}")
fi

exec "$VLLM" "${ARGS[@]}"
