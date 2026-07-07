#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-/gpfs/public/artifacts/models/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8/}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
ALLOWED_LOCAL_MEDIA_PATH="${ALLOWED_LOCAL_MEDIA_PATH:-/gpfs/public/datasets}"
export VLLM_MAX_AUDIO_DECODE_DURATION_S="${VLLM_MAX_AUDIO_DECODE_DURATION_S:-3600}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-$(python - <<'PY'
import os
import subprocess

visible = os.environ.get("CUDA_VISIBLE_DEVICES")
if visible:
    devices = [item.strip() for item in visible.split(",") if item.strip()]
    print(len(devices) if devices and devices != ["-1"] else 1)
else:
    result = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, check=False)
    print(sum(1 for line in result.stdout.splitlines() if line.startswith("GPU ")) or 1)
PY
)}"

exec vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --data-parallel-size "${DATA_PARALLEL_SIZE}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code \
  --moe-backend triton \
  --no-enable-flashinfer-autotune \
  --allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}"
