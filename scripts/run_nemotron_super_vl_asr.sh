#!/usr/bin/env bash
# Nemotron-3.5-Super-VL ASR comparison chain: BF16 no-think with Nemotron streaming
# ASR transcripts (cache-only) instead of Whisper. Same bench params as bench_nothink.yaml.
#   bash scripts/run_nemotron_super_vl_asr.sh [bench ...]
# bench default: worldsense av_speakerbench omnivideobench (sequential). Resumable.
# Launch in background:
#   nohup bash scripts/run_nemotron_super_vl_asr.sh \
#     > /gpfs/public/artifacts/ail/nemotron-compare/omni-bench/logs/bf16_nothink_nemotron_asr.chain.log 2>&1 &
set -uo pipefail
BENCHES=("$@"); [ ${#BENCHES[@]} -eq 0 ] && BENCHES=(worldsense av_speakerbench omnivideobench)

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/gpfs/private/ail/venvs/omni-bench}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # NDA model: no outbound calls besides the vLLM server
CFG=configs/nemotron_super_vl
TAG=bf16_nothink_nemotron_asr
LOGS=/gpfs/public/artifacts/ail/nemotron-compare/omni-bench/logs
mkdir -p "$LOGS"

URL=http://vllm-nemotron-3-5-super-vl-bf16.vllm:8000/v1/models
curl -sf -m 10 "$URL" -o /dev/null || { echo "!! server unreachable: $URL"; exit 1; }

for B in "${BENCHES[@]}"; do
  LOG=$LOGS/${TAG}_$B.log
  echo "### START $TAG/$B $(date -u +%FT%TZ) → $LOG"
  uv run --no-sync omni-bench run --config $CFG/$TAG.yaml \
    --benchmark-config $CFG/bench_nothink.yaml --benchmark "$B" >> "$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then echo "### DONE $B"; grep -E "samples in" "$LOG" | tail -1
  else echo "### FAILED $B rc=$RC"; tail -15 "$LOG"; fi
done
echo "### CHAIN DONE $TAG $(date -u +%FT%TZ)"
