#!/usr/bin/env bash
# Nemotron-3.5-Super-VL full runs (remote shared vLLM; never --serve).
#   bash scripts/run_nemotron_super_vl.sh <bf16|nvfp4> <nothink|think> [bench ...]
# bench default: videomme worldsense omnivideobench av_speakerbench (sequential).
# Resumable: rerunning skips finished items and retries error records.
# Launch in background:
#   nohup bash scripts/run_nemotron_super_vl.sh bf16 nothink \
#     > /gpfs/public/artifacts/ail/nemotron-compare/omni-bench/logs/bf16_nothink.chain.log 2>&1 &
set -uo pipefail
VARIANT=${1:?bf16|nvfp4}; MODE=${2:?nothink|think}; shift 2
BENCHES=("$@"); [ ${#BENCHES[@]} -eq 0 ] && BENCHES=(videomme worldsense omnivideobench av_speakerbench)

REPO=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-/gpfs/private/ail/venvs/omni-bench}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1   # NDA model: no outbound calls besides the vLLM server
CFG=configs/nemotron_super_vl
LOGS=/gpfs/public/artifacts/ail/nemotron-compare/omni-bench/logs
mkdir -p "$LOGS"

URL=http://vllm-nemotron-3-5-super-vl-$VARIANT.vllm:8000/v1/models
curl -sf -m 10 "$URL" -o /dev/null || { echo "!! server unreachable: $URL"; exit 1; }

for B in "${BENCHES[@]}"; do
  LOG=$LOGS/${VARIANT}_${MODE}_$B.log
  echo "### START $VARIANT/$MODE/$B $(date -u +%FT%TZ) → $LOG"
  uv run --no-sync omni-bench run --config $CFG/${VARIANT}_$MODE.yaml \
    --benchmark-config $CFG/bench_$MODE.yaml --benchmark "$B" >> "$LOG" 2>&1
  RC=$?
  if [ $RC -eq 0 ]; then echo "### DONE $B"; grep -E "samples in" "$LOG" | tail -1
  else echo "### FAILED $B rc=$RC"; tail -15 "$LOG"; fi
done
echo "### CHAIN DONE $VARIANT/$MODE $(date -u +%FT%TZ)"
