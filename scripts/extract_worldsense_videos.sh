#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/gpfs/public/datasets/WorldSense}"
VIDEO_DIR="${VIDEO_DIR:-${DATA_ROOT}/videos}"

mkdir -p "${VIDEO_DIR}"

shopt -s nullglob
for archive in "${DATA_ROOT}"/worldsense_videos_*.zip; do
  echo "Extracting ${archive} -> ${VIDEO_DIR}"
  unzip -n "${archive}" -d "${VIDEO_DIR}"
done

if [[ -f "${DATA_ROOT}/worldsense_subtitles.zip" ]]; then
  SUBTITLE_DIR="${SUBTITLE_DIR:-${DATA_ROOT}/subtitles}"
  mkdir -p "${SUBTITLE_DIR}"
  echo "Extracting ${DATA_ROOT}/worldsense_subtitles.zip -> ${SUBTITLE_DIR}"
  unzip -n "${DATA_ROOT}/worldsense_subtitles.zip" -d "${SUBTITLE_DIR}"
fi
