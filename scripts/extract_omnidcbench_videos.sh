#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-/gpfs/public/datasets/OmniDCBench}"
VIDEO_DIR="${VIDEO_DIR:-${DATA_ROOT}/Video}"

mkdir -p "${VIDEO_DIR}"

shopt -s nullglob

movie_parts=("${DATA_ROOT}"/Movie.tar.gz.*)
if (( ${#movie_parts[@]} > 0 )); then
  echo "Extracting Movie videos -> ${VIDEO_DIR}"
  cat "${movie_parts[@]}" | tar -xzf - -C "${VIDEO_DIR}"
fi

youtube_parts=("${DATA_ROOT}"/Youtube.tar.gz.*)
if (( ${#youtube_parts[@]} > 0 )); then
  echo "Extracting Youtube videos -> ${VIDEO_DIR}/Youtube"
  mkdir -p "${VIDEO_DIR}/Youtube"
  cat "${youtube_parts[@]}" | tar -xf - -C "${VIDEO_DIR}/Youtube"
fi
