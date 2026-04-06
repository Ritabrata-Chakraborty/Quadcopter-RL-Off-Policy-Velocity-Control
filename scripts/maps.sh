#!/bin/bash
# 1) Generate simple empty-room maps under maps_train/simple and maps_test/simple.
# 2) Run goals.py on every maps_train/<set>/ that contains PNGs (skips *_goals dirs),
#    writing to maps_train/<set>_goals/.
#
# Run from repo root: ./scripts/maps.sh
# Requires: Python 3, Pillow, NumPy.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET="${REPO_ROOT}/dataset"
MAP_GEN="${DATASET}/map_gen"
COLORS="${DATASET}/colors.json"
MAPS_TRAIN="${DATASET}/maps_train"
MAPS_EVAL="${DATASET}/maps_eval"

MIN_GOALS="${MIN_GOALS:-5}"
MAX_GOALS="${MAX_GOALS:-5}"
GOAL_SEED="${GOAL_SEED:-42}"

log() { echo "[maps] $*"; }

main() {
  local d base out

  [[ -d "${DATASET}" ]] || { echo "Missing ${DATASET}" >&2; exit 1; }
  [[ -f "${COLORS}" ]] || { echo "Missing ${COLORS}" >&2; exit 1; }

  log "1/3 simple.py → ${MAPS_TRAIN}/simple, ${MAPS_EVAL}/simple"
  python3 "${MAP_GEN}/simple.py"

  log "2/3 goals for each map set under ${MAPS_TRAIN}/"
  shopt -s nullglob
  for d in "${MAPS_TRAIN}"/*/; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d")
    [[ "$base" == *_goals ]] && continue
    imgs=("${d}"*.png "${d}"*.PNG)
    if ((${#imgs[@]} == 0)); then
      log "skip ${base} (no PNGs)"
      continue
    fi
    out="${MAPS_TRAIN}/${base}_goals"
    log "goals: ${base}/ → ${base}_goals/"
    python3 "${MAP_GEN}/goals.py" \
      --input-dir "$d" \
      --output-dir "$out" \
      --colors "${COLORS}" \
      --min-goals "${MIN_GOALS}" \
      --max-goals "${MAX_GOALS}" \
      --seed "${GOAL_SEED}"
  done
  shopt -u nullglob

  log "3/3 goals for each map set under ${MAPS_EVAL}/"
  shopt -s nullglob
  for d in "${MAPS_EVAL}"/*/; do
    [[ -d "$d" ]] || continue
    base=$(basename "$d")
    [[ "$base" == *_goals ]] && continue
    imgs=("${d}"*.png "${d}"*.PNG)
    if ((${#imgs[@]} == 0)); then
      log "skip ${base} (no PNGs)"
      continue
    fi
    out="${MAPS_EVAL}/${base}_goals"
    log "goals: ${base}/ → ${base}_goals/"
    python3 "${MAP_GEN}/goals.py" \
      --input-dir "$d" \
      --output-dir "$out" \
      --colors "${COLORS}" \
      --min-goals "${MIN_GOALS}" \
      --max-goals "${MAX_GOALS}" \
      --seed "${GOAL_SEED}"
  done
  shopt -u nullglob
  log "done."
}

main "$@"
