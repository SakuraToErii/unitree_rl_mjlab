#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source ./.venv/bin/activate

# 使用示例：
#   bash ghost_record.sh
#   OUTPUT_FILE=datasets/my_motion_ghost.npz bash ghost_record.sh
#   INITIAL_FOOT_PENETRATION_MM=1 FOOT=sole bash ghost_record.sh
TASK_ID="TK3-Ghost-Tracking-QRef-Prototype"
FOOT="${FOOT:-xml}"
MOTION_FILE="${MOTION_FILE:-datasets/wu老师/Skeleton2_mjlab.npz}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-logs/rsl_rl/tk3_ghost_qref_residual_prototype/2026-08-15_18-29-16_qref-residual-xml-clip10.0/model_13000.pt}"
OUTPUT_FILE="${OUTPUT_FILE:-datasets/wu老师/Skeleton2_mjlab_ghost.npz}"
INITIAL_FOOT_PENETRATION_MM="${INITIAL_FOOT_PENETRATION_MM:-3}"

case "${FOOT}" in
  sole|xml)
    ;;
  *)
    echo "FOOT must be one of: sole, xml" >&2
    exit 2
    ;;
esac

INITIAL_FOOT_PENETRATION_M="$(
  python -c \
    'import sys; print(float(sys.argv[1]) / 1000.0)' \
    "${INITIAL_FOOT_PENETRATION_MM}"
)"

python scripts/record_ghost_npz.py "${TASK_ID}" \
  --checkpoint-file "${CHECKPOINT_FILE}" \
  --motion-file "${MOTION_FILE}" \
  --output-file "${OUTPUT_FILE}" \
  --foot "${FOOT}" \
  --initial-foot-penetration-m "${INITIAL_FOOT_PENETRATION_M}" \
  "$@"
