#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source ./.venv/bin/activate

# 使用示例：
#   CHECKPOINT_FILE=logs/rsl_rl/.../model_20000.pt bash ghost_play.sh
#   VIEWER=native FOOT=sole bash ghost_play.sh
#
# 播放 PyTorch checkpoint 时，从同级 params 目录恢复训练时的动作参数。
TASK_ID="TK3-Ghost-Tracking-QRef-Prototype"
FOOT="${FOOT:-xml}"
MOTION_FILE="${MOTION_FILE:-datasets/wu老师/Skeleton2_mjlab.npz}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-logs/rsl_rl/tk3_ghost_qref_residual_prototype/2026-08-15_18-29-16_qref-residual-xml-clip10.0/model_13000.pt}"
VIEWER="${VIEWER:-viser}"
NUM_ENVS="${NUM_ENVS:-1}"

foot_args=()
case "${FOOT}" in
  sole|xml)
    foot_args=(--foot "${FOOT}")
    ;;
  default)
    ;;
  *)
    echo "FOOT must be one of: sole, xml, default" >&2
    exit 2
    ;;
esac

python scripts/play.py "${TASK_ID}" \
  --num-envs "${NUM_ENVS}" \
  --viewer "${VIEWER}" \
  "${foot_args[@]}" \
  --motion-file "${MOTION_FILE}" \
  --checkpoint-file "${CHECKPOINT_FILE}" \
  --restore-action-params True \
  "$@"
