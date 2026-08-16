#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"
source ./.venv/bin/activate

# 使用示例：
#   FOOT=sole bash ghost_train.sh
#   ACTION_CLIP=3 ACTION_SCALE=0.0833333333 \
#     ACTION_RATE_WEIGHT=-0.0111111111 bash ghost_train.sh
#
# FOOT：sole 为 10 mm 凸包脚底，xml 为原始圆柱导轨，default 使用任务默认值。
TASK_ID="TK3-Ghost-Tracking-QRef-Prototype"
FOOT="${FOOT:-xml}"
MOTION_FILE="${MOTION_FILE:-datasets/wu老师/Skeleton2_mjlab.npz}"
ACTION_CLIP="${ACTION_CLIP:-10.0}"
ACTION_SCALE="${ACTION_SCALE:-0.5}"
ACTION_RATE_WEIGHT="${ACTION_RATE_WEIGHT:-}"
RUN_NAME="${RUN_NAME:-qref-residual-${FOOT}-clip${ACTION_CLIP}}"

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

action_args=(
  --agent.clip-actions "${ACTION_CLIP}"
  --env.actions.joint-pos.residual-clip "${ACTION_CLIP}"
)
if [[ -n "${ACTION_SCALE}" ]]; then
  action_args+=(--env.actions.joint-pos.scale "${ACTION_SCALE}")
fi
if [[ -n "${ACTION_RATE_WEIGHT}" ]]; then
  action_args+=(
    --env.rewards.action-rate-l2.weight "${ACTION_RATE_WEIGHT}"
  )
fi

python scripts/train.py "${TASK_ID}" \
  --motion-file "${MOTION_FILE}" \
  "${foot_args[@]}" \
  "${action_args[@]}" \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 50000 \
  --agent.save-interval 1000 \
  --agent.seed 3407 \
  --agent.logger wandb \
  --agent.wandb-project tk3-tracking-mjlab \
  --agent.run-name "${RUN_NAME}" \
  "$@"