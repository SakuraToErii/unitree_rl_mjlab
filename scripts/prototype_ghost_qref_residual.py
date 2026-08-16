#!/usr/bin/env python3
"""用于验证 Ghost q_ref 残差与奖励优先级的交互式终端原型。

运行：
  uv run python scripts/prototype_ghost_qref_residual.py
"""

from __future__ import annotations

import argparse
import sys
import termios
import tty

from src.tasks.ghost.prototypes.qref_residual_model import (
  ACTION_SCALE_RAD,
  ENDPOINT_NAMES,
  Q_REFERENCE,
  PrototypeState,
  evaluate,
  reduce_state,
)


BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
CLEAR = "\x1b[2J\x1b[H"


def render(state: PrototypeState) -> None:
  result = evaluate(state)
  selected = state.selected_endpoint
  print(CLEAR, end="")
  print(f"{BOLD}Ghost q_ref residual prototype{RESET}")
  print(
    f"{DIM}Toy model: left foot -5 cm, right foot +3 cm, "
    f"supporting fist/TCP +3 cm.{RESET}\n"
  )
  print(f"{BOLD}Current state{RESET}")
  print(f"  action baseline       {state.baseline}")
  print(f"  residual scale        {ACTION_SCALE_RAD:.2f} rad/action")
  print(f"  selected endpoint     {ENDPOINT_NAMES[selected]}")
  print(f"  tracking weight       {state.tracking_weight:g}")
  print(f"  feasibility weight    {state.feasibility_weight:g}")
  print()
  print(
    f"{BOLD}{'endpoint':18} {'q_ref':>8} {'action':>8} "
    f"{'q_target':>9} {'clearance':>11}{RESET}"
  )
  for index, name in enumerate(ENDPOINT_NAMES):
    marker = ">" if index == selected else " "
    print(
      f"{marker} {name:16} {Q_REFERENCE[index]:8.3f} "
      f"{state.action[index]:8.3f} {result.target[index]:9.3f} "
      f"{result.endpoint_clearance_m[index] * 100:9.2f} cm"
    )

  print(f"\n{BOLD}Objective (lower is preferred){RESET}")
  print(f"  style RMSE            {result.style_rmse_rad:.4f} rad")
  print(f"  target-step RMSE      {result.target_step_rmse_rad:.4f} rad")
  print(f"  tracking cost         {result.tracking_cost:.3f}")
  print(f"  physical cost         {result.feasibility_cost:.3f}")
  print(f"  smoothness cost       {result.smoothness_cost:.3f}")
  print(f"  weighted total        {result.total_cost:.3f}")
  quality = "PASS" if result.quality_passed else "FAIL"
  print(
    f"  rollout quality gate  {BOLD}{quality}{RESET} "
    f"{DIM}(penetration {result.penetration_m * 100:.2f} cm, "
    f"support gap {result.support_gap_m * 100:.2f} cm){RESET}"
  )

  print(f"\n{BOLD}Keys{RESET}")
  print(
    f"  {BOLD}[b]{RESET} baseline q0/q_ref   "
    f"{BOLD}[1-3]{RESET} select endpoint   "
    f"{BOLD}[j/k]{RESET} residual -/+"
  )
  print(
    f"  {BOLD}[c]{RESET} best contact correction   "
    f"{BOLD}[x]{RESET} zero residual   "
    f"{BOLD}[t/p]{RESET} cycle tracking/physics weight   "
    f"{BOLD}[q]{RESET} quit"
  )
  print(
    f"\n{DIM}The weighted objective can trade style for bad physics; "
    f"the rollout gate cannot.{RESET}"
  )


def read_key() -> str:
  fd = sys.stdin.fileno()
  previous = termios.tcgetattr(fd)
  try:
    tty.setraw(fd)
    return sys.stdin.read(1)
  finally:
    termios.tcsetattr(fd, termios.TCSADRAIN, previous)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument(
    "--once",
    action="store_true",
    help="仅渲染初始帧后退出，用于冒烟检查。",
  )
  args = parser.parse_args()

  state = PrototypeState()
  while True:
    render(state)
    if args.once:
      return
    if not sys.stdin.isatty():
      raise SystemExit("Interactive mode requires a TTY; use --once to render.")
    key = read_key().lower()
    if key == "q":
      print()
      return
    state = reduce_state(state, key)


if __name__ == "__main__":
  main()
