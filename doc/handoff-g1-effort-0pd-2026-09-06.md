> 恢复说明（2026-11-22）：原文写在 `/tmp/handoff-g1-effort-0pd-2026-09-06.md`，没有进过提交。它记录的是 `8106a6c` 当天的 effort 奖励和训练约束，不是当前分支。

# Handoff: G1 0-PD effort locomotion (reward / PPO layout)

Date: 2026-09-06
Workspace: `/home/ordis/projects/unitree_rl_mjlab`
Branch: `main` = `origin/main`
HEAD: `8106a6c` — *Replace effort pose kernel with Isaac L1 split and stronger arms.*

Speak Chinese unless the user switches. Do not retune/train-patch unless asked. Do not kill running trains unless asked. When disabling a reward, **comment it out and keep original weights in the comment** — do not set `weight=0.0` as a substitute (except Go2 arms/waists, which have no such joints and are legitimately `0.0`). After code edits: `graphify update .`. Live `resume.sh` does not hot-reload.

Next session will most likely: watch a **new** train on this HEAD (L1 split + obs norm), or iterate arm L1 / gait if zombie/shuffle remains.

## Do not redo

Older plant / ONNX / 0-PD notes live in untracked local files (gitignored research/handoff). Do not recopy them. Current product, already decided:

- Plant: `IdealPdActuatorCfg(stiffness=0, damping=0)` + `JointEffortAction`. G1 also has **passive** MuJoCo `viscous_damping=0.05` and `frictionloss=0.2` (wrists 0.1) forwarded by `zero_pd.py`.
- Action: \(\tau = \mathrm{clip}(a \cdot 0.25 \cdot Y_1,\ \pm Y_1)\), `offset=0`. Saturation at `|a|=4`. `EFFORT_ACTION_SCALE_FRACTION = 0.25`.
- Actor: `ResidualMlpModel` / `ResidualMhaModel`, `output_gain=0.01`. MHA is 4+1 late fusion. `std_type="scalar"`, `init_std=1`, `std_range=(1e-3, 2.0)`, `clip_actions=None`, `entropy_coef=0.001`.
- Deploy C++ is still `JointPositionAction`; ONNX ≠ robot.
- Command curriculum and play `lin_vel_x ±2.0`: see `src/tasks/effort/velocity_env_cfg.py`. Train uses tyro; `--agent.resume True` needs `FlagConversionOff`.

## What this conversation changed (on origin/main)

| Commit | Why |
|---|---|
| `a6d9980` | Replaced Isaac L1 with mjlab `variable_posture` + `stand_still`; raised gait/air-time. Copied pose functions **into** `src/tasks/effort/mdp/rewards.py` (do not import from `src.tasks.velocity`). |
| `2e12422` | Restored `base_height_l2` with **2 cm deadzone**; `feet_air_time` weight **2.0** (normalized by threshold). |
| `ad70db3` | Actor/critic/PPO assembly moved to per-robot `config/*/rl_cfg.py` like velocity. `obs_normalization=True` for **all 12** effort tasks (MLP and MHA). `ppo_cfg.py` only has `ResidualMlpModelCfg` / `ResidualMhaModelCfg`. |
| `8106a6c` | **Current:** comment `pose`; restore Isaac L1 split; arms **-0.3**. |

## Current reward intent (G1 effort)

Active: tracking 10 / yaw 5; `stand_still` -1 (cmd &lt; 0.1); `base_height` -10, deadzone 0.02, G1 target `EFFORT_STANDING_ROOT_HEIGHT`; `foot_gait` 5.0 period 0.6; `feet_air_time` 2.0 threshold 0.30 **÷ T**; Isaac L1 arms **-0.3** (shoulder/elbow/wrist), waists **-1**, legs **-1** (**hip roll/yaw only** — hip pitch and knee free); plus the existing clearance/slip/AM/self-collision/energy/action-rate terms.

Commented (values preserved): `variable_posture` weight 1.0 and per-robot std tables.

Go2: L1 arms/waists weight **0**; legs `.*_hip_joint`.

Tests: `tests/test_effort_task.py` (`test_g1_effort_rewards_use_walkable_pose_split`, `test_other_effort_variants_configure_new_rewards`, PPO `load_rl_cfg` obs-norm loop).

## Diagnosis the next agent must not unlearn

**Zombie walk + `Episode_Reward/pose` ≈ 0.0015 was not a joint-order bug.** After `env.reset()`, `q == default_joint_pos` and pose at the standing keyframe is `1.0`. Std tables map by name.

Cause: `variable_posture` is \(\exp(-\mathrm{mean}_i e_i^2/\sigma_i^2)\) over **all 29 joints**, peak at standing. Walking only inflates σ; optimum is still “don’t move.” Elbow defaults are ~0.93 / 0.97 rad with walking σ=0.1. 0-PD has no position loop, gravity drops the arms. Two hanging elbows alone: \(\exp(-6.22)\approx 0.002\). That matches the log. PD mjlab can keep `pose` because PD holds the elbows; this plant cannot.

`Episode_Reward/*` = \(\sum(\mathrm{raw}\times\mathrm{weight}\times dt)/T_{\max}\) with \(T_{\max}\) usually **20 s**. Short falls **dilute** the number. Teacher’s “pose 0.6+” assumed surviving walkers under the old kernel — obsolete now that pose is commented.

Isaac L1 is the structural fix: arms visible on their own term; sagittal swing not pinned. **-0.3 on arms is a first try** (Isaac PD used -0.1; that is too weak here). It will not by itself fix 碎步 — that is still gait/air-time vs tracking.

## PPO / resume trap

`obs_normalization=True` as of `ad70db3`. Old checkpoints (Identity normalizer) **must not be resumed** onto this code.

`resume.sh` is gitignored. Last seen contents still pointed at:

- task `Unitree-G1-Effort-Flat-MHA`
- `--agent.load-run 2026-09-04_21-12-37_g1-effort-flat-mha`
- `--agent.load-checkpoint model_27000.pt`

That run predates L1 split, deadzone height, and obs-norm. A terminal may still be `bash resume.sh`; that process will **not** pick up `8106a6c` until restart, and should not resume that ckpt onto current main.

Play artifact from before this reward pass: `logs/rsl_rl/g1_effort_mha/2026-09-04_21-12-37_g1-effort-flat-mha/` (e.g. `model_26000.pt`, 1 m/s play clip). Cadence at 1 m/s was ~3.4 Hz vs 0.6 s clock wanting 1.67 Hz — 碎步 history.

## What to watch on a fresh train

- `Episode_Reward/joint_deviation_arms` should go clearly negative if elbows hang (not sit at 0).
- Play: elbows bent vs limp; hip pitch/knee actually swinging.
- `feet_air_time` (normalized, weight 2 → peak contribution 2.0); `foot_gait` 5.0.
- `base_height` near 0 inside ±2 cm; sustained negative = crouch.
- Mean episode length vs 20 s (dilution).

If arms still limp: raise arm L1 toward -0.5…-1.0, do not bring unified `pose` back without splitting arms vs legs. If 碎步: gait/air-time vs tracking 10, not L1.

## Suggested skills

- `graphify` (mandatory in this repo before Read/Grep/Glob/Bash exploration): `.cursor/rules/graphify.mdc`, `/home/ordis/.claude/skills/graphify/SKILL.md`
- `tdd` if changing rewards/config: `/home/ordis/projects/unitree_rl_mjlab/.agents/skills/tdd/SKILL.md`
- `diagnosing-bugs` if play/logs still look wrong after a **new** train on this HEAD: `/home/ordis/projects/unitree_rl_mjlab/.agents/skills/diagnosing-bugs/SKILL.md`
- `code-review` if asked to review since `8106a6c` or a later commit: `/home/ordis/projects/unitree_rl_mjlab/.agents/skills/code-review/SKILL.md`
- `grilling` only if the user wants to stress-test another reward swap: `/home/ordis/projects/unitree_rl_mjlab/.agents/skills/grilling/SKILL.md`

Do **not** start from `research` or rewrite plant/action unless asked. Do **not** import effort rewards from `src.tasks.velocity`.
