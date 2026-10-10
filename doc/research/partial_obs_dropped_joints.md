# Partial-obs dropped joints survey

Survey date: 2026-09-14.

Primary sources only: arXiv HTML/PDF, official Unitree robot-description repos, and first-party GitHub observation/action configs. Joint lists below are copied from those files or quoted from papers. Secondary write-ups (DeepWiki, Hugging Face model cards without matching paper/repo) are not used as evidence.

## 中文摘要（给本仓库任务用）

文献几乎不做「观测里没有、动作里还有」。走路里被当成多余的关节，作者会 **冻死或从模型删掉**（观测和动作一起没了），或者 **两边都留着** 再用 `joint_deviation_*` 罚回默认角。Teacher–student 的 partial obs 藏的是线速度、高度、接触，不是关节编码器。

两套都叫 23-DoF，关节集合不同，不要混：

| 名字 | 组成 | 腕 | 腰 roll/pitch |
|------|------|----|----------------|
| **官方 Unitree 23**（本仓库 `g1_23dof.xml`） | 12 腿 + 1 腰 yaw + 10 臂 | **只留 roll**；pitch/yaw 硬件就没有 | **没有** |
| **ASAP / GMT / UniTracker / ExBody2 的 23** | 12 腿 + 3 腰 + 8 臂 | **六轴全冻** | **保留** |

本仓库若坚持「29 维力矩仍输出、只从 `joint_pos`/`joint_vel` 切片」，这是新实验，不是复现。关节先验仍应对齐其中一套：官方 23 切 6 轴（腰 RP + 腕 pitch/yaw）；ASAP 23 切 6 个腕（含 roll），腰三轴都留。踝不要切。`last_action` 若仍含被藏维，策略能开环补上，实验就不干净。

下文是按来源拆开的 A vs O 表。

## Question

For Unitree G1/H1-style humanoid (and analogous quadruped) locomotion RL, which joints do existing systems treat as **unnecessary** in the **policy observation**, while the actor still outputs position or torque commands for those joints?

This repo already has G1 29-DoF and G1 23-DoF models. The practical question is which joints a locomotion-oriented *partial observation* task should drop.

## Distinction: unobserved vs frozen vs unused

These three patterns are routinely conflated. They are not the same MDP.

| Pattern | In observation (`q`, `dq`)? | In action (PD target or torque)? | Typical implementation |
|---|---|---|---|
| **Unobserved-but-actuated** | No | Yes | Observation term uses a joint subset; action term still covers those motors. This is the asked-for pattern. |
| **Frozen / locked** | No (joint is absent or fixed) | No | Fixed URDF joint, reduced XML, or `has_dof_subset` / lock-waist model. Joint is not a free DoF. |
| **Unused / not in the robot** | No | No | Hardware never had the motor (G1 basic 23-DoF has no wrist pitch/yaw; stock H1 has no wrists). |
| **Observed-and-actuated, regularized** | Yes | Yes | Full `joint_names=[".*"]` plus a `joint_deviation_*` reward that pins arms/wrists/waist near default. |
| **Observed-but-not-actuated** | Yes | No | Inverse of the asked-for pattern: extra DoF in `dof_pos` / last-action history, policy outputs only a subset (usually legs). |

**Headline finding:** across the G1/H1 locomotion and motion-tracking line, the asked-for pattern (omit joints from **obs** but still **actuate** them) is almost never used. When authors call a joint “unnecessary for locomotion,” they **freeze** it (drop from both obs and action) or they **keep it in both** and penalize deviation from a default pose. Teacher–student “partial observation” papers drop *privileged* channels (base linear velocity, height map, contacts, randomized mass), not a subset of joint encoders.

Hiding a joint from `q`/`dq` while still commanding it is a POMDP the policy can often close using `last_action`. That is likely why papers freeze instead of unobserve.

## Consensus (which joints people drop)

Counts below are **independent first-party sources** that take a clear stance. “Drop” here means *not in the policy’s joint-state / action vectors*, usually by freezing.

### Strong consensus: wrists are the first thing to drop

| Joints (G1 names) | What papers/code do | Why they say so | Source count |
|---|---|---|---|
| `left/right_wrist_pitch_joint`, `left/right_wrist_yaw_joint` | Frozen, or physically absent on 23-DoF G1 | Hardware: 23-DoF G1 has `wrist motor = null` for pitch/yaw. Papers: “fixed wrists”, “fix the wrist joints … reducing the action space to 23”, “excluding the 3 DoFs in each wrist”. | **≥10** (Unitree `g1_description`, ASAP, GMT, UniTracker, official 23-DoF URDF, ALMI on H1-2, HumanPlus HST, OmniH2O, Adam Lite wrist-yaw, SSR 21-DoF block) |
| `left/right_wrist_roll_joint` | Split. Official 23-DoF **keeps** wrist roll (5-DoF arms). ASAP / UniTracker / GMT-style 23-DoF **drop all six wrists** and keep 3-DoF waist instead. | Official 23-DoF arm is shoulder×3 + elbow + wrist_roll. Research 23-DoF is often 12 legs + 3 waist + 8 arms (no wrists). | Split: official 23-DoF keeps roll; ASAP-family freezes all six wrists |

### Strong consensus: do **not** drop ankles from locomotion

Ankle pitch/roll stay in both obs and action whenever the robot has them. ASAP’s real-world residual even **focuses on a 4-DoF ankle delta** because the G1 ankle 4-bar linkage is the large sim-to-real gap, not something to unobserve ([ASAP §real-world data](https://arxiv.org/html/2502.01143)).

### Waist roll / pitch: split, not a wrist-like consensus

| Setup | Waist yaw | Waist roll / pitch |
|---|---|---|
| Official G1 23-DoF (`g1_23dof*`) | Actuated | **Absent** (torso welded to yaw) |
| Official `g1_29dof_lock_waist*` | Actuated | **Fixed joints** in URDF |
| ASAP `g1_29dof_anneal_23dof` | Actuated | **Actuated** (this “23” drops wrists, not waist) |
| unitree_rl_lab G1-29 velocity | Actuated | Actuated, **heavily regularized** (`joint_deviation_waists` weight `-1`) |
| SE-Policy G1 “27-DoF” | Actuated | Frozen (12 legs + 1 waist + 14 arm/wrist = 27) |

### Arms (shoulder / elbow): usually kept, or removed from the **model**

Locomotion stacks that keep a full-body G1 almost always **observe and actuate** shoulders and elbows, then add `joint_deviation_arms`. The 12-DoF Unitree RL Gym G1 instead uses `g1_12dof.urdf` (legs only): arms are unused, not unobserved-but-actuated ([`g1_config.py`](https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py)).

### Fingers / neck

G1 locomotion policies never include finger DoF. Isaac Lab’s older `G1_MINIMAL_CFG` is a different asset (named finger joints `*_five_joint`, …) and regularizes them rather than dropping them from obs ([Isaac Lab `rough_env_cfg.py`](https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py)). Stock G1/H1 have no neck joints.

### G1 29 vs 23 vs 21, and H1 vs H1-2

| Label | Composition | Wrists | Waist | Notes |
|---|---|---|---|---|
| Official **G1 29-DoF** | 12 leg + 3 waist + 14 arm | roll+pitch+yaw ×2 | yaw+roll+pitch | [`g1_description` README](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md): `dof#leg 6*2`, `dof#waist 3`, `dof#arm 7*2`, wrist motor 4010/5010 |
| Official **G1 23-DoF** | 12 leg + 1 waist + 10 arm | **roll only** ×2 | **yaw only** | Same README: `wrist motor = null`, `dof#waist 1`, `dof#arm 5*2` |
| Research **“anneal 23”** (ASAP / UniTracker) | 12 leg + 3 waist + 8 arm | **all six frozen** | all three kept | Not the same 23 as Unitree’s 23 |
| Research **21-DoF** | 12 leg + 1 waist yaw + 8 arm | all wrists out | roll/pitch out | **No official G1 21-DoF URDF.** Pattern appears on H1-2 (ALMI) and AgiBot X2 (SSR). 29 − 6 wrists − 2 waist RP = 21 |
| **H1** | 10 leg (5×2, 1-DoF ankle) + 1 torso + 8 arm | **none** | 1 (`torso_joint`) | [Unitree H1 About](https://support.unitree.com/home/en/H1_developer/About_H1): 19 DoF |
| **H1-2** | 12 leg (6×2) + 1 waist + 14 arm | 3×2 present | 1 | 27 motors. ALMI **controls 21**, excluding both wrists |

## Per-source table/notes with citations

Abbreviations: **A** = in action, **O** = in joint-position/velocity observation. Frozen means not A and not O.

### Hardware / official models

#### Unitree G1 description (URDF/MJCF), ongoing

- **URL:** https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md
- **Robot:** G1. Table of `mode_machine` IDs.
- **23-DoF:** `dof#leg=6*2`, `dof#waist=1`, `dof#arm=5*2`, `wrist motor=null`. Wrist pitch/yaw **do not exist**. Waist roll/pitch **do not exist**. Wrist roll **does exist**.
- **29-DoF:** `dof#leg=6*2`, `dof#waist=3`, `dof#arm=7*2`, wrist motor 4010 or 5010. All six wrists exist.
- **`g1_29dof_*lock_waist*`:** same 29 names, `lock waist = yes` / `mode_machine` 6, 12, 14, 16. Waist roll/pitch are fixed joints, not free DoF.
- **Unobserved-but-actuated:** none. This is a kinematic description, not a policy.
- **Reason:** product variants, not a locomotion ablation.

This repo’s `src/assets/robots/unitree_g1/xmls/g1.xml` is the 29-DoF chain (includes `waist_roll_joint`, `waist_pitch_joint`, `*_wrist_pitch_joint`, `*_wrist_yaw_joint`). `g1_23dof.xml` matches official 23-DoF: `waist_yaw_joint` plus `*_wrist_roll_joint`, no pitch/yaw wrists, no waist roll/pitch.

#### Unitree H1 About page

- **URL:** https://support.unitree.com/home/en/H1_developer/About_H1
- **Robot:** H1, **19 DoF**. Leg 5 (hip×3 + knee + **1-DoF ankle**), arm 4 (shoulder×3 + elbow), waist 1 (`torso_joint`).
- **Wrists:** not on the stock robot. HumanPlus later bolts on a 1-DoF Dynamixel wrist per arm (see HumanPlus).
- **A/O:** N/A (spec sheet).

#### G1 joint index (IDL / LowCmd)

- **URL (search hit; site WAF-blocked on fetch):** https://support.unitree.com/home/en/G1_developer/joint_motor_sequence
- Cross-check from first-party code, Hugging Face LeRobot `G1_29_JointIndex`: legs 0–11, `WAIST_YAW/ROLL/PITCH` 12–14, left arm 15–21, right arm 22–28, with pitch/yaw wrists at 20–21 and 27–28 ([`g1_utils.py`](https://github.com/huggingface/lerobot/blob/192a0b92/src/lerobot/robots/unitree_g1/g1_utils.py)).
- 23-DoF LowCmd still *addresses* 29 slots; wrist pitch/yaw (and often waist roll/pitch) are empty/invalid. That is unused slots, not unobserved-but-actuated policy joints.

### Unitree / Isaac Lab / mjlab locomotion stacks

#### unitree_rl_gym G1 (official Unitree RL)

- **URL:** https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py
- **Year:** repo 2024–.
- **Robot:** G1 via `g1_12dof.urdf`.
- **Action:** `num_actions = 12`. Default angles listed: both hips (yaw/roll/pitch), knees, ankle pitch/roll, `torso_joint`. No arms, no wrists.
- **Observation:** `num_observations = 47` (standard legged_gym packing of base ang-vel, gravity, commands, 12 `q`, 12 `dq`, 12 last actions — not a joint subset of a 29-DoF model).
- **Frozen vs unobserved:** **unused.** Arms/wrists/waist-RP are not in the URDF.
- **Reason:** locomotion baseline; 23/29 training is left to users ([issue #54](https://github.com/unitreerobotics/unitree_rl_gym/issues/54) reports naive 23/29 conversions failing).

#### unitree_rl_lab G1-29 velocity (official Unitree Isaac Lab)

- **URLs:**
  - https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py
  - https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py (`UNITREE_G1_29DOF_CFG`)
- **Year:** 2025.
- **Robot:** G1 29-DoF. `joint_sdk_names` is the full 29-name list (legs, 3 waist, 7+7 arms including all wrists).
- **Action:** `JointPositionActionCfg(..., joint_names=[".*"], scale=0.25)` → **all 29 actuated**.
- **Observation (policy):** `base_ang_vel`, `projected_gravity`, `velocity_commands`, `joint_pos_rel`, `joint_vel_rel`, `last_action`, history 5. **No joint-name filter** → all 29 in obs.
- **Privileged (critic):** adds `base_lin_vel` (not a joint subset).
- **Dropped from obs but actuated:** **none.**
- **“Unnecessary” handling:** rewards, not obs masking:
  - `joint_deviation_arms` on `.*_shoulder_.*`, `.*_elbow_joint`, `.*_wrist_.*` (weight `-0.1`)
  - `joint_deviation_waists` on `waist.*` (weight `-1`)
- **Reason:** locomotion with a full EDU body; pin unused-for-walking joints at default rather than hide them.

This repo’s effort G1 task copies that pattern (`src/tasks/effort/config/g1/env_cfgs.py` regularizes `.*_wrist_.*` and `waist.*`; `g1_effort_action_cfg()` commands **all 29**).

#### Isaac Lab bundled G1 velocity (`G1_MINIMAL_CFG`)

- **URLs:**
  - https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_assets/isaaclab_assets/robots/unitree.py
  - https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1/rough_env_cfg.py
- **Robot:** Isaac Lab’s older G1 USD (`torso_joint`, `*_elbow_pitch_joint`, named finger joints). **Not** the commercial 23/29 G1 chain.
- **Action/obs:** default locomotion `joint_pos_rel` / `joint_pos` action with `[".*"]`.
- **Regularized as non-essential:** `joint_deviation_arms` (shoulder, elbow_pitch/roll), `joint_deviation_fingers` (`*_five/three/six/four/zero/one/two_joint`), `joint_deviation_torso` (`torso_joint`). Comment in env cfg: “Penalize deviation from default of the joints that are not essential for locomotion.”
- **Dropped from obs but actuated:** **none.**

#### mjlab / this repo tracking (BeyondMimic clone)

- **URLs:**
  - Upstream: https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py
  - This repo: `src/tasks/tracking/tracking_env_cfg.py`, `src/tasks/tracking/config/g1/env_cfgs.py`
- **Robot:** G1 29-DoF. Actuators include `*_wrist_roll/pitch/yaw_joint` ([BeyondMimic `g1.py`](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/robots/g1.py)).
- **Action:** `joint_names=[".*"]` → all actuated joints, **including wrists**.
- **Observation:** `joint_pos_rel` / `joint_vel_rel` with no subset → all joints in obs.
- **Partial obs that *is* implemented:** if `has_state_estimation=False`, actor drops `motion_anchor_pos_b` and `base_lin_vel` — privileged *root* quantities, **not joints**.
- **Dropped from obs but actuated:** **none.**

### Whole-body tracking / teleop papers (G1)

#### ASAP (LeCAR, 2025)

- **Paper:** https://arxiv.org/abs/2502.01143 — https://arxiv.org/html/2502.01143
- **Code:** https://github.com/LeCAR-Lab/ASAP/blob/main/humanoidverse/config/robot/g1/g1_29dof_anneal_23dof.yaml
- **Robot:** Unitree G1. Config name `g1_29dof_anneal_23dof`. `dof_obs_size: 23`, `actions_dim: 23`.
- **Explicit `dof_names` (23, copied from YAML):**
  - Legs (12): `left/right_hip_pitch/roll/yaw`, `left/right_knee`, `left/right_ankle_pitch/roll`
  - Waist (3): `waist_yaw_joint`, `waist_roll_joint`, `waist_pitch_joint`
  - Arms (8): `left/right_shoulder_pitch/roll/yaw`, `left/right_elbow`
- **Not in `dof_names`:** all six wrist joints.
- **A vs O:** same 23 in both (`dof_obs_size` = `actions_dim`). Wrists are **frozen**, not unobserved-but-actuated. `has_dof_subset: True`.
- **Paper quote:** “we deploy ASAP on Unitree G1 robot **with fixed wrists**.” Real-world residual is a **4-DoF ankle** delta (linkage sim-to-real gap + data hunger), not a wrist residual.
- **Reason:** hardware/sim subset for agile tracking; ankles are the opposite of “unnecessary.”

#### GMT (2025)

- **Paper:** https://arxiv.org/abs/2506.14770 — https://arxiv.org/html/2506.14770v1
- **Robot:** “Unitree G1 … **23 DoFs**”, height 1.32 m.
- **Teacher proprioception `o_t`:** root ang-vel (3), roll/pitch (2), **joint pos 23, joint vel 23, last action 23**.
- **Student:** same `o_t` plus history; **does not** receive privileged `e_t` (root lin-vel, height, key bodies, feet contact, mass rand, motor strength).
- **A vs O:** 23/23. Matches ASAP-family 23 (wrists out). Paper does not print the 23 names; dimensions match ASAP’s published list, not official 23-DoF (which also has 23, but different waist/wrist split). Treat the **name list** as ASAP’s when using GMT numbers, or verify GMT’s robot XML before copying names.
- **Dropped from student obs:** privileged extras, **not joints**.
- **Reason (stated):** teacher–student for sim-to-real / partial observability of base velocity and contacts.

#### UniTracker (2025)

- **Paper:** https://arxiv.org/abs/2507.07356 — https://arxiv.org/html/2507.07356v1
- **Quote:** “To better focus on motion tracking at the whole-body level, we **fix the wrist joints** of our **29-DoF** Unitree G1 robot, **reducing the action space to 23**. The action \(a_t\in\mathbb{R}^{23}\) specifies target joint positions.”
- **A vs O:** 23 actuated; wrists **frozen**. Same 12+3+8 split as ASAP (wrists out, 3-DoF waist in).
- **Reason (stated):** focus whole-body tracking; wrists not needed for the tracked skills.

#### TWIST (CoRL 2025)

- **Paper:** https://arxiv.org/abs/2505.02833 — https://arxiv.org/html/2505.02833
- **Code:** https://github.com/YanjieZe/TWIST (g1 env path 404 on fetch; paper is the source).
- **Robot:** G1 **29 DoF**. Tasks include whole-body manipulation (picking, crouching to a box) that **need wrists**.
- **A/O:** paper describes a single 29-DoF controller; GMR retargeter lists `unitree_g1` as “Leg (2*6) + Waist (3) + Arm (2*7) = 29” ([GMR README](https://github.com/YanjieZe/GMR)). No statement that wrists are omitted from obs while remaining in action.
- **Contrast with ASAP:** TWIST keeps wrists because teleop/manipulation needs them.

#### BeyondMimic (2025)

- **Paper:** https://arxiv.org/abs/2508.08241
- **Code:** `joint_pos = JointPositionActionCfg(..., joint_names=[".*"])` and `joint_pos_rel` with no subset ([`tracking_env_cfg.py`](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/tasks/tracking/tracking_env_cfg.py)). Actuator list includes all three wrist axes ([`g1.py`](https://github.com/HybridRobotics/whole_body_tracking/blob/main/source/whole_body_tracking/whole_body_tracking/robots/g1.py)).
- **Robot:** G1 29-DoF, wrists **in both** A and O.
- **Undesired contacts** exclude `left/right_wrist_yaw_link` from the penalty (those links *are* allowed to touch), which is a reward mask, not an observation mask.
- **Dropped from obs but actuated:** **none.**

#### HoloMotion G1-29 (Horizon Robotics)

- **URL:** https://github.com/HorizonRobotics/HoloMotion/blob/5b51cc24/holomotion/config/robot/unitree/G1/29dof/29dof_training_isaaclab.yaml
- **`dof_obs_size: 29`, `actions_dim: 29`**, full 29-name `dof_names` including all wrists and 3 waist.
- **Dropped from obs but actuated:** **none.**

#### SE-Policy (AAAI 2026 / arXiv 2025)

- **Paper:** https://arxiv.org/abs/2508.01247 — https://arxiv.org/html/2508.01247v1
- **Robot:** Unitree G1 described as **27-DoF**. Action \(|\mathcal{A}|=27\).
- **Appendix joint breakdown:** each arm = 3 shoulder + elbow + **3 wrist**; each leg = 3 hip + knee + 2 ankle; **one waist joint**. 14+12+1 = 27.
- **Interpretation:** **lock waist roll/pitch, keep all six wrists.** Matches official `g1_29dof_lock_waist` kinematics (29 names, 2 waist axes fixed).
- **A vs O:** 27/27. Previous action is 27-D.
- **Dropped from obs but actuated:** **none.** Waist RP are frozen.

### H1 / H1-2 / HumanPlus / HOVER / OmniH2O / ExBody

#### OmniH2O (2024)

- **Paper:** https://arxiv.org/abs/2406.08858 — https://arxiv.org/html/2406.08858
- **Robot:** Unitree H1 **19-DoF body**, plus Damiao wrist motors and Inspire hands on the real robot.
- **Policy A/O tables:** DoF position 19, DoF velocity 19, actions **19** (stock H1 body). History 25 → student obs 1665 in one table.
- **Wrists/fingers:** “control for the robot’s **fingers and wrists does not require inference**, it is **directly mapped from the Vision Pro** data to the corresponding joints.” Wrists are **actuated by teleop, not by the RL policy**. That is *externally commanded*, not unobserved-but-actuated by the actor.
- **Privileged teacher:** full motion goals / extra DoF features; student drops those **commands**, not encoder joints.

#### HOVER (2024, NVIDIA)

- **Paper:** https://arxiv.org/abs/2410.21229
- **Code:** https://github.com/NVlabs/HOVER/blob/main/neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/neural_wbc_env_cfg_h1.py
- **Robot:** H1. `action_space = 19`. `joint_names` (19): hips/knees/ankles, `torso`, shoulders, elbows. **No wrists.**
- **Masks (`DISTILL_MASK_MODES_ALL`)** apply to **task commands** (which reference joints/keypoints to track), not to proprioception:
  - `exbody`: upper `torso/shoulder/elbow` joints; lower `root.*`
  - `humanplus`: upper same; lower `hip/knee/ankle` + root
  - `omnih2o`: `hand` + `head` links only
- **Student still outputs 19 joint targets.** Masked dimensions are missing *references*, not missing encoder readings.
- **Dropped from obs but actuated:** **none** for joints. Command masking ≠ joint unobservation.

#### HumanPlus (CoRL 2024)

- **Paper:** https://arxiv.org/abs/2406.10454 — https://arxiv.org/html/2406.10454v1
- **Robot:** customized H1, **33 DoF** = 19 body + **two 1-DoF wrists** + two 6-DoF Inspire hands.
- **HST (low-level RL) output:** “**19-dimensional** joint position setpoints for humanoid **body** joints.” “The **target hand joint angles are directly passed to the PD controller**.” Wrist angle is **retargeted** from forearm–hand relative rotation, not produced by HST.
- **HST proprioception:** root (roll, pitch, base ang-vel), joint positions, joint velocities, last action — for the **19 body joints**.
- **Wrists:** actuated (Dynamixel via serial), **not** in the 19-D actor. Frozen-from-policy / externally set, not unobserved-but-actuated by HST.

#### ExBody (RSS 2024)

- **Paper:** https://arxiv.org/abs/2402.16796 — https://ar5iv.labs.arxiv.org/html/2402.16796
- **Robot:** H1, 19 DoF. `a_t ∈ R^{19}`.
- **Proprioception:** `ω, roll, pitch, Δyaw, q, dq, a_{t-1}` — **all 19 joints**.
- **Stated unobserved quantities:** root linear velocity, absolute height, absolute yaw (privileged). **Not joints.**
- **Upper vs lower:** imitation reward is relaxed on **legs** (velocity tracking instead of pose imitation). Legs are still fully observed and actuated.

#### ExBody2 (2024)

- **Paper:** https://arxiv.org/abs/2412.13196
- **Robot:** G1. **Action 23-D.** Proprioception table: DoF pos 23, vel 23, last action 23, plus IMU (table “75*10” with history 10).
- **23-DoF here:** authors say they “extended Exbody to tracking **all three dofs of the waist**” on G1. That matches ASAP-family 23 (3 waist, no wrists), not official 23 (1 waist, wrist_roll).
- **Dropped from obs but actuated:** **none** in the published table.

#### ALMI (2025) — the clean H1-2 “21-DoF” source

- **Paper:** https://arxiv.org/abs/2504.14305 — https://arxiv.org/html/2504.14305v3
- **Robot:** H1-2, **27 DoF** total. “The policy controls **21 DoFs, excluding the 3 DoFs in each wrist**.”
- **Proprioception:** `q_t ∈ R^{21}`, `dq_t ∈ R^{21}`. Lower policy action 12 (legs); upper policy action 9 (waist yaw + 4+4 arm without wrists); whole-body 21.
- **A vs O:** 21/21. Wrists **frozen / not commanded by these policies** (VR IK may command them at deploy for loco-manipulation; that is a different controller).
- **Privileged:** critic adds 3-D base linear velocity.
- **Reason (stated):** exclude wrist DoF of the hands from the body policy.

### Other humanoids (same design pattern)

#### SSR (2026) — 21-DoF action on a 29-joint robot

- **Paper:** https://arxiv.org/abs/2605.30770 — https://arxiv.org/html/2605.30770v1
- **Robot:** AgiBot X2, “**29 joints**”; policy `a_t ∈ R^{21}`. Proprioception `o_t^p ∈ R^{72}` = ang-vel 3 + gravity 3 + cmd 3 + **θ 21 + dθ 21 + a_{t-1} 21**.
- **Appendix joint block:** `x = [leg_R(6), leg_L(6), waist(1), arm_R(4), arm_L(4)] ∈ R^{21}` with arm = shoulder pitch/roll/yaw + elbow. **No wrists. One waist axis.**
- **A vs O:** 21/21. The 8 unused motors (on a 29-joint body: 6 wrists + waist roll/pitch) are **not** in the actor. Frozen/unused, not unobserved-but-actuated.
- **Privileged critic:** base velocity, feet, contacts — not hidden joints.

#### Adam Lite contrastive locomotion (2025)

- **Paper:** https://arxiv.org/abs/2509.12858 — https://arxiv.org/html/2509.12858v1
- **Quote:** “25 degrees of freedom … we **only control 23** of these DOFs because the **wrist yaw is not actively commanded**.”
- **Pattern:** freeze wrist yaw (not A, not O), not unobserve-while-actuate.

#### Masked Humanoid Controller (2024)

- **Paper:** https://arxiv.org/abs/2408.07295
- **Robot:** Digit V3. **Masks motion directives** (which joints the *command* specifies). Proprioception remains full `θ, dθ`. Output is PD setpoints for **all** motors.
- **Not** joint-encoder dropout.

#### HumanoidUMI (2026)

- **Paper:** https://arxiv.org/html/2606.27239v1
- **Obs:** 15-D lower-body proprioception = **12 leg + 3 waist**. “Upper-body joints are omitted because the policy expresses manipulation intent through task-space TCP keypoints.”
- **Those omitted joints are not outputs of this policy** (IK/SKR downstream). Not unobserved-but-actuated by the same actor.

### Inverse pattern (observe extra, actuate subset)

#### OpenWBT G1 squat config

- **URL:** https://github.com/GalaxyGeneralRobotics/OpenWBT/blob/main/deploy/configs/g1_squat.yaml
- **`num_dof: 29`, `num_actions: 12`.** `dof_idx` = 0..28; `action_idx` = 0..11 (legs). Comment lists wrists in the 29-DoF name legend.
- **This is observed-but-not-actuated** (or observed full-state, legs-only actor). Opposite of the asked-for pattern. Useful as a warning: several “partial” G1 locomotion stacks shrink **action**, not observation.

### Quadrupeds

Unitree Go2 / A1 / Go1 locomotion (unitree_rl_gym, Isaac Lab, Walk-These-Ways-style) uses **all 12** leg joints in both obs and action. There is no wrist/waist analogue. “Partial observation” means missing **base linear velocity / height scan / friction**, fed to a privileged critic or teacher (RMA, Rapid Locomotion, etc.). No first-party quadruped source in this search dropped a subset of the 12 actuated joints from actor proprioception while still commanding them.

Go2W adds wheel joints; those are extra actuated DoF, not dropped ones ([unitree_rl_lab `UNITREE_GO2W_CFG`](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/assets/robots/unitree.py)).

### Privileged / teacher–student (what “partial obs” usually means here)

| Source | Actor (deployable) joint proprioception | What is actually dropped |
|---|---|---|
| GMT student | full 23 `q,dq,a` | `e_t`: lin-vel, height, contacts, mass, motor strength |
| unitree_rl_lab G1-29 | full 29 | critic-only `base_lin_vel` |
| BeyondMimic / this repo tracking without state estimation | full joints | `base_lin_vel`, `motion_anchor_pos_b` |
| HOVER student | full 19 H1 joints | **command** mask (which references to track) |
| ALMI critic | full 21 | + base lin-vel |
| ExBody | full 19 | lin-vel, height, absolute yaw |

None of these drop wrist/ankle/waist **encoders** from the actor while keeping those motors in the action vector.

## Mapping to Unitree G1/H1 joint names

G1 names as used in this repo (`g1.xml` / `g1_23dof.xml`) and Unitree `joint_sdk_names`.

### G1 29-DoF (this repo `g1.xml`)

| Group | Joint names | Official 23-DoF | ASAP-family 23 | Loco 21-pattern | unitree_rl_lab 29 loco |
|---|---|---|---|---|---|
| Legs (12) | `left/right_hip_pitch_joint`, `_hip_roll_joint`, `_hip_yaw_joint`, `_knee_joint`, `_ankle_pitch_joint`, `_ankle_roll_joint` | A+O | A+O | A+O | A+O |
| Waist yaw | `waist_yaw_joint` | A+O | A+O | A+O | A+O (deviation penalty) |
| Waist roll/pitch | `waist_roll_joint`, `waist_pitch_joint` | **absent** | A+O | **frozen** | A+O (strong deviation penalty) |
| Shoulders + elbow (8) | `left/right_shoulder_pitch/roll/yaw_joint`, `left/right_elbow_joint` | A+O | A+O | A+O | A+O (arm deviation) |
| Wrist roll (2) | `left/right_wrist_roll_joint` | A+O | **frozen** | **frozen** | A+O (arm/wrist deviation) |
| Wrist pitch/yaw (4) | `left/right_wrist_pitch_joint`, `left/right_wrist_yaw_joint` | **absent** (`wrist motor=null`) | **frozen** | **frozen** | A+O (wrist deviation) |

### H1 / H1-2

| Robot | Joints in typical body policy | Wrists |
|---|---|---|
| H1 19 | `*_hip_yaw/roll/pitch`, `*_knee`, `*_ankle` (1-DoF), `torso`, `*_shoulder_pitch/roll/yaw`, `*_elbow` ([HOVER `joint_names`](https://github.com/NVlabs/HOVER/blob/main/neural_wbc/isaac_lab_wrapper/neural_wbc/isaac_lab_wrapper/neural_wbc_env_cfg_h1.py)) | Not on stock robot; HumanPlus/OmniH2O add them **outside** the 19-D policy |
| H1-2 27 | ALMI uses 21: 12 leg (now 2-DoF ankles) + `torso`/waist yaw + 8 arm (no 6 wrists) | Present in hardware; **excluded** from ALMI A and O |

## Recommendation for a locomotion-oriented partial task

Match what the field actually does, and keep the three patterns separate in the task definition.

1. **Do not implement “wrists in action, wrists missing from `joint_pos`/`joint_vel`” as the default “literature partial obs.”** No G1/H1 locomotion paper found here does that. They freeze wrists or they keep them in both streams.

2. **If the goal is “joints unnecessary for locomotion” in the Unitree sense:**
   - **Minimum / official-23-aligned:** drop `*_wrist_pitch_joint` and `*_wrist_yaw_joint` **and** `waist_roll_joint` / `waist_pitch_joint` from **both** action and observation (use the existing `g1_23dof` asset). Keep `*_wrist_roll_joint` and `waist_yaw_joint`.
   - **ASAP/GMT/UniTracker-aligned 23:** freeze **all six wrists**; keep **three** waist joints. This is **not** Unitree’s 23-DoF SKU. Document the mismatch in the task name (`g1_anneal23` vs `g1_23dof`).
   - **21-DoF locomotion body (ALMI/SSR pattern):** 12 legs + `waist_yaw_joint` + 8 shoulder/elbow. Freeze six wrists **and** waist roll/pitch. There is **no** official G1 21-DoF URDF; this is a policy subset of 29-DoF.

3. **If the goal is a POMDP over joints (true unobserved-but-actuated):** that is a **research experiment**, not a reproduction. The only joints literature treats as locomotion-irrelevant are **wrists** (then waist roll/pitch). If you still command them, also decide whether `last_action` includes those dims (it usually should, or the policy has no local feedback). Ankles must stay in obs.

4. **If the goal is Unitree-style velocity tracking on 29-DoF:** keep all joints in A and O; copy `joint_deviation_arms` / `joint_deviation_waists` from [unitree_rl_lab G1-29](https://github.com/unitreerobotics/unitree_rl_lab/blob/main/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/g1/29dof/velocity_env_cfg.py). That is the official “arms/wrists/waist are unnecessary for walking” mechanism.

5. **Privileged partial obs (what teacher–student papers mean):** drop `base_lin_vel`, height scan, contacts from the **actor**; keep **all commanded joints** in actor proprioception. This repo’s tracking `has_state_estimation=False` path already does the lin-vel / anchor-pos part.

6. **Never drop ankles** from a locomotion actor. ASAP’s real-robot residual is ankle-only for the opposite reason (they matter more, and the 4-bar is hard to sim).

## Open questions / contradictions

1. **Two different “G1 23-DoF”s.** Official Unitree 23 = 12 + 1 waist yaw + 10 arm (wrist_roll kept). ASAP `g1_29dof_anneal_23dof` = 12 + 3 waist + 8 arm (all wrists fixed). Both are 23-dimensional. Mixing them silently breaks joint order and PD gains.

2. **No official G1 21-DoF.** The 21-DoF *policy* pattern (12+1+8) is documented on H1-2 ([ALMI](https://arxiv.org/html/2504.14305v3)) and AgiBot X2 ([SSR](https://arxiv.org/html/2605.30770v1)), not in `g1_description`.

3. **Waist roll/pitch vs wrists, which to freeze first?** Official 23-DoF and `lock_waist` freeze waist RP and **keep** wrist roll (and, on 29 lock-waist, all wrists). ASAP-family freezes **wrists** and **keeps** waist RP. SE-Policy’s 27-DoF G1 freezes waist RP and **keeps** all wrists. There is no unique answer; pick the SKU you will deploy.

4. **Unobserved-but-actuated is essentially undocumented** for these robots. If a prior chat or internal task assumed “drop wrists from obs, still output wrist PD,” that is **not** what ASAP/GMT/unitree_rl_lab/HOVER do.

5. **HOVER/HumanPlus/OmniH2O “partial” is command-side.** They mask *references* (head/hands vs full joints), not proprioceptive joint subsets. Copying `DISTILL_MASK_MODES_ALL` does not create a partial-obs locomotion task.

6. **Isaac Lab `G1_MINIMAL_CFG` ≠ commercial G1.** Finger joint names and `torso_joint` / `elbow_pitch` will not map to this repo’s MJCF.

7. **TWIST vs ASAP on the same robot:** 29-DoF with wrists (teleop/manipulation) vs 23-DoF fixed wrists (agile tracking). Task family drives the freeze, not a universal “wrists are noisy sensors” argument. No source in this survey claimed wrist *encoders* are missing or too noisy to use.

8. **Ankle 1-DoF vs 2-DoF:** H1 has `*_ankle` only; G1 and H1-2 have pitch+roll. Do not drop G1 `ankle_roll` because H1 lacked it.

## Source index

| # | Source | Year | URL |
|---|---|---|---|
| 1 | Unitree `g1_description` README | 2024– | https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/README.md |
| 2 | Unitree H1 About | — | https://support.unitree.com/home/en/H1_developer/About_H1 |
| 3 | unitree_rl_gym `g1_config.py` | 2024– | https://github.com/unitreerobotics/unitree_rl_gym/blob/main/legged_gym/envs/g1/g1_config.py |
| 4 | unitree_rl_lab G1-29 velocity | 2025 | https://github.com/unitreerobotics/unitree_rl_lab |
| 5 | Isaac Lab `unitree.py` + G1 rough env | 2024– | https://github.com/isaac-sim/IsaacLab |
| 6 | ASAP | 2025 | https://arxiv.org/abs/2502.01143 , https://github.com/LeCAR-Lab/ASAP |
| 7 | GMT | 2025 | https://arxiv.org/abs/2506.14770 |
| 8 | UniTracker | 2025 | https://arxiv.org/abs/2507.07356 |
| 9 | TWIST | 2025 | https://arxiv.org/abs/2505.02833 |
| 10 | BeyondMimic | 2025 | https://arxiv.org/abs/2508.08241 , https://github.com/HybridRobotics/whole_body_tracking |
| 11 | OmniH2O | 2024 | https://arxiv.org/abs/2406.08858 |
| 12 | HOVER | 2024 | https://arxiv.org/abs/2410.21229 , https://github.com/NVlabs/HOVER |
| 13 | HumanPlus | 2024 | https://arxiv.org/abs/2406.10454 |
| 14 | ExBody | 2024 | https://arxiv.org/abs/2402.16796 |
| 15 | ExBody2 | 2024 | https://arxiv.org/abs/2412.13196 |
| 16 | ALMI | 2025 | https://arxiv.org/abs/2504.14305 |
| 17 | SSR | 2026 | https://arxiv.org/abs/2605.30770 |
| 18 | SE-Policy | 2025 | https://arxiv.org/abs/2508.01247 |
| 19 | Adam Lite contrastive loco | 2025 | https://arxiv.org/abs/2509.12858 |
| 20 | MHC | 2024 | https://arxiv.org/abs/2408.07295 |
| 21 | HoloMotion G1-29 yaml | 2025– | https://github.com/HorizonRobotics/HoloMotion |
| 22 | GMR robot DoF table | 2025 | https://github.com/YanjieZe/GMR |
| 23 | This repo G1 MJCF / effort / tracking configs | — | `src/assets/robots/unitree_g1/`, `src/tasks/effort/config/g1/`, `src/tasks/tracking/` |
