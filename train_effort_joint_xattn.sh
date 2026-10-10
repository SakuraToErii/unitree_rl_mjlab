#!/usr/bin/env bash

source ./.venv/bin/activate
python scripts/train.py Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-JointSharedXAttn-CmdFixed-ObsNorm \
    --env.scene.num-envs 8192 \
    --agent.max-iterations 50000 \
    --agent.save-interval 1000 \
    --agent.seed 3407 \
    --agent.logger tensorboard \
    --agent.run-name "imu-jposnoankle-nojvel-h5-jointsharedxattn-bf16" \
    --agent.actor.encoder-hidden-dim 64 \
    --agent.actor.hidden-dims '(64,)' \
    --agent.actor.joint-shared-head True \
    --agent.algorithm.class-name src.tasks.effort.rl.bf16_autocast:Bf16AutocastPPO
