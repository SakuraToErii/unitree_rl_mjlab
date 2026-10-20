#!/usr/bin/env bash

source ./.venv/bin/activate
python scripts/train.py Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H1-MLP-L1Pose-NoCurr \
    --gpu-ids '[0,1]' \
    --env.scene.num-envs 2048 \
    --agent.max-iterations 50000 \
    --agent.save-interval 1000 \
    --agent.seed 3407 \
    --agent.logger tensorboard \
    --agent.run-name "imu-jposnoankle-nojvel-h1-mlp-nocurr"
