#!/usr/bin/env bash

source ./.venv/bin/activate
python scripts/train.py Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-NoPhase-H5-MLP-L1Pose-NoCurr \
    --env.scene.num-envs 4096 \
    --agent.max-iterations 50000 \
    --agent.save-interval 1000 \
    --agent.seed 3407 \
    --agent.logger tensorboard \
    --agent.run-name "imu-jposnoankle-nojvel-h5-mlp-nocurr"
