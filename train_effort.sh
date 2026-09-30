#!/usr/bin/env bash
export CUDA_VISIBLE_DEVICES=0
source ./.venv/bin/activate
python scripts/train.py Unitree-G1-Effort-0PD-Flat-NoIMU-JPos-NoJVel-H5-MLP-CmdCurr-ObsNorm \
    --env.scene.num-envs 8192 \
    --agent.max-iterations 50000 \
    --agent.save-interval 1000 \
    --agent.seed 3407 \
    --agent.logger tensorboard \
    --agent.run-name "noimu-jpos-nojvel-h5-mlp"
