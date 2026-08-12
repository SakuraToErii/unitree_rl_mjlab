source ./.venv/bin/activate
python scripts/train.py TK3-Tracking \
  --motion-file datasets/1-1_paddingv2.npz \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 50000 \
  --agent.save-interval 5000 \
  --agent.seed 3407 \
  --agent.logger wandb \
  --agent.wandb-project tk3-tracking-mjlab \
  --agent.run-name "后空翻-天工的pd和scale-8192"