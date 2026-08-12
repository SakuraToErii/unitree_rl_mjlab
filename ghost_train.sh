source ./.venv/bin/activate
python scripts/train.py TK3-Ghost-Tracking \
  --motion-file datasets/1-1_paddingv2.npz \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 50000 \
  --agent.save-interval 5000 \
  --agent.seed 3407 \
  --agent.logger wandb \
  --agent.wandb-project tk3-tracking-mjlab \
  --agent.run-name "后空翻-physics-ghost-4096-8自然频率全0.25-简化观测"