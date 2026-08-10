source ./.venv/bin/activate
python scripts/train.py TK3-Tracking \
  --motion-file datasets/1_1_padding.npz \
  --env.scene.num-envs 16384 \
  --agent.max-iterations 50000 \
  --agent.save-interval 1000 \
  --agent.seed 3407 \
  --agent.logger wandb \
  --agent.wandb-project tk3-tracking-mjlab \
  --agent.run-name "dexevt-100hz" \
  --agent.resume True \
  --agent.load-run "2026-08-09_01-04-07_dexevt-100hz-天工的pd和scale-16384" \
  --agent.load-checkpoint "model_49500.pt"