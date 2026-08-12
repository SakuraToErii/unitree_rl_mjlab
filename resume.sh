source ./.venv/bin/activate
python scripts/train.py TK3-Tracking \
  --motion-file datasets/backup_original/1_1_padding.npz \
  --env.scene.num-envs 8192 \
  --agent.max-iterations 50000 \
  --agent.save-interval 1000 \
  --agent.seed 3407 \
  --agent.logger wandb \
  --agent.wandb-project tk3-tracking-mjlab \
  --agent.run-name "非grounded1-1无后空翻版本" \
  --agent.resume True \
  --agent.load-run "2026-08-12_10-34-32_dexevt-100hz-天工的pd和scale-8192-grounded" \
  --agent.load-checkpoint "model_20000.pt"