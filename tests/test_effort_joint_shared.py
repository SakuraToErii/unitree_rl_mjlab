"""Joint-shared effort actor contracts from the V2 architecture diagram."""

from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import onnxruntime as ort
import torch
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.rl.runner import MjlabOnPolicyRunner
from tensordict import TensorDict

import src.tasks  # noqa: F401
from src.tasks.effort.config.g1.rl_cfg import unitree_g1_ppo_cross_attn_runner_cfg
from src.tasks.effort.rl.bf16_autocast import Bf16AutocastModel, Bf16AutocastPPO
from src.tasks.effort.rl.models import ResidualCrossAttnModel


TASK = (
  "Unitree-G1-Effort-0PD-Flat-IMU-JPosNoAnkle-NoJVel-H5-"
  "JointSharedXAttn-CmdFixed-ObsNorm"
)
BASELINE_TASK = TASK.replace("JointSharedXAttn", "JointXAttn")


def _model(batch_size: int = 3):
  cfg = asdict(load_rl_cfg(TASK).actor)
  cfg.pop("class_name")
  torch.manual_seed(0)
  obs = TensorDict(
    {"actor": torch.randn(batch_size, 5, 63)}, batch_size=[batch_size]
  )
  model = ResidualCrossAttnModel(obs, {"actor": ["actor"]}, "actor", 29, **cfg)
  return model, obs


class JointSharedActorTest(unittest.TestCase):
  def test_registered_v2_matches_design_and_keeps_the_environment(self):
    cfg = load_rl_cfg(TASK)
    baseline = load_rl_cfg(BASELINE_TASK)
    self.assertEqual(cfg.actor.encoder_hidden_dim, 128)
    self.assertEqual(cfg.actor.hidden_dims, (64,))
    self.assertTrue(cfg.actor.joint_shared_head)
    self.assertFalse(baseline.actor.joint_shared_head)
    self.assertNotEqual(cfg.experiment_name, baseline.experiment_name)
    env, baseline_env = load_env_cfg(TASK), load_env_cfg(BASELINE_TASK)
    for field in ("observations", "actions", "commands", "rewards", "curriculum"):
      self.assertEqual(getattr(env, field), getattr(baseline_env, field))
    self.assertEqual(
      cfg.algorithm.class_name,
      "src.tasks.effort.rl.bf16_autocast:Bf16AutocastPPO",
    )
    model, obs = _model()
    self.assertEqual(sum(p.numel() for p in model.parameters()), 167818)
    self.assertEqual(tuple(model.get_latent(obs).shape), (3, 29, 83))
    self.assertEqual(tuple(model(obs).shape), (3, 29))
    self.assertEqual(tuple(model.attention_weights(obs).shape), (3, 4, 29, 30))
    self.assertEqual(tuple(model.mlp.shared_trunk[0].weight.shape), (64, 83))
    self.assertEqual(tuple(model.mlp.joint_output.weight.shape), (29, 64))
    torch.testing.assert_close(model.mlp.joint_output.bias, torch.zeros(29))
    torch.testing.assert_close(
      model.mlp.joint_output.weight.norm(dim=1), torch.full((29,), 0.01)
    )

  def test_design_parameter_counts_for_other_observation_layouts(self):
    for variant, expected in (
      ("NoIMU-JPos-NoJVel", 163594),
      ("NoIMU-JPosNoAnkle-NoJVel", 163082),
    ):
      env_cfg = load_env_cfg(
        f"Unitree-G1-Effort-0PD-Flat-{variant}-H5-JointXAttn-CmdCurr-ObsNorm"
      )
      cfg = asdict(
        unitree_g1_ppo_cross_attn_runner_cfg(env_cfg, joint_shared_head=True).actor
      )
      cfg.pop("class_name")
      width = sum(dim for _, dim in cfg["term_dims"])
      obs = TensorDict({"actor": torch.randn(2, 5, width)}, batch_size=[2])
      model = ResidualCrossAttnModel(obs, {"actor": ["actor"]}, "actor", 29, **cfg)
      self.assertEqual(sum(p.numel() for p in model.parameters()), expected)
      self.assertEqual(tuple(model(obs).shape), (2, 29))

  def test_local_features_use_normalized_current_state_and_fixed_missing_flags(self):
    model, obs = _model()
    model.update_normalization(obs)
    sequence = model.obs_normalizer(obs["actor"])
    latent = model.get_latent(obs)
    positions, actions = model.cross_attention.slicer(sequence)
    torch.testing.assert_close(latent[:, :, -3], positions[:, -1])
    torch.testing.assert_close(latent[:, :, -2], actions[:, -1])
    torch.testing.assert_close(
      latent[:, :, -1], model.cross_attention.missing.expand(3, -1)
    )
    torch.testing.assert_close(latent[:, [4, 5, 10, 11], -3], torch.zeros(3, 4))
    torch.testing.assert_close(latent[:, [4, 5, 10, 11], -1], torch.ones(3, 4))
    torch.testing.assert_close(
      latent[:, :, 32:64], latent[:, :1, 32:64].expand(-1, 29, -1)
    )
    torch.testing.assert_close(
      latent[:, :, 64:80],
      model.cross_attention.joint_embedding.unsqueeze(0).expand(3, -1, -1),
    )

    cfg = asdict(load_rl_cfg(TASK).actor)
    cfg.pop("class_name")
    cfg["joint_shared_head"] = False
    baseline = ResidualCrossAttnModel(obs, {"actor": ["actor"]}, "actor", 29, **cfg)
    baseline.cross_attention.load_state_dict(
      {k: v for k, v in model.cross_attention.state_dict().items()
       if not k.startswith("current_projection.")},
      strict=True,
    )
    baseline.obs_normalizer.load_state_dict(model.obs_normalizer.state_dict())
    torch.testing.assert_close(
      model.attention_weights(obs), baseline.attention_weights(obs)
    )

  def test_shared_trunk_is_row_local_and_readout_is_per_joint(self):
    model, _ = _model()
    head = model.mlp
    features = torch.randn(3, 29, 83)
    output = head(features)
    changed = features.clone()
    changed[:, 0] += 1.0
    torch.testing.assert_close(head(changed)[:, 1:], output[:, 1:])
    identical = features[:, :1].expand(-1, 29, -1).contiguous()
    same_input = head(identical)
    self.assertFalse(torch.allclose(same_input[:, 0], same_input[:, 1]))
    with torch.no_grad():
      head.joint_output.weight[0] *= 2
      head.joint_output.bias[-1] += 0.5
    shifted = head(features)
    torch.testing.assert_close(shifted[:, 0], 2 * output[:, 0])
    torch.testing.assert_close(shifted[:, -1], output[:, -1] + 0.5)
    torch.testing.assert_close(shifted[:, 1:-1], output[:, 1:-1])

  def test_bf16_trunk_and_readout_backpropagate_with_fp32_parameters(self):
    model, obs = _model()
    dtypes = []
    hook = model.mlp.shared_trunk[0].register_forward_hook(
      lambda _module, _inputs, output: dtypes.append(output.dtype)
    )
    try:
      with torch.autocast("cpu", dtype=torch.bfloat16):
        output = model(obs)
      output.square().mean().backward()
    finally:
      hook.remove()
    self.assertEqual(dtypes, [torch.bfloat16])
    self.assertEqual(output.dtype, torch.float32)
    for param in model.parameters():
      self.assertEqual(param.dtype, torch.float32)
      if param.grad is not None:
        self.assertTrue(torch.isfinite(param.grad).all())
    for module in (model.cross_attention.current_projection, model.mlp):
      self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

  def test_ppo_can_update_the_shared_head_and_action_distribution(self):
    torch.manual_seed(0)
    cfg = asdict(load_rl_cfg(TASK))
    cfg["num_steps_per_env"] = 3
    cfg["algorithm"].update(num_learning_epochs=1, num_mini_batches=1)
    def observations():
      return TensorDict(
        {"actor": torch.randn(4, 5, 63), "critic": torch.randn(4, 11)},
        batch_size=[4],
      )
    obs = observations()
    env = SimpleNamespace(
      num_envs=4, num_actions=29, cfg={}, get_observations=lambda: obs
    )
    alg = MjlabOnPolicyRunner(env, cfg, device="cpu").alg
    self.assertIsInstance(alg, Bf16AutocastPPO)
    self.assertIsInstance(alg.actor, Bf16AutocastModel)
    before = alg.actor.mlp.joint_output.weight.detach().clone()
    with torch.no_grad():
      for _ in range(3):
        actions = alg.act(obs)
        self.assertEqual(tuple(actions.shape), (4, 29))
        self.assertEqual(alg.actor.get_output_log_prob(actions).dtype, torch.float32)
        obs = observations()
        alg.process_env_step(obs, torch.randn(4), torch.zeros(4, dtype=torch.bool), {})
      alg.compute_returns(obs)
    losses = alg.update()
    self.assertTrue(torch.isfinite(torch.tensor(list(losses.values()))).all())
    self.assertFalse(torch.equal(alg.actor.mlp.joint_output.weight, before))
    self.assertEqual(tuple(alg.actor.output_std.shape), (12, 29))
    restored, restored_obs = _model(batch_size=1)
    restored.load_state_dict(alg.save()["actor_state_dict"], strict=True)
    self.assertEqual(tuple(restored(restored_obs).shape), (1, 29))

  def test_jit_and_onnx_keep_learned_readout_and_dynamic_batch(self):
    model, obs = _model()
    model.update_normalization(obs)
    model.eval()
    with torch.no_grad():
      model.mlp.joint_output.weight.copy_(torch.linspace(0.5, 1.5, 29 * 64).view(29, 64))
      model.mlp.joint_output.bias.copy_(torch.linspace(-0.1, 0.1, 29))
    scripted = torch.jit.script(model.as_jit())
    with tempfile.TemporaryDirectory() as tmp:
      path = Path(tmp) / "joint_shared.onnx"
      torch.onnx.export(
        model.as_onnx(), (obs["actor"],), path,
        opset_version=17, dynamo=False,
        input_names=["obs"], output_names=["actions"],
        dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
      )
      session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
      for batch_size in (1, 3):
        batch = obs["actor"][:batch_size]
        with torch.no_grad():
          expected = model(TensorDict({"actor": batch}, batch_size=[batch_size]))
        torch.testing.assert_close(scripted(batch), expected)
        actual = session.run(["actions"], {"obs": batch.numpy()})[0]
        torch.testing.assert_close(torch.from_numpy(actual), expected, atol=1e-6, rtol=1e-5)

  def test_shared_head_requires_one_action_per_controlled_joint(self):
    _, obs = _model()
    cfg = asdict(load_rl_cfg(TASK).actor)
    cfg.pop("class_name")
    with self.assertRaisesRegex(ValueError, "one action per joint"):
      ResidualCrossAttnModel(obs, {"actor": ["actor"]}, "actor", 28, **cfg)

  @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
  def test_cuda_bf16_wrapper_outputs_and_gradients(self):
    if not torch.cuda.is_bf16_supported():
      self.skipTest("CUDA BF16 is unavailable")
    model, obs = _model()
    model, obs = model.to("cuda"), obs.to("cuda")
    wrapped = Bf16AutocastModel(model)
    output = wrapped(obs)
    output.square().mean().backward()
    self.assertEqual(output.dtype, torch.float32)
    self.assertTrue(torch.isfinite(output).all())
    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.mlp.parameters()))


if __name__ == "__main__":
  unittest.main()
