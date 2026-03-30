import torch
import pytest
from src.gpu.vec_env import VecRingRoadEnv
from train_gpu_ppo import build_vec_env_config as build_vec_env_config_ppo
from train_gpu_sac import build_vec_env_config as build_vec_env_config_sac
from train_gpu_sac import load_config


BUILDERS = [
    pytest.param(build_vec_env_config_sac, id="sac_builder"),
    pytest.param(build_vec_env_config_ppo, id="ppo_builder"),
]


def get_base_cfg(build_cfg, shuffle: bool, meso_enabled: bool = True):
    cfg = load_config("config/rl_train.yaml")
    cfg["rl"]["shuffle_cav_positions"] = shuffle
    cfg["rl"]["hr_options"] = [0.5]
    cfg["rl"]["hr_weights"] = None
    cfg["mesoscopic"]["enabled"] = meso_enabled
    cfg["noise_Q"] = 0.0
    return build_cfg(cfg)


@pytest.mark.parametrize("build_cfg", BUILDERS)
def test_gpu_cav_shuffling_enabled(build_cfg):
    """Verify that CAVs are randomly interspersed when shuffle is enabled."""
    cfg = get_base_cfg(build_cfg, True)
    device = torch.device("cpu")
    env = VecRingRoadEnv(num_envs=10, cfg=cfg, device=device)
    env.reset()

    # The boolean mask of CAV presence should not be identical across runs
    cav_masks = env.is_cav
    
    # Ensure there are precisely 11 CAVs per environment
    assert (cav_masks.sum(dim=1) == 11).all()

    # If it is perfectly deterministic, the variance across environments would be 0
    # True shuffling means different environments have different CAV slots
    assert cav_masks.float().std(dim=0).mean() > 0.0


@pytest.mark.parametrize("build_cfg", BUILDERS)
def test_gpu_cav_shuffling_disabled(build_cfg):
    """Verify that CAVs are tightly clustered at the end when shuffle is disabled."""
    cfg = get_base_cfg(build_cfg, False)
    device = torch.device("cpu")
    env = VecRingRoadEnv(num_envs=10, cfg=cfg, device=device)
    env.reset()

    cav_masks = env.is_cav
    assert (cav_masks.sum(dim=1) == 11).all()

    # When disabled, all environments should have CAVs strictly in the last N slots
    # Standard deviation across environments should be exactly zero
    assert cav_masks.float().std(dim=0).mean() == 0.0
    
    # Ensure they are exactly at the end
    expected_mask = torch.zeros(22, dtype=torch.bool)
    expected_mask[11:] = True
    assert (cav_masks[0] == expected_mask).all()


@pytest.mark.parametrize("build_cfg", BUILDERS)
def test_gpu_meso_disabled_keeps_neutral_alpha(build_cfg):
    """Disabling mesoscopic adaptation should keep alpha at exactly 1.0."""
    cfg = get_base_cfg(build_cfg, shuffle=False, meso_enabled=False)
    device = torch.device("cpu")
    env = VecRingRoadEnv(num_envs=1, cfg=cfg, device=device)
    env.reset()
    env.v[0] = torch.linspace(1.0, 12.0, steps=cfg.N, device=device)
    env.step(torch.zeros(1, cfg.N, device=device))

    cav_alpha = env.alpha[0, env.is_cav[0]]
    assert torch.allclose(cav_alpha, torch.ones_like(cav_alpha))


@pytest.mark.parametrize("build_cfg", BUILDERS)
def test_gpu_meso_enabled_can_change_alpha(build_cfg):
    """Enabled mesoscopic adaptation should react to heterogeneous speeds."""
    cfg = get_base_cfg(build_cfg, shuffle=False, meso_enabled=True)
    device = torch.device("cpu")
    env = VecRingRoadEnv(num_envs=1, cfg=cfg, device=device)
    env.reset()
    env.v[0] = torch.linspace(1.0, 12.0, steps=cfg.N, device=device)
    env.step(torch.zeros(1, cfg.N, device=device))

    cav_alpha = env.alpha[0, env.is_cav[0]]
    assert torch.any((cav_alpha - 1.0).abs() > 1e-6)
