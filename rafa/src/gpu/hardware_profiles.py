"""Hardware-aware accelerator helpers for vectorized RL training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass(frozen=True)
class AcceleratorProfile:
    """Execution defaults for a class of accelerator hardware."""

    name: str
    description: str
    num_envs: int
    rollout_steps: int
    minibatch_size: int
    eval_num_envs: int


ACCELERATOR_PROFILES: Dict[str, AcceleratorProfile] = {
    "cpu_debug": AcceleratorProfile(
        name="cpu_debug",
        description="Conservative debug profile for CPU fallback and smoke tests.",
        num_envs=16,
        rollout_steps=64,
        minibatch_size=256,
        eval_num_envs=4,
    ),
    "local_8core_gpu": AcceleratorProfile(
        name="local_8core_gpu",
        description=(
            "Profile for smaller local GPUs, including 8-core integrated GPUs."
        ),
        num_envs=128,
        rollout_steps=256,
        minibatch_size=1024,
        eval_num_envs=8,
    ),
    "colab_t4": AcceleratorProfile(
        name="colab_t4",
        description="Profile matching the previous Colab/T4 notebook workflow.",
        num_envs=512,
        rollout_steps=256,
        minibatch_size=2048,
        eval_num_envs=16,
    ),
    "l4_large": AcceleratorProfile(
        name="l4_large",
        description="Larger CUDA profile for stronger datacenter GPUs.",
        num_envs=1024,
        rollout_steps=256,
        minibatch_size=4096,
        eval_num_envs=16,
    ),
}


def mps_is_available() -> bool:
    """Return True when the current PyTorch build exposes an MPS device."""
    return bool(
        getattr(torch.backends, "mps", None)
        and torch.backends.mps.is_built()
        and torch.backends.mps.is_available()
    )


def select_torch_device(requested: str = "auto") -> torch.device:
    """Resolve ``auto`` / explicit device strings to a torch device."""
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if mps_is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but no CUDA device is available.")
        return torch.device("cuda")

    if requested == "mps":
        if not mps_is_available():
            raise RuntimeError("MPS requested but no MPS device is available.")
        return torch.device("mps")

    if requested == "cpu":
        return torch.device("cpu")

    raise ValueError(f"Unsupported device request: {requested}")


def default_profile_name(device: torch.device) -> str:
    """Pick a sensible default execution profile for the selected device."""
    if device.type == "cuda":
        return "colab_t4"
    if device.type == "mps":
        return "local_8core_gpu"
    return "cpu_debug"


def resolve_profile(
    device: torch.device, profile_name: str = "auto"
) -> AcceleratorProfile:
    """Resolve a named or automatic execution profile."""
    chosen_name = default_profile_name(device) if profile_name == "auto" else profile_name
    try:
        return ACCELERATOR_PROFILES[chosen_name]
    except KeyError as exc:
        known = ", ".join(sorted(ACCELERATOR_PROFILES))
        raise ValueError(
            f"Unknown accelerator profile '{chosen_name}'. Known profiles: {known}."
        ) from exc


def empty_accelerator_cache(device: torch.device) -> None:
    """Release cached memory when the backend exposes a cache API."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
        return
    if device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
