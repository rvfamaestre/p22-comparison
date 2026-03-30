# -------------------------------------------------------------
# File: src/mesoscopic/rl_layer.py
# -------------------------------------------------------------

from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from src.mesoscopic.rl_common import log_prob_bounded_normal, sample_bounded_normal


# -------------------------------------------------------------
# RL configuration
# -------------------------------------------------------------

@dataclass
class RLConfig:
    enabled: bool = False
    mode: str = "inference"   # "inference" or "train"
    algorithm: str = "ppo"    # "ppo" or "sac"
    model_path: str = "models/rl_policy.pt"

    delta_alpha_max: float = 0.1
    alpha_min: float = 0.9
    alpha_max: float = 1.6


# -------------------------------------------------------------
# RL state container
# -------------------------------------------------------------

@dataclass
class RLState:
    """Minimal state summary exposed to the residual policy."""

    mu_v: float
    sigma_v2: float
    speed_mismatch: float
    v: float
    s: float
    delta_v: float
    a_lead: float
    alpha_prev: float


# -------------------------------------------------------------
# Actor-Critic network
# -------------------------------------------------------------

class ActorCritic(nn.Module):
    """Bounded-action PPO actor-critic used by the residual layer."""

    def __init__(self, state_dim=8, hidden=64):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.actor = nn.Linear(hidden, 1)
        self.critic = nn.Linear(hidden, 1)

        # Global log std for scalar action
        self.log_std = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # Ensure batch-safe behavior
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.shared(x)

        mean = self.actor(h)
        log_std = self.log_std.expand_as(mean)
        value = self.critic(h)

        return mean, log_std, value

    def sample_action(self, x, action_low, action_high, deterministic=False):
        mean, log_std, value = self.forward(x)
        action, log_prob, entropy = sample_bounded_normal(
            mean,
            log_std,
            action_low,
            action_high,
            deterministic=deterministic,
        )

        return action.squeeze(-1), log_prob, entropy, value.squeeze(-1)

    def evaluate_actions(self, x, actions, action_low, action_high):
        mean, log_std, value = self.forward(x)
        entropy = (0.5 + 0.5 * np.log(2.0 * np.pi) + log_std).sum(dim=-1)

        log_prob = log_prob_bounded_normal(
            mean,
            log_std,
            actions,
            action_low,
            action_high,
        )

        return log_prob, entropy, value.squeeze(-1)


def create_policy_model(algorithm: str, state_dim: int = 8):
    """Instantiate the policy network matching the selected RL algorithm."""
    algorithm = (algorithm or "ppo").lower()

    if algorithm == "ppo":
        return ActorCritic(state_dim)

    if algorithm == "sac":
        from src.mesoscopic.sac_training import SACActor
        return SACActor(state_dim)

    raise ValueError(f"Unsupported RL algorithm: {algorithm}")


# -------------------------------------------------------------
# RL residual headway layer
# -------------------------------------------------------------

class ResidualHeadwayRLLayer:
    """
    Residual controller that adjusts the rule-based mesoscopic headway parameter.

    The RL policy proposes a bounded residual delta_alpha, which is then added to
    the rule-based alpha produced by the mesoscopic adapter.
    """

    def __init__(self, config: RLConfig):
        self.config = config
        self.model = create_policy_model(config.algorithm)
        self.policy_available = True

        # Store previous alpha per vehicle
        self.alpha_prev = {}

        if config.enabled and config.mode == "inference":
            self.load_model(config.model_path)
            self.model.eval()
        else:
            # In train mode we keep the model in train mode
            self.model.train()

    # ---------------------------------------------------------
    # state construction
    # ---------------------------------------------------------

    def build_state(self, mu_v, sigma_v2, speed_mismatch, v, s, delta_v, a_lead, alpha_prev):
        """Package simulator measurements into the state structure used by RL."""
        return RLState(
            mu_v=mu_v,
            sigma_v2=sigma_v2,
            speed_mismatch=speed_mismatch,
            v=v,
            s=s,
            delta_v=delta_v,
            a_lead=a_lead,
            alpha_prev=alpha_prev
        )

    def state_to_vector(self, state: RLState):
        """Convert the structured RL state into the fixed feature order used by the networks."""
        return np.array([
            state.mu_v,
            state.sigma_v2,
            state.speed_mismatch,
            state.v,
            state.s,
            state.delta_v,
            state.a_lead,
            state.alpha_prev
        ], dtype=np.float32)

    # ---------------------------------------------------------
    # policy evaluation / action sampling
    # ---------------------------------------------------------

    def evaluate_policy(self, state_vector, action_low, action_high):
        # Disabled or unavailable policy -> zero residual.
        if not self.config.enabled or not self.policy_available:
            return {
                "raw_action": 0.0,
                "log_prob": 0.0,
                "value": 0.0
            }

        x = torch.tensor(state_vector, dtype=torch.float32)
        algorithm = (self.config.algorithm or "ppo").lower()
        deterministic = (self.config.mode == "inference")

        if algorithm == "sac":
            with torch.no_grad():
                action, log_prob, _ = self.model.sample(
                    x,
                    action_low=action_low,
                    action_high=action_high,
                    deterministic=deterministic,
                )

            return {
                "raw_action": float(action.item()),
                "log_prob": float(log_prob.item()),
                "value": 0.0
            }

        with torch.no_grad():
            action, log_prob, _, value = self.model.sample_action(
                x,
                action_low=action_low,
                action_high=action_high,
                deterministic=deterministic,
            )

        return {
            "raw_action": float(action.item()),
            "log_prob": float(log_prob.item()),
            "value": float(value.item())
        }

    # ---------------------------------------------------------
    # residual correction
    # ---------------------------------------------------------

    def apply_residual(self, alpha_rule, delta_alpha):
        """Final safety guard before the residual is applied to the rule-based alpha."""
        delta_lower = max(
            -self.config.delta_alpha_max,
            self.config.alpha_min - alpha_rule
        )
        delta_upper = min(
            self.config.delta_alpha_max,
            self.config.alpha_max - alpha_rule
        )

        delta_alpha = np.clip(delta_alpha, delta_lower, delta_upper)
        alpha = alpha_rule + delta_alpha

        return float(alpha), float(delta_alpha)

    def compute_action_bounds(self, alpha_rule):
        """Compute the feasible residual interval for the current rule-based alpha."""
        delta_lower = max(
            -self.config.delta_alpha_max,
            self.config.alpha_min - alpha_rule
        )
        delta_upper = min(
            self.config.delta_alpha_max,
            self.config.alpha_max - alpha_rule
        )
        return float(delta_lower), float(delta_upper)

    # ---------------------------------------------------------
    # main simulator interface
    # ---------------------------------------------------------

    def compute_alpha(
        self,
        cav_id,
        alpha_rule,
        mu_v,
        sigma_v2,
        speed_mismatch,
        v,
        s,
        delta_v,
        a_lead
    ):
        alpha_prev = self.alpha_prev.get(cav_id, alpha_rule)

        state = self.build_state(
            mu_v=mu_v,
            sigma_v2=sigma_v2,
            speed_mismatch=speed_mismatch,
            v=v,
            s=s,
            delta_v=delta_v,
            a_lead=a_lead,
            alpha_prev=alpha_prev
        )

        state_vector = self.state_to_vector(state)
        action_low, action_high = self.compute_action_bounds(alpha_rule)
        policy_out = self.evaluate_policy(state_vector, action_low, action_high)

        raw_delta_alpha = policy_out["raw_action"]
        alpha, delta_alpha = self.apply_residual(alpha_rule, raw_delta_alpha)

        self.alpha_prev[cav_id] = alpha

        return {
            "state_vector": state_vector,
            "alpha_rule": float(alpha_rule),
            "delta_alpha_raw": float(raw_delta_alpha),
            "delta_alpha": float(delta_alpha),
            "alpha": float(alpha),
            "action_low": float(action_low),
            "action_high": float(action_high),
            "log_prob": float(policy_out["log_prob"]),
            "value": float(policy_out["value"])
        }

    # ---------------------------------------------------------
    # utilities
    # ---------------------------------------------------------

    def reset(self):
        # Call this at the beginning of each episode
        self.alpha_prev = {}

    # ---------------------------------------------------------
    # model loading
    # ---------------------------------------------------------

    def load_model(self, path):
        try:
            state_dict = torch.load(path, map_location="cpu")
            try:
                self.model.load_state_dict(state_dict)
            except RuntimeError:
                self.model.load_state_dict(state_dict, strict=False)
                print("[RL] Loaded policy with compatibility fallback (strict=False)")
            print(f"[RL] Loaded policy from {path}")
        except FileNotFoundError:
            self.policy_available = False
            print(f"[RL] WARNING: model file not found: {path}")
            print("[RL] Running with zero residual policy")
