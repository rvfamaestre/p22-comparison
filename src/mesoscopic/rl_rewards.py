# -------------------------------------------------------------
# File: src/mesoscopic/rl_rewards.py
# -------------------------------------------------------------

def compute_residual_headway_reward(
    reward_cfg,
    cth_cfg,
    acc_cfg,
    v,
    s,
    mu_v,
    sigma_v2,
    alpha,
    a,
    a_prev,
    dt,
    e_leader=0.0,
):
    """
    Reward used by both PPO and SAC for residual headway control.

    The terms penalize unsafe spacing, poor spacing regulation, speed mismatch,
    jerk, and macroscopic speed variance.
    """
    d0 = cth_cfg.get("d0", 2.0)
    v_max = acc_cfg.get("v_max", 20.0)

    s_min = reward_cfg.get("s_min", d0)
    tau_min = reward_cfg.get("tau_min", 0.6)
    j_ref = reward_cfg.get("j_ref", 3.0)
    epsilon_v = reward_cfg.get("epsilon_v", 1e-3)
    epsilon_s = reward_cfg.get("epsilon_s", 1e-3)

    w_s = reward_cfg.get("w_s", 2.0)
    w_tau = reward_cfg.get("w_tau", 1.0)
    w_e = reward_cfg.get("w_e", 1.0)
    w_v = reward_cfg.get("w_v", 1.0)
    w_j = reward_cfg.get("w_j", 0.2)
    w_ss = reward_cfg.get("w_ss", 0.5)
    w_sigma2 = reward_cfg.get("w_sigma2", 0.1)

    s_des = d0 + alpha * v
    e = s - s_des
    tau = s / max(v, epsilon_v)
    jerk = (a - a_prev) / max(dt, 1e-3)

    r_sf = -(
        w_s * max(0.0, s_min - s) ** 2 +
        w_tau * max(0.0, tau_min - tau) ** 2
    )
    r_sp = -w_e * (e / max(s_des, epsilon_s)) ** 2
    r_ef = -w_v * ((v - mu_v) / max(v_max, epsilon_v)) ** 2
    r_cf = -w_j * (jerk / max(j_ref, 1e-3)) ** 2
    r_ss = -w_ss * max(0.0, abs(e) - abs(e_leader))
    r_sigma2 = -w_sigma2 * sigma_v2

    reward = r_sf + r_sp + r_ef + r_cf + r_ss + r_sigma2

    return float(reward)
