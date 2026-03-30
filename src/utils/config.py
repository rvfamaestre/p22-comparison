# -*- coding: utf-8 -*-

# Default IDM parameters
DEFAULT_IDM_PARAMS = {
    "s0": 2.0,          # minimum gap (m)
    "T": 1.2,           # desired time headway (s)
    "a": 1.2,           # maximum acceleration (m/s)
    "b": 1.5,           # comfortable deceleration (m/s)
    "v0": 33.0,         # desired speed (m/s)
    "delta": 4          # acceleration exponent
}

# Default ACC parameters
DEFAULT_ACC_PARAMS = {
    "ks": 0.2,          # spacing gain
    "kv": 0.6,          # velocity matching gain
    "kv0": 0.3,         # free-flow gain
    "b_max": 3.0,       # max braking (m/s)
    "v_max": 33.0,      # max speed (m/s)
    "v_des": 25.0       # desired free-flow speed (m/s)
}

# Default CTH parameters
DEFAULT_CTH_PARAMS = {
    "d0": 5.0,          # standstill distance (m)
    "hc": 1.0           # time headway constant (s)
}

# Default simulation parameters
DEFAULT_SIM_PARAMS = {
    "N": 20,                    # number of vehicles
    "L": 300.0,                 # ring length (m)
    "dt": 0.05,                 # timestep (s)
    "T": 60.0,                  # simulation duration (s)
    "initial_speed": 15.0,      # initial velocity (m/s)
    "human_ratio": 0.8,         # fraction of human vehicles
    "noise_Q": 0.2,             # human noise variance
    "seed": 42,                 # random seed
    "output_path": "output/default",  # output directory
    "dx": 1.0,                  # macro grid spacing
    "kernel_h": 3.0,            # SPH kernel smoothing length
    
    # Initial conditions (uniform vs random)
    "initial_conditions": "uniform",     # "uniform" or "random"
    "perturbation_enabled": True,        # Enable perturbation injection
    "perturbation_vehicle": 0,           # Vehicle ID to perturb
    "perturbation_time": 11.0,           # After warm-up + one 20-step baseline window
    "perturbation_delta_v": -2.0,        # Speed reduction (m/s)
    "noise_warmup_time": 10.0,           # Suppress noise until this time (0 = always on)
    "warmup_duration": 10.0,             # Clamp acceleration during startup transients
    "warmup_accel_limit": 1.0            # Warm-up acceleration limit (m/s^2)
}


def get_default_config():
    """
    Get complete default configuration dictionary.
    
    Returns:
    --------
    dict
        Full configuration with all default parameters
    """
    config = DEFAULT_SIM_PARAMS.copy()
    config["idm_params"] = DEFAULT_IDM_PARAMS.copy()
    config["acc_params"] = DEFAULT_ACC_PARAMS.copy()
    config["cth_params"] = DEFAULT_CTH_PARAMS.copy()
    return config


def merge_config(user_config, defaults=None):
    """
    Merge user configuration with defaults.
    
    Parameters:
    -----------
    user_config : dict
        User-provided configuration
    defaults : dict, optional
        Default configuration (uses get_default_config() if not provided)
    
    Returns:
    --------
    dict
        Merged configuration with user values taking precedence
    """
    if defaults is None:
        defaults = get_default_config()
    
    merged = defaults.copy()
    
    for key, value in user_config.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            # Recursively merge nested dictionaries
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    
    return merged


def validate_config(config):
    """
    Validate configuration parameters.
    
    Parameters:
    -----------
    config : dict
        Configuration to validate
    
    Raises:
    -------
    ValueError
        If configuration is invalid
    """
    # Check required top-level keys
    required_keys = ["N", "L", "dt", "T"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: '{key}'")
    
    # Validate numerical values
    if config["N"] <= 0:
        raise ValueError("N must be positive")
    if config["L"] <= 0:
        raise ValueError("L must be positive")
    if config["dt"] <= 0:
        raise ValueError("dt must be positive")
    if config["T"] <= 0:
        raise ValueError("T must be positive")
    if not 0 <= config.get("human_ratio", 0.8) <= 1:
        raise ValueError("human_ratio must be between 0 and 1")
    
    # Validate IDM parameters
    if "idm_params" in config:
        idm = config["idm_params"]
        if idm.get("s0", 2.0) < 0:
            raise ValueError("IDM s0 must be non-negative")
        if idm.get("T", 1.2) <= 0:
            raise ValueError("IDM T must be positive")
        if idm.get("a", 1.2) <= 0:
            raise ValueError("IDM a must be positive")
        if idm.get("b", 1.5) <= 0:
            raise ValueError("IDM b must be positive")
        if idm.get("v0", 33.0) <= 0:
            raise ValueError("IDM v0 must be positive")
    
    # Validate ACC parameters
    if "acc_params" in config:
        acc = config["acc_params"]
        if acc.get("b_max", 3.0) <= 0:
            raise ValueError("ACC b_max must be positive")
        if acc.get("v_max", 33.0) <= 0:
            raise ValueError("ACC v_max must be positive")
    
    return True
