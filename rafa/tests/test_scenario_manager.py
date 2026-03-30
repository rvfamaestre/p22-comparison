from copy import deepcopy

from src.simulation.scenario_manager import ScenarioManager
from src.vehicles.cav_vehicle import CAVVehicle
from src.utils.logger import NullLogger


_BASE_CFG = {
    "N": 10,
    "L": 200.0,
    "dt": 0.1,
    "T": 1.0,
    "initial_speed": 10.0,
    "seed": 42,
    "output_path": "output/_test_scenario_manager",
    "initial_conditions": "uniform",
    "perturbation_enabled": False,
    "noise_warmup_time": 0.0,
    "enable_live_viz": False,
    "play_recording": False,
    "human_ratio": 0.5,
    "idm_params": {"s0": 2.0, "T": 1.2, "a": 1.0, "b": 1.5, "v0": 15.0, "delta": 4},
    "noise_Q": 0.0,
    "acc_params": {
        "ks": 0.4,
        "kv": 1.3,
        "kf": 0.7,
        "kv0": 0.05,
        "b_max": 3.0,
        "b_comfort": 1.5,
        "a_max": 1.5,
        "v_max": 20.0,
        "v_des": 15.0,
    },
    "cth_params": {"d0": 3.5, "hc": 1.0},
    "mesoscopic": {"enabled": True, "M": 4},
    "macro_teacher": "none",
    "save_macro_dataset": False,
    "dx": 1.0,
    "kernel_h": 3.0,
}


def _cav_ids(cfg):
    mgr = ScenarioManager(cfg)
    vehicles = mgr.create_vehicles()
    return [v.id for v in vehicles if isinstance(v, CAVVehicle)]


def test_nested_rl_shuffle_flag_is_honored_by_scenario_manager():
    cfg = deepcopy(_BASE_CFG)
    cfg["rl"] = {"shuffle_cav_positions": True}

    cav_ids = _cav_ids(cfg)
    default_tail_ids = list(range(int(cfg["N"] * cfg["human_ratio"]), cfg["N"]))

    assert cav_ids != default_tail_ids


def test_build_with_logging_disabled_uses_null_logger_and_skips_macro():
    cfg = deepcopy(_BASE_CFG)
    cfg["logging_enabled"] = False
    cfg["compute_macro_fields"] = False

    sim = ScenarioManager(cfg).build(live_viz=None)

    assert isinstance(sim.logger, NullLogger)
    assert sim.macro_gen is None


def test_simulator_initializes_and_preserves_sorted_topology():
    cfg = deepcopy(_BASE_CFG)
    sim = ScenarioManager(cfg).build(live_viz=None)

    initial_positions = [v.x for v in sim.env.vehicles]
    assert initial_positions == sorted(initial_positions)
    assert all(v.leader is not None for v in sim.env.vehicles)

    sim.step()

    step_positions = [v.x for v in sim.env.vehicles]
    assert step_positions == sorted(step_positions)
    assert all(v.leader is not None for v in sim.env.vehicles)


def test_simulator_run_forever_never_reports_done():
    cfg = deepcopy(_BASE_CFG)
    cfg["run_forever"] = True
    cfg["T"] = 0.2

    sim = ScenarioManager(cfg).build(live_viz=None)

    assert sim.steps is None
    assert sim.done is False

    sim.step()
    sim.step()
    sim.step()

    assert sim.done is False
