from __future__ import annotations

from pathlib import Path

from src.gui import (
    ROOT,
    build_command,
    extract_markdown_section,
    get_fields,
    get_workflow,
)


def test_introspected_fields_for_sac_training() -> None:
    """build_parser() in train_gpu_sac.py is found and produces fields."""
    spec = get_workflow("train_sac")
    fields = get_fields(spec)
    field_keys = {f.key for f in fields}
    assert "config" in field_keys
    assert "device" in field_keys
    assert "ppo_ckpt" in field_keys
    assert "total_timesteps" in field_keys


def test_build_command_for_sac_training_with_extra_args() -> None:
    spec = get_workflow("train_sac")
    command = build_command(
        spec,
        {
            "config": "config/rl_train.yaml",
            "device": "cuda",
            "profile": "local_8core_gpu",
            "output_dir": "output/custom_sac",
            "num_envs": "64",
            "total_timesteps": "500000",
            "ppo_ckpt": "output/rl_train/ckpt_final.pt",
        },
        extra_args="--eval_num_envs 8",
        python_executable="python",
        root=ROOT,
    )

    assert command[:2] == ["python", str(ROOT / "train_gpu_sac.py")]
    assert "--config" in command
    assert "--device" in command
    assert "--ppo_ckpt" in command
    assert command[-2:] == ["--eval_num_envs", "8"]


def test_build_command_handles_tokens_and_toggles() -> None:
    spec = get_workflow("experiment_sweep")
    command = build_command(
        spec,
        {
            "config": "config/config.yaml",
            "modes": "baseline adaptive sac",
            "human_rates": "1.0 0.75",
            "seeds": "0,1",
            "ppo_checkpoint": "",
            "sac_checkpoint": "output/sac_train/gpu_sac_final.pt",
            "results_dir": "Results_compare_all",
            "insight_plots": True,
        },
        python_executable="python",
        root=ROOT,
    )

    assert command[:2] == ["python", str(ROOT / "run_experiments.py")]
    assert "--modes" in command
    assert "--insight_plots" in command


def test_extract_markdown_section_reads_canonical_help() -> None:
    section = extract_markdown_section(ROOT / "README.md", "### 6. Generate comparison plots")
    assert "python generate_summary_compare_plots.py" in section


def test_choices_become_dropdown_fields() -> None:
    """Arguments with choices should produce kind='choice' fields."""
    spec = get_workflow("visualize_sac")
    fields = get_fields(spec)
    mode_field = next((f for f in fields if f.key == "mode"), None)
    assert mode_field is not None
    assert mode_field.kind == "choice"
    assert "rl" in mode_field.choices
    assert "adaptive" in mode_field.choices
    assert "baseline" in mode_field.choices


def test_toggle_pairs_detected() -> None:
    """Paired --shuffle / --no-shuffle should produce a toggle field."""
    spec = get_workflow("visualize_sac")
    fields = get_fields(spec)
    shuffle_field = next((f for f in fields if f.key == "shuffle"), None)
    assert shuffle_field is not None
    assert shuffle_field.kind == "toggle"
    assert shuffle_field.true_flag == "--shuffle"
    assert shuffle_field.false_flag == "--no-shuffle"


def test_store_true_becomes_bool() -> None:
    """store_true args become kind='bool'."""
    spec = get_workflow("eval_sac")
    fields = get_fields(spec)
    skip_field = next((f for f in fields if f.key == "skip_plots"), None)
    assert skip_field is not None
    assert skip_field.kind == "bool"


def test_all_workflows_introspect_successfully() -> None:
    """Every registered workflow should produce at least one field."""
    from src.gui import WORKFLOWS
    for spec in WORKFLOWS:
        fields = get_fields(spec)
        assert len(fields) > 0, f"Workflow {spec.key!r} ({spec.script}) produced no fields"
