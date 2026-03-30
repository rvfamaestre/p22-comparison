"""Local GUI launcher for the project's existing CLI entrypoints.

Fields are auto-discovered from each script's ``build_parser()`` via
lightweight AST introspection — no heavy imports required at GUI startup.
"""

from __future__ import annotations

import argparse
import ast
import shlex
import subprocess
import sys
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from threading import Thread

import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk


ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
README_PATH = ROOT / "README.md"
MASTER_REFERENCE_PATH = ROOT / "Reports" / "MASTER_REFERENCE.md"
CLEANUP_REPORT_PATH = ROOT / "Reports" / "REPO_CLEANUP_REPORT.md"


# ---------------------------------------------------------------------------
# Data-classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    flag: str | None = None
    kind: str = "text"
    default: str = ""
    choices: tuple[str, ...] = ()
    browse: str | None = None
    true_flag: str | None = None
    false_flag: str | None = None
    help_text: str = ""
    required: bool = False


@dataclass(frozen=True)
class WorkflowSpec:
    key: str
    label: str
    category: str
    script: str
    runtime: str
    inputs: str
    outputs: str
    doc_path: Path
    doc_heading: str


# ---------------------------------------------------------------------------
# Workflow registry  (metadata only — fields come from build_parser())
# ---------------------------------------------------------------------------

WORKFLOWS: tuple[WorkflowSpec, ...] = (
    WorkflowSpec(
        key="simulation_run",
        label="Baseline Simulation",
        category="Simulation",
        script="run_simulation.py",
        runtime="Local CPU",
        inputs="YAML config for the inherited simulator/control stack.",
        outputs="Generated rollout data under output/<run>/ with micro, macro, and metadata files.",
        doc_path=README_PATH,
        doc_heading="### 1. Baseline sanity check",
    ),
    WorkflowSpec(
        key="train_sac",
        label="Train GPU SAC",
        category="Training",
        script="train_gpu_sac.py",
        runtime="HPC or local GPU",
        inputs="RL training config, optional PPO bootstrap checkpoint, and optional device/profile overrides.",
        outputs="SAC checkpoints plus effective config files under output/sac_train or the chosen output directory.",
        doc_path=README_PATH,
        doc_heading="### 2. Train the main RL model",
    ),
    WorkflowSpec(
        key="train_gpu_ppo",
        label="Train GPU PPO",
        category="Training",
        script="train_gpu_ppo.py",
        runtime="HPC or local GPU",
        inputs="RL training config plus optional device/profile and rollout overrides.",
        outputs="GPU PPO checkpoints plus effective config files under output/<run>/.",
        doc_path=README_PATH,
        doc_heading="### 2. Train the main RL model",
    ),
    WorkflowSpec(
        key="train_cpu_ppo",
        label="Train CPU PPO",
        category="Training",
        script="train_rl.py",
        runtime="Local CPU",
        inputs="A smoke or training YAML config for the legacy CPU PPO path.",
        outputs="Legacy PPO checkpoints under output/rl_*/.",
        doc_path=README_PATH,
        doc_heading="### 2. Train the main RL model",
    ),
    WorkflowSpec(
        key="eval_sac",
        label="Evaluate SAC",
        category="Evaluation",
        script="evaluate_gpu_sac.py",
        runtime="Local CPU or HPC/GPU",
        inputs="Evaluation config, SAC checkpoint, seed count, and optional device/env overrides.",
        outputs="Raw and summary CSV/JSON evaluation files under output/<run>/evaluation or the chosen output directory.",
        doc_path=README_PATH,
        doc_heading="### 3. Evaluate checkpoints",
    ),
    WorkflowSpec(
        key="eval_ppo",
        label="Evaluate PPO",
        category="Evaluation",
        script="evaluate_gpu_ppo.py",
        runtime="Local CPU or HPC/GPU",
        inputs="Evaluation config, PPO checkpoint, seed count, and optional device/env overrides.",
        outputs="Raw and summary CSV/JSON evaluation files under output/<run>/evaluation or the chosen output directory.",
        doc_path=README_PATH,
        doc_heading="### 3. Evaluate checkpoints",
    ),
    WorkflowSpec(
        key="rank_checkpoints",
        label="Rank Evaluated Checkpoints",
        category="Evaluation",
        script="rank_gpu_checkpoints.py",
        runtime="Local CPU",
        inputs="Evaluation root containing per-checkpoint evaluation folders.",
        outputs="Text ranking summary in the evaluation directory.",
        doc_path=README_PATH,
        doc_heading="### 3. Evaluate checkpoints",
    ),
    WorkflowSpec(
        key="visualize_sac",
        label="Visualize SAC Live/Replay",
        category="Visualization",
        script="visualize_sac.py",
        runtime="Local machine with a display",
        inputs="SAC checkpoint or replay directory, config, and optional mode/seed overrides.",
        outputs="Live visualization output or replay session data under output/_viz_live by default.",
        doc_path=README_PATH,
        doc_heading="### 4. Visualize behavior locally",
    ),
    WorkflowSpec(
        key="visualize_run",
        label="Visualize Stored RL Run",
        category="Visualization",
        script="visualize_run.py",
        runtime="Local machine with a display",
        inputs="Checkpoint or replay directory plus optional simulator overrides.",
        outputs="Interactive replay session and temporary output under the configured replay/output path.",
        doc_path=README_PATH,
        doc_heading="### 4. Visualize behavior locally",
    ),
    WorkflowSpec(
        key="experiment_sweep",
        label="Run Experiment Sweep",
        category="Experiments",
        script="run_experiments.py",
        runtime="Local CPU or HPC batch execution",
        inputs="Base config, mode list, checkpoint mapping, human-rate sweep, seed sweep, and output directory.",
        outputs="Per-run folders plus summary CSV files and standardized plots under the selected results directory.",
        doc_path=README_PATH,
        doc_heading="### 5. Run experiment sweeps",
    ),
    WorkflowSpec(
        key="summary_plots",
        label="Generate Comparison Plots",
        category="Plotting",
        script="generate_summary_compare_plots.py",
        runtime="Local CPU",
        inputs="summary_runs.csv file from an experiment sweep and a target output directory.",
        outputs="PNG and SVG comparison plots under the selected summary_plots directory.",
        doc_path=README_PATH,
        doc_heading="### 6. Generate comparison plots",
    ),
)


CATEGORY_ORDER = ("Simulation", "Training", "Evaluation", "Visualization", "Experiments", "Plotting", "Docs")


# ---------------------------------------------------------------------------
# AST-based parser introspection
# ---------------------------------------------------------------------------

_PATH_FILE_KEYWORDS = {"config", "checkpoint", "ckpt", "csv", "summary_csv", "ppo_ckpt"}
_PATH_DIR_KEYWORDS = {"dir", "root", "replay", "results_dir", "output_dir", "eval_root"}
_PATH_FILE_EXTENSIONS = (".yaml", ".yml", ".pt", ".csv", ".json")


def _dest_to_label(dest: str) -> str:
    """Convert ``some_flag_name`` to ``Some Flag Name``."""
    return dest.replace("_", " ").title()


def _is_path_arg(dest: str, default: object) -> bool:
    low = dest.lower()
    if any(kw in low for kw in (_PATH_FILE_KEYWORDS | _PATH_DIR_KEYWORDS)):
        return True
    if isinstance(default, str) and ("/" in default or any(default.endswith(ext) for ext in _PATH_FILE_EXTENSIONS)):
        return True
    return False


def _is_dir_arg(dest: str) -> bool:
    low = dest.lower()
    return any(kw in low for kw in _PATH_DIR_KEYWORDS)


def _parser_to_fields(parser: argparse.ArgumentParser) -> tuple[FieldSpec, ...]:
    """Convert an argparse parser's actions into :class:`FieldSpec` objects."""
    actions = [a for a in parser._actions if not isinstance(a, argparse._HelpAction)]

    # Group by dest to detect toggle pairs (store_true + store_false)
    by_dest: dict[str, list[argparse.Action]] = {}
    for a in actions:
        by_dest.setdefault(a.dest, []).append(a)

    fields: list[FieldSpec] = []
    seen: set[str] = set()

    for a in actions:
        if a.dest in seen:
            continue
        seen.add(a.dest)
        group = by_dest[a.dest]

        # --- toggle pair ---
        if len(group) >= 2:
            true_act = next((g for g in group if isinstance(g, argparse._StoreTrueAction)), None)
            false_act = next((g for g in group if isinstance(g, argparse._StoreFalseAction)), None)
            if true_act and false_act:
                fields.append(FieldSpec(
                    key=a.dest,
                    label=_dest_to_label(a.dest),
                    kind="toggle",
                    default="default",
                    choices=("default", "enable", "disable"),
                    true_flag=true_act.option_strings[0],
                    false_flag=false_act.option_strings[0],
                    help_text=true_act.help or "",
                ))
                continue

        act = group[0]
        flag = act.option_strings[0] if act.option_strings else None
        help_text = act.help or ""
        default_str = str(act.default) if act.default is not None else ""

        # --- store_true (standalone bool) ---
        if isinstance(act, argparse._StoreTrueAction):
            fields.append(FieldSpec(
                key=act.dest, label=_dest_to_label(act.dest),
                kind="bool", default="", true_flag=flag,
                help_text=help_text,
            ))
            continue

        # --- store_false without pair — skip ---
        if isinstance(act, argparse._StoreFalseAction):
            continue

        # --- choices → dropdown ---
        if act.choices:
            ch = tuple(str(c) for c in act.choices)
            if not act.required:
                ch = ("",) + ch
            fields.append(FieldSpec(
                key=act.dest, label=_dest_to_label(act.dest),
                flag=flag, kind="choice",
                default=default_str if default_str != "None" else "",
                choices=ch,
                help_text=help_text, required=act.required,
            ))
            continue

        # --- nargs="+" / "*" → tokens ---
        if act.nargs in ("+", "*"):
            if isinstance(act.default, (list, tuple)):
                tok_default = " ".join(str(v) for v in act.default)
            else:
                tok_default = default_str if default_str != "None" else ""
            fields.append(FieldSpec(
                key=act.dest, label=_dest_to_label(act.dest),
                flag=flag, kind="tokens", default=tok_default,
                help_text=help_text,
            ))
            continue

        # --- path heuristic ---
        if _is_path_arg(act.dest, act.default):
            browse = "dir" if _is_dir_arg(act.dest) else "file"
            fields.append(FieldSpec(
                key=act.dest, label=_dest_to_label(act.dest),
                flag=flag, kind="path",
                default=default_str if default_str != "None" else "",
                browse=browse,
                help_text=help_text, required=act.required,
            ))
            continue

        # --- generic text ---
        fields.append(FieldSpec(
            key=act.dest, label=_dest_to_label(act.dest),
            flag=flag, kind="text",
            default=default_str if default_str != "None" else "",
            help_text=help_text, required=act.required,
        ))

    return tuple(fields)


def _introspect_parser(script_path: Path) -> argparse.ArgumentParser | None:
    """Parse a script's source with AST, exec ``build_parser()``, return the parser.

    Only ``argparse``, ``pathlib.Path``, and module-level constant assignments
    are made available — heavy imports (torch, numpy, …) are skipped entirely,
    so this is virtually instant.
    """
    try:
        source = script_path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        tree = ast.parse(source, filename=str(script_path))
    except SyntaxError:
        return None

    namespace: dict[str, object] = {
        "argparse": argparse,
        "Path": Path,
        "__builtins__": __builtins__,
        "__file__": str(script_path),
    }

    # Execute lightweight imports and module-level constant assignments.
    # Heavy imports (torch, numpy, …) will fail silently and that is fine —
    # we only need the constants that build_parser() references.
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            try:
                code = compile(ast.Module(body=[node], type_ignores=[]), str(script_path), "exec")
                exec(code, namespace)  # noqa: S102
            except Exception:
                pass

    # … then the build_parser function itself.
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "build_parser":
            try:
                code = compile(ast.Module(body=[node], type_ignores=[]), str(script_path), "exec")
                exec(code, namespace)  # noqa: S102
                return namespace["build_parser"]()  # type: ignore[operator]
            except Exception:
                return None

    return None


# ---------------------------------------------------------------------------
# Field cache — introspect once, reuse everywhere
# ---------------------------------------------------------------------------

_FIELDS_CACHE: dict[str, tuple[FieldSpec, ...]] = {}


def get_fields(spec: WorkflowSpec) -> tuple[FieldSpec, ...]:
    """Return the auto-discovered :class:`FieldSpec` tuple for *spec*."""
    if spec.key not in _FIELDS_CACHE:
        parser = _introspect_parser(ROOT / spec.script)
        if parser is not None:
            _FIELDS_CACHE[spec.key] = _parser_to_fields(parser)
        else:
            _FIELDS_CACHE[spec.key] = ()
    return _FIELDS_CACHE[spec.key]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def category_specs(category: str) -> tuple[WorkflowSpec, ...]:
    return tuple(spec for spec in WORKFLOWS if spec.category == category)


def get_workflow(key: str) -> WorkflowSpec:
    return next(spec for spec in WORKFLOWS if spec.key == key)


def tokenize_field(text: str) -> list[str]:
    cleaned = text.replace(",", " ").split()
    return [token for token in cleaned if token]


def parse_extra_args(text: str) -> list[str]:
    stripped = text.strip()
    return shlex.split(stripped, posix=False) if stripped else []


def build_command(
    spec: WorkflowSpec,
    values: dict[str, object],
    extra_args: str = "",
    python_executable: str | None = None,
    root: Path | None = None,
) -> list[str]:
    fields = get_fields(spec)
    cmd = [python_executable or PYTHON, str((root or ROOT) / spec.script)]
    for field in fields:
        value = values.get(field.key, field.default)
        if field.kind == "bool":
            if bool(value) and field.true_flag:
                cmd.append(field.true_flag)
            continue
        if field.kind == "toggle":
            if value == "enable" and field.true_flag:
                cmd.append(field.true_flag)
            elif value == "disable" and field.false_flag:
                cmd.append(field.false_flag)
            continue
        if value in ("", None):
            continue
        if field.kind == "tokens":
            tokens = tokenize_field(str(value))
            if tokens and field.flag:
                cmd.append(field.flag)
                cmd.extend(tokens)
            continue
        if field.flag:
            cmd.extend([field.flag, str(value)])
    cmd.extend(parse_extra_args(extra_args))
    return cmd


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def extract_markdown_section(path: Path, heading: str) -> str:
    text = load_text(path)
    if not heading:
        return text
    lines = text.splitlines()
    start = None
    level = None
    for index, line in enumerate(lines):
        if line.strip() == heading.strip():
            start = index
            level = len(line) - len(line.lstrip("#"))
            break
    if start is None or level is None:
        return text
    collected: list[str] = []
    for index in range(start, len(lines)):
        line = lines[index]
        if index > start and line.startswith("#"):
            current_level = len(line) - len(line.lstrip("#"))
            if current_level <= level:
                break
        collected.append(line)
    return "\n".join(collected).strip()


# ---------------------------------------------------------------------------
# Tooltip helper
# ---------------------------------------------------------------------------

class _ToolTip:
    """Minimal hover-tooltip for tkinter widgets."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget = widget
        self.text = text
        self._tip_window: tk.Toplevel | None = None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._tip_window or not self.text:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            tw, text=self.text, justify=tk.LEFT, wraplength=400,
            background="#FFFFE0", relief=tk.SOLID, borderwidth=1,
            font=("Segoe UI", 9),
        )
        label.pack(ipadx=4, ipady=2)
        self._tip_window = tw

    def _hide(self, _event: tk.Event) -> None:  # type: ignore[type-arg]
        if self._tip_window:
            self._tip_window.destroy()
            self._tip_window = None


# ---------------------------------------------------------------------------
# Command runner
# ---------------------------------------------------------------------------

class CommandRunner:
    def __init__(self, console: scrolledtext.ScrolledText):
        self.console = console
        self.process: subprocess.Popen[str] | None = None

    def run(self, command: list[str]) -> None:
        self._write(f"$ {subprocess.list2cmdline(command)}\n\n", clear=True)
        thread = Thread(target=self._execute, args=(command,), daemon=True)
        thread.start()

    def stop(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()

    def _execute(self, command: list[str]) -> None:
        try:
            self.process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self._write(line)
            self.process.wait()
            self._write(f"\n[exit code {self.process.returncode}]\n")
        except Exception as exc:  # pragma: no cover - GUI runtime path
            self._write(f"\n[error] {exc}\n")

    def _write(self, text: str, clear: bool = False) -> None:
        self.console.configure(state="normal")
        if clear:
            self.console.delete("1.0", tk.END)
        self.console.insert(tk.END, text)
        self.console.see(tk.END)
        self.console.configure(state="disabled")


# ---------------------------------------------------------------------------
# Workflow tab — dynamic form from auto-discovered fields
# ---------------------------------------------------------------------------

class WorkflowTab:
    def __init__(self, parent: ttk.Notebook, specs: tuple[WorkflowSpec, ...], runner: CommandRunner):
        self.specs = specs
        self.runner = runner
        self.frame = ttk.Frame(parent, padding=10)
        self.spec_var = tk.StringVar(value=specs[0].key)
        self.extra_args = tk.StringVar()
        self.preview_var = tk.StringVar()
        self.value_vars: dict[str, tk.Variable] = {}
        self.form_widgets: list[tk.Widget] = []

        chooser_row = ttk.Frame(self.frame)
        chooser_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(chooser_row, text="Workflow:").pack(side=tk.LEFT)
        labels = [spec.label for spec in specs]
        self.label_to_key = {spec.label: spec.key for spec in specs}
        self.key_to_label = {spec.key: spec.label for spec in specs}
        chooser = ttk.Combobox(
            chooser_row,
            values=labels,
            state="readonly",
            width=28,
        )
        chooser.set(specs[0].label)
        chooser.pack(side=tk.LEFT, padx=(6, 0))
        chooser.bind("<<ComboboxSelected>>", lambda _event: self._on_workflow_change(chooser.get()))

        info_box = ttk.LabelFrame(self.frame, text="Runtime Context", padding=8)
        info_box.pack(fill=tk.X, pady=(0, 8))
        self.runtime_var = tk.StringVar()
        self.inputs_var = tk.StringVar()
        self.outputs_var = tk.StringVar()
        ttk.Label(info_box, textvariable=self.runtime_var, wraplength=520, justify=tk.LEFT).pack(anchor="w")
        ttk.Label(info_box, textvariable=self.inputs_var, wraplength=520, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))
        ttk.Label(info_box, textvariable=self.outputs_var, wraplength=520, justify=tk.LEFT).pack(anchor="w", pady=(4, 0))

        self.form_box = ttk.LabelFrame(self.frame, text="Command Options", padding=8)
        self.form_box.pack(fill=tk.X, pady=(0, 8))

        extra_box = ttk.LabelFrame(self.frame, text="Additional Arguments", padding=8)
        extra_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(extra_box, textvariable=self.extra_args).pack(fill=tk.X)
        ttk.Label(
            extra_box,
            text="Use this for any CLI flag not exposed in the form. Quote paths with spaces.",
            foreground="#5A6270",
        ).pack(anchor="w", pady=(4, 0))
        self.extra_args.trace_add("write", lambda *_args: self._refresh_preview())

        preview_box = ttk.LabelFrame(self.frame, text="Command Preview", padding=8)
        preview_box.pack(fill=tk.X, pady=(0, 8))
        ttk.Entry(preview_box, textvariable=self.preview_var, state="readonly").pack(fill=tk.X)

        button_row = ttk.Frame(self.frame)
        button_row.pack(fill=tk.X, pady=(0, 8))
        ttk.Button(button_row, text="Run", command=self._run).pack(side=tk.LEFT)
        ttk.Button(button_row, text="Reload Help", command=self._load_help_text).pack(side=tk.LEFT, padx=(6, 0))

        help_box = ttk.LabelFrame(self.frame, text="Canonical Help", padding=8)
        help_box.pack(fill=tk.BOTH, expand=True)
        self.help_text = scrolledtext.ScrolledText(
            help_box,
            wrap=tk.WORD,
            height=14,
            font=("Consolas", 9),
        )
        self.help_text.pack(fill=tk.BOTH, expand=True)
        self.help_text.configure(state="disabled")

        self._render_form(self.current_spec)

    @property
    def current_spec(self) -> WorkflowSpec:
        return next(spec for spec in self.specs if spec.key == self.spec_var.get())

    def _on_workflow_change(self, label: str) -> None:
        self.spec_var.set(self.label_to_key[label])
        self.extra_args.set("")
        self._render_form(self.current_spec)

    def _render_form(self, spec: WorkflowSpec) -> None:
        for widget in self.form_widgets:
            widget.destroy()
        self.form_widgets.clear()
        self.value_vars.clear()
        self.runtime_var.set(f"Run on: {spec.runtime}")
        self.inputs_var.set(f"Inputs: {spec.inputs}")
        self.outputs_var.set(f"Outputs: {spec.outputs}")

        fields = get_fields(spec)
        for row, fld in enumerate(fields):
            label_text = f"{fld.label}:" if not fld.required else f"{fld.label} *:"
            label = ttk.Label(self.form_box, text=label_text)
            label.grid(row=row, column=0, sticky="w", padx=(0, 8), pady=4)
            self.form_widgets.append(label)
            if fld.help_text:
                _ToolTip(label, fld.help_text)

            if fld.kind == "choice":
                var: tk.Variable = tk.StringVar(value=fld.default)
                widget = ttk.Combobox(self.form_box, textvariable=var, values=list(fld.choices), state="readonly")
                widget.grid(row=row, column=1, sticky="ew", pady=4)
                self.form_widgets.append(widget)
            elif fld.kind == "toggle":
                var = tk.StringVar(value=fld.default or "default")
                widget = ttk.Combobox(self.form_box, textvariable=var, values=list(fld.choices), state="readonly")
                widget.grid(row=row, column=1, sticky="ew", pady=4)
                self.form_widgets.append(widget)
            elif fld.kind == "bool":
                var = tk.BooleanVar(value=bool(fld.default))
                widget = ttk.Checkbutton(self.form_box, variable=var)
                widget.grid(row=row, column=1, sticky="w", pady=4)
                self.form_widgets.append(widget)
            else:
                var = tk.StringVar(value=fld.default)
                widget = ttk.Entry(self.form_box, textvariable=var)
                widget.grid(row=row, column=1, sticky="ew", pady=4)
                self.form_widgets.append(widget)
                if fld.browse:
                    browse = ttk.Button(
                        self.form_box,
                        text="...",
                        width=3,
                        command=lambda key=fld.key, mode=fld.browse: self._browse(key, mode),
                    )
                    browse.grid(row=row, column=2, sticky="w", padx=(6, 0), pady=4)
                    self.form_widgets.append(browse)
            var.trace_add("write", lambda *_args: self._refresh_preview())
            self.value_vars[fld.key] = var

        self.form_box.columnconfigure(1, weight=1)
        self._load_help_text()
        self._refresh_preview()

    def _browse(self, key: str, mode: str) -> None:
        if mode == "dir":
            selected = filedialog.askdirectory(initialdir=ROOT)
        else:
            selected = filedialog.askopenfilename(initialdir=ROOT)
        if not selected:
            return
        try:
            value = str(Path(selected).resolve().relative_to(ROOT))
        except ValueError:
            value = selected
        var = self.value_vars[key]
        if isinstance(var, tk.StringVar):
            var.set(value)

    def _current_values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, var in self.value_vars.items():
            values[key] = var.get()
        return values

    def _refresh_preview(self) -> None:
        command = build_command(self.current_spec, self._current_values(), self.extra_args.get())
        self.preview_var.set(subprocess.list2cmdline(command))

    def _load_help_text(self) -> None:
        text = extract_markdown_section(self.current_spec.doc_path, self.current_spec.doc_heading)
        self.help_text.configure(state="normal")
        self.help_text.delete("1.0", tk.END)
        self.help_text.insert("1.0", text)
        self.help_text.configure(state="disabled")

    def _run(self) -> None:
        command = build_command(self.current_spec, self._current_values(), self.extra_args.get())
        self.runner.run(command)


# ---------------------------------------------------------------------------
# Docs tab
# ---------------------------------------------------------------------------

class DocsTab:
    def __init__(self, parent: ttk.Notebook):
        self.frame = ttk.Frame(parent, padding=10)
        self.doc_map = {
            "README": README_PATH,
            "Master Reference": MASTER_REFERENCE_PATH,
            "Cleanup Report": CLEANUP_REPORT_PATH,
        }
        header = ttk.Frame(self.frame)
        header.pack(fill=tk.X, pady=(0, 8))
        ttk.Label(header, text="Document:").pack(side=tk.LEFT)
        self.doc_var = tk.StringVar(value="README")
        chooser = ttk.Combobox(
            header,
            textvariable=self.doc_var,
            values=list(self.doc_map),
            state="readonly",
            width=22,
        )
        chooser.pack(side=tk.LEFT, padx=(6, 0))
        chooser.bind("<<ComboboxSelected>>", lambda _event: self._load())
        self.text = scrolledtext.ScrolledText(self.frame, wrap=tk.WORD, font=("Consolas", 9))
        self.text.pack(fill=tk.BOTH, expand=True)
        self.text.configure(state="disabled")
        self._load()

    def _load(self) -> None:
        content = load_text(self.doc_map[self.doc_var.get()])
        self.text.configure(state="normal")
        self.text.delete("1.0", tk.END)
        self.text.insert("1.0", content)
        self.text.configure(state="disabled")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------

class ProjectGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("P22-MCFMT Launcher")
        self.root.geometry("1180x780")
        self.root.minsize(980, 620)

        style = ttk.Style()
        style.theme_use("clam")

        paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        left = ttk.Frame(paned)
        right = ttk.Frame(paned)
        paned.add(left, weight=1)
        paned.add(right, weight=1)

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.BOTH, expand=True)

        console_label = ttk.Label(right, text="Console")
        console_label.pack(anchor="w", padx=4, pady=(4, 0))
        console = scrolledtext.ScrolledText(
            right,
            wrap=tk.WORD,
            font=("Consolas", 9),
            bg="#10151C",
            fg="#E6EDF3",
            insertbackground="#E6EDF3",
        )
        console.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)
        console.configure(state="disabled")

        controls = ttk.Frame(right)
        controls.pack(fill=tk.X, padx=4, pady=(0, 4))
        self.runner = CommandRunner(console)
        ttk.Button(controls, text="Stop", command=self.runner.stop).pack(side=tk.RIGHT)
        ttk.Button(controls, text="Clear", command=lambda: self._clear_console(console)).pack(side=tk.RIGHT, padx=(0, 6))

        for category in CATEGORY_ORDER[:-1]:
            specs = category_specs(category)
            tab = WorkflowTab(notebook, specs, self.runner)
            notebook.add(tab.frame, text=f" {category} ")
        docs_tab = DocsTab(notebook)
        notebook.add(docs_tab.frame, text=" Docs ")

    @staticmethod
    def _clear_console(console: scrolledtext.ScrolledText) -> None:
        console.configure(state="normal")
        console.delete("1.0", tk.END)
        console.configure(state="disabled")


def launch_gui() -> None:
    root = tk.Tk()
    ProjectGUI(root)
    root.mainloop()
