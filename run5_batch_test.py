"""
Batch test checkpoints for multiple runner entry scripts.

By default, this script scans `runs/**/nn` for checkpoint names and dispatches
each checkpoint to the proper runner script based on the first subfolder under
`runs-dir` (algorithm folder):
    PPO/EPPO/PPPO/RPPO (+ _dist/_param variants) -> run5.py
    RARL/K-RARL (+ _dist/_param variants)         -> run_rarl.py
    SAC (+ _dist/_param variants)                 -> run_sac.py
    TD3 (+ _dist/_param variants)                 -> run_TD3.py
"""

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import importlib.util
import os
from queue import Empty, Queue
import re
import shlex
import subprocess
import sys
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from utils.disturbance_utils import redact_disturbance_scale_tags


SUPPORTED_ENV_NAMES = ("singlelane", "moose", "fixed_circle_iwd")
RAW_ENV_CHECKPOINT_NAMES = tuple(f"{env_name}.pth" for env_name in SUPPORTED_ENV_NAMES)
DEFAULT_CHECKPOINT_NAMES = RAW_ENV_CHECKPOINT_NAMES
ALLOWED_CHECKPOINT_NAMES = frozenset(DEFAULT_CHECKPOINT_NAMES)
ENV_TO_RENDER_SCRIPT_NAME = {
    "moose": "render_moose.py",
    "singlelane": "render_singlelane.py",
    "fixed_circle_iwd": "render_fixed_circle_iwd.py",
}
ALGORITHM_TO_RUNNER_KEY = {
    "PPO": "run5",
    "EPPO": "run5",
    "PPPO": "run5",
    "RPPO": "run5",
    "RARL": "run_rarl",
    "K-RARL": "run_rarl",
    "SAC": "run_sac",
    "TD3": "run_td3",
}
ALGORITHM_VARIANT_SUFFIXES = ("-DIST", "-PARAM")
SUPPORTED_ALGORITHM_FOLDER_HINT = (
    "PPO/EPPO/PPPO/RPPO/RARL/K-RARL/SAC/TD3 plus _dist/_param variants"
)
TIMESTAMP_SEED_PATTERN = re.compile(r"^(?P<base>.+?)_(?P<date>\d{4,8})_(?P<time>\d{6})_(?P<seed>\d+)$")
SEED_ONLY_PATTERN = re.compile(r"^(?P<base>.+?)_(?P<seed>\d+)$")
BATCH_ONLY_PASSTHROUGH_MODES = {
    "--devices": "many",
    "--runs-dir": "one",
    "--run5-script": "one",
    "--run-rarl-script": "one",
    "--run-sac-script": "one",
    "--run-td3-script": "one",
    "--render-script": "one",
    "--env": "one",
    "--checkpoint-names": "many",
    "--python-exe": "one",
    "--max-checkpoints": "one",
    "--output-dir": "one",
    "--dry-run": "flag",
    "--skip-render": "flag",
    "--split-panels": "flag",
    "--no-split-panels": "flag",
    "--stop-on-error": "flag",
    "--extra-run5-args": "many",
    "--extra-render-args": "many",
    "--max-plot-trajectories": "one",
}
LOG_REDACTED_ARG_COUNTS = {
    "--disturbance-scale": 1,
    "--disturbance-scales": 15,
}
DEFAULT_SPLIT_PANEL_EXPORT_FILENAMES = (
    "a_survival_distance.png",
    "b_roll_angle.png",
    "c_pitch_angle.png",
    "d_sideslip_angle.png",
    "e_trajectory_error.png",
    "f_signed_trajectory_error.png",
    "g_bcdf_stack.png",
)
HIGH_PARALLEL_RENDER_THRESHOLD = 100


def as_abs_path(base_dir: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def resolve_batch_output_root(project_root: Path, raw_output_dir: Optional[str], timestamp: str) -> Path:
    if raw_output_dir:
        return as_abs_path(project_root, raw_output_dir)
    return (project_root / "_batch_test_outputs" / f"batchtest_{timestamp}").resolve()


def collect_checkpoints(runs_dir: Path, checkpoint_names: Iterable[str]) -> List[Path]:
    found = []
    for name in dict.fromkeys(checkpoint_names):
        if name not in ALLOWED_CHECKPOINT_NAMES:
            continue
        for path in runs_dir.rglob(name):
            if path.is_file() and path.parent.name == "nn":
                found.append(path.resolve())
    unique_sorted = sorted(set(found))
    return unique_sorted


def filter_allowed_checkpoint_names(checkpoint_names: Iterable[str]) -> Tuple[List[str], List[str]]:
    allowed: List[str] = []
    ignored: List[str] = []
    for name in dict.fromkeys(checkpoint_names):
        if name in ALLOWED_CHECKPOINT_NAMES:
            allowed.append(name)
        else:
            ignored.append(name)
    return allowed, ignored


def collect_rollout_dirs(runs_dir: Path) -> set:
    rollout_dirs = set()
    for path in runs_dir.rglob("*"):
        if path.parent.name != "rollouts":
            continue
        if path.is_dir():
            rollout_dirs.add(path.resolve())
    return rollout_dirs


def latest_rollout_dir(candidates: Iterable[Path]) -> Optional[Path]:
    candidates = list(candidates)
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def model_run_dir_name_from_checkpoint(checkpoint: Path) -> str:
    # .../runs/<model_dir>/nn/<checkpoint>.pth
    return checkpoint.parent.parent.name


def strip_timestamp_seed_suffix(name: str) -> str:
    match = TIMESTAMP_SEED_PATTERN.match(name)
    if match:
        return str(match.group("base"))
    return name


def strip_seed_suffix(name: str) -> str:
    match = SEED_ONLY_PATTERN.match(name)
    if match:
        return str(match.group("base"))
    return name


def build_group_name_lookup(run_dir_names: Sequence[str]) -> Dict[str, str]:
    ordered_names = list(dict.fromkeys(run_dir_names))
    lookup: Dict[str, str] = {}
    seed_only_candidates: Dict[str, List[str]] = defaultdict(list)

    for run_dir_name in ordered_names:
        timestamp_base = strip_timestamp_seed_suffix(run_dir_name)
        if timestamp_base != run_dir_name:
            lookup[run_dir_name] = timestamp_base
            continue
        seed_base = strip_seed_suffix(run_dir_name)
        if seed_base != run_dir_name:
            seed_only_candidates[seed_base].append(run_dir_name)
            continue
        lookup[run_dir_name] = run_dir_name

    for seed_base, grouped_names in seed_only_candidates.items():
        if len(grouped_names) >= 2 or seed_base in lookup.values():
            for run_dir_name in grouped_names:
                lookup[run_dir_name] = seed_base
        else:
            for run_dir_name in grouped_names:
                lookup[run_dir_name] = run_dir_name

    return lookup


def build_checkpoint_group_lookup(checkpoints: Sequence[Path]) -> Dict[Path, str]:
    run_dir_names = [model_run_dir_name_from_checkpoint(checkpoint) for checkpoint in checkpoints]
    group_lookup = build_group_name_lookup(run_dir_names)
    return {
        checkpoint: group_lookup[model_run_dir_name_from_checkpoint(checkpoint)]
        for checkpoint in checkpoints
    }


def image_stem_from_model_group(model_group_name: str) -> str:
    return model_group_name


def normalize_algorithm_folder_name(value: str) -> str:
    return value.strip().upper().replace("_", "-")


def infer_algorithm_name_from_checkpoint(checkpoint: Path, runs_dir: Path) -> Optional[str]:
    try:
        relative_path = checkpoint.resolve().relative_to(runs_dir.resolve())
    except ValueError:
        return None
    if not relative_path.parts:
        return None
    return relative_path.parts[0]


def resolve_runner_key_from_algorithm_name(algorithm_name: str) -> Optional[str]:
    normalized_name = normalize_algorithm_folder_name(algorithm_name)
    runner_key = ALGORITHM_TO_RUNNER_KEY.get(normalized_name)
    if runner_key is not None:
        return runner_key
    for suffix in ALGORITHM_VARIANT_SUFFIXES:
        if normalized_name.endswith(suffix):
            return ALGORITHM_TO_RUNNER_KEY.get(normalized_name[: -len(suffix)])
    return None


def infer_env_name_from_checkpoint(checkpoint: Path) -> Optional[str]:
    checkpoint_name = checkpoint.name.lower()
    run_dir_name = checkpoint.parent.parent.name.lower()

    for env_name in SUPPORTED_ENV_NAMES:
        known_names = {
            f"{env_name}.pth",
            f"{env_name}_pro_for_runpy.pth",
        }
        if checkpoint_name in known_names:
            return env_name

    for env_name in SUPPORTED_ENV_NAMES:
        if env_name in checkpoint_name or env_name in run_dir_name:
            return env_name

    return None


def infer_env_name_from_value(value: object) -> Optional[str]:
    lowered = str(value).lower()
    if "fixed_circle" in lowered:
        return "fixed_circle_iwd"
    for env_name in SUPPORTED_ENV_NAMES:
        if env_name in lowered:
            return env_name
    return None


def resolve_render_script(
    project_root: Path,
    raw_render_script: str,
    runs_dir: Path,
    checkpoints: Sequence[Path],
    requested_env_name: str,
) -> Tuple[Path, Optional[str], str]:
    if raw_render_script.lower() != "auto":
        return as_abs_path(project_root, raw_render_script), None, "explicit"

    detected_env_name = infer_env_name_from_value(runs_dir)
    detection_reason = "--runs-dir"
    if detected_env_name is None:
        checkpoint_env_names = {
            env_name
            for env_name in (infer_env_name_from_checkpoint(checkpoint) for checkpoint in checkpoints)
            if env_name is not None
        }
        if len(checkpoint_env_names) == 1:
            detected_env_name = next(iter(checkpoint_env_names))
            detection_reason = "checkpoint-set"
    if detected_env_name is None and requested_env_name != "auto":
        detected_env_name = infer_env_name_from_value(requested_env_name)
        detection_reason = "--env"

    render_script_name = ENV_TO_RENDER_SCRIPT_NAME.get(detected_env_name, "render5.py")
    return (project_root / render_script_name).resolve(), detected_env_name, detection_reason


def build_render_command(
    python_exe: str,
    render_script: Path,
    rollout_dirs: Sequence[Path],
    extra_render_args: List[str],
    model_label: Optional[str] = None,
    output_dir: Optional[Path] = None,
    skip_summary_table: bool = False,
    split_panels: bool = False,
    hide_panel_titles: bool = False,
) -> List[str]:
    primary_rollout_dir = rollout_dirs[0]
    cmd = [
        python_exe,
        str(render_script),
        str(primary_rollout_dir),
        "--no-display",
    ]
    cmd.extend(extra_render_args)
    cmd.append("--split-panels" if split_panels else "--no-split-panels")
    if model_label:
        cmd.extend(["--model-label", model_label])
    if output_dir is not None:
        cmd.extend(["--output-dir", str(output_dir)])
    if len(rollout_dirs) > 1:
        cmd.extend(["--extra-input-dirs", *(str(path) for path in rollout_dirs[1:])])
    if hide_panel_titles:
        cmd.append("--hide-panel-titles")
    if skip_summary_table:
        cmd.append("--skip-summary-table")
    return cmd


def load_render_module_and_args(
    render_script: Path,
    extra_render_args: List[str],
    split_panels: bool,
):
    module_name = f"_batch_render_{sanitize_token(render_script.stem)}_{abs(hash(str(render_script.resolve())))}"
    spec = importlib.util.spec_from_file_location(module_name, render_script)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load render module from {render_script}")
    render_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(render_module)

    parser = render_module.build_parser()
    render_args, _ = parser.parse_known_args(
        [
            "placeholder",
            "--no-display",
            *extra_render_args,
            "--split-panels" if split_panels else "--no-split-panels",
        ]
    )
    return render_module, render_args


def get_render_split_panel_filenames(render_module) -> List[str]:
    module_getter = getattr(render_module, "get_split_panel_filenames", None)
    if callable(module_getter):
        file_names = [str(name) for name in module_getter()]
        if file_names:
            return file_names
    return list(DEFAULT_SPLIT_PANEL_EXPORT_FILENAMES)


def expected_render_output_path(
    primary_rollout_dir: Path,
    output_name: str,
    model_label: Optional[str] = None,
    output_dir: Optional[Path] = None,
    split_panels: bool = False,
) -> Path:
    if output_dir is None or output_dir.resolve() == primary_rollout_dir.resolve():
        stem = output_name
        base_dir = primary_rollout_dir
    else:
        stem = output_name if not model_label else f"{model_label}_{output_name}"
        base_dir = output_dir
    if split_panels:
        return base_dir / stem
    return base_dir / f"{stem}.png"


def render_output_exists(
    output_path: Path,
    split_panels: bool,
    split_panel_filenames: Optional[Sequence[str]] = None,
) -> bool:
    if not split_panels:
        return output_path.is_file()
    if not output_path.is_dir():
        return False
    expected_files = split_panel_filenames or DEFAULT_SPLIT_PANEL_EXPORT_FILENAMES
    return all((output_path / file_name).is_file() for file_name in expected_files)


def build_test_command(
    python_exe: str,
    runner_script: Path,
    runner_key: str,
    env_name: str,
    checkpoint: Path,
    car_preset: str,
    mlp_size_last: int,
    device: str,
    extra_run5_args: List[str],
) -> List[str]:
    cmd = [
        python_exe,
        str(runner_script),
        "test",
        env_name,
        "--checkpoint",
        str(checkpoint),
        "--car-preset",
        car_preset,
        "--device",
        device,
    ]
    # run_rarl.py does not expose --mlp-size-last, while PPO/SAC/TD3 runners do.
    if runner_key != "run_rarl":
        cmd.extend(["--mlp-size-last", str(mlp_size_last)])
    cmd.extend(extra_run5_args)
    return cmd


def format_command_for_log(cmd: Sequence[str]) -> str:
    redacted: List[str] = []
    index = 0
    while index < len(cmd):
        token = str(cmd[index])
        option_name, has_inline_value, _ = token.partition("=")
        value_count = LOG_REDACTED_ARG_COUNTS.get(option_name)
        if value_count is None:
            redacted.append(redact_disturbance_scale_tags(token))
            index += 1
            continue

        redacted.append("<disturbance-args-hidden>")
        if has_inline_value:
            index += 1
        else:
            index += 1 + value_count

    return " ".join(shlex.quote(part) for part in redacted)


def sanitize_token(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return cleaned or "x"


def build_seed_render_label(task_index: int, algorithm_name: str, model_name: str, checkpoint: Path) -> str:
    clean_model_name = redact_disturbance_scale_tags(model_name)
    label = sanitize_token(f"{task_index:03d}_{algorithm_name}_{clean_model_name}_{checkpoint.stem}")
    return label[:120]


def find_named_rollout_dir(rollout_root: Path, rollout_dir_name: str) -> Optional[Path]:
    candidates = []
    direct = rollout_root / rollout_dir_name
    if direct.is_dir():
        candidates.append(direct.resolve())
    if rollout_root.exists():
        candidates.extend(
            path.resolve()
            for path in rollout_root.glob(f"{rollout_dir_name}_*")
            if path.is_dir()
        )
    return latest_rollout_dir(candidates)


def collect_rollout_dirs_for_run_dir(run_dir: Path) -> set:
    rollout_root = run_dir / "rollouts"
    if not rollout_root.exists():
        return set()
    rollout_dirs = set()
    for path in rollout_root.iterdir():
        if path.is_dir():
            rollout_dirs.add(path.resolve())
    return rollout_dirs


def locate_rollout_dir(
    runs_dir: Path,
    project_root: Path,
    preferred_run_dir: Path,
    rollout_dir_name: str,
    preferred_rollout_roots: Sequence[Path] = (),
) -> Optional[Path]:
    for rollout_root in preferred_rollout_roots:
        rollout_dir = find_named_rollout_dir(rollout_root, rollout_dir_name)
        if rollout_dir is not None:
            return rollout_dir
    if preferred_rollout_roots:
        return None

    preferred_rollout_root = preferred_run_dir / "rollouts"
    preferred_rollout_dir = find_named_rollout_dir(preferred_rollout_root, rollout_dir_name)
    if preferred_rollout_dir is not None:
        return preferred_rollout_dir

    global_candidates = []
    if runs_dir.exists():
        global_candidates.extend(
            path.resolve()
            for path in runs_dir.rglob(rollout_dir_name)
            if path.is_dir() and path.parent.name == "rollouts"
        )
        global_candidates.extend(
            path.resolve()
            for path in runs_dir.rglob(f"{rollout_dir_name}_*")
            if path.is_dir() and path.parent.name == "rollouts"
        )
    if global_candidates:
        return latest_rollout_dir(global_candidates)

    data_dir = project_root / "data"
    data_candidates = []
    direct_data = data_dir / rollout_dir_name
    if direct_data.is_dir():
        data_candidates.append(direct_data.resolve())
    if data_dir.exists():
        data_candidates.extend(
            path.resolve()
            for path in data_dir.glob(f"{rollout_dir_name}_*")
            if path.is_dir()
        )
    return latest_rollout_dir(data_candidates)


def build_rollout_dir_name(
    checkpoint: Path,
    task_index: int,
    device: str,
) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir_token = sanitize_token(redact_disturbance_scale_tags(model_run_dir_name_from_checkpoint(checkpoint)))[:48]
    device_token = sanitize_token(device)
    return f"batchtest_{timestamp}_{task_index:03d}_{device_token}_{run_dir_token}"


def tail_text(text: str, max_lines: int = 40) -> str:
    if not text:
        return ""
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-max_lines:])


def extract_rollout_dir_from_output(project_root: Path, output_text: str) -> Optional[Path]:
    if not output_text:
        return None
    match = re.search(r"Saved rollout data .* directory:\s*(.+)", output_text)
    if not match:
        return None
    rollout_dir = Path(match.group(1).strip())
    if not rollout_dir.is_absolute():
        rollout_dir = (project_root / rollout_dir).resolve()
    return rollout_dir


def rollout_dir_has_render_data(rollout_dir: Path) -> bool:
    if (rollout_dir / "render_rollout.npz").is_file():
        return True
    return any(path.is_file() and path.suffix.lower() == ".csv" for path in rollout_dir.glob("*.csv"))


def rollout_dir_trajectory_count(rollout_dir: Path) -> int:
    npz_path = rollout_dir / "render_rollout.npz"
    if npz_path.is_file():
        try:
            with np.load(npz_path, allow_pickle=False) as archive:
                if "data" in archive.files:
                    data = archive["data"]
                    if data.ndim >= 1:
                        return int(data.shape[0])
        except Exception:
            return 0
    return sum(1 for path in rollout_dir.glob("*.csv") if path.is_file())


def has_high_parallel_rollout(rollout_dirs: Sequence[Path], threshold: int = HIGH_PARALLEL_RENDER_THRESHOLD) -> bool:
    return any(rollout_dir_trajectory_count(path) > threshold for path in rollout_dirs)


def strip_batch_only_passthrough_args(raw_args: Sequence[str]) -> Tuple[List[str], List[str]]:
    cleaned: List[str] = []
    dropped: List[str] = []
    index = 0
    while index < len(raw_args):
        token = raw_args[index]
        option_name, has_inline_value, _ = token.partition("=")
        mode = BATCH_ONLY_PASSTHROUGH_MODES.get(option_name)
        if mode is None:
            cleaned.append(token)
            index += 1
            continue

        dropped.append(token)
        index += 1
        if has_inline_value:
            continue
        if mode == "flag":
            continue
        if mode == "one":
            if index < len(raw_args) and not raw_args[index].startswith("-"):
                dropped.append(raw_args[index])
                index += 1
            continue
        if mode == "many":
            while index < len(raw_args) and not raw_args[index].startswith("-"):
                dropped.append(raw_args[index])
                index += 1
            continue

    return cleaned, dropped


def print_with_lock(print_lock: threading.Lock, message: str) -> None:
    with print_lock:
        print(message)


def run_single_checkpoint_task(
    task: Dict[str, object],
    device: str,
    project_root: Path,
    runs_dir: Path,
    render_script: Path,
    split_panel_filenames: Sequence[str],
    rollout_output_root: Path,
    seed_figure_dir: Path,
    runner_log_root: Path,
    args: argparse.Namespace,
    render_args: Optional[argparse.Namespace],
    extra_runner_args: List[str],
    print_lock: threading.Lock,
) -> Dict[str, object]:
    checkpoint = Path(task["checkpoint"])
    model_name = str(task["model_name"])
    model_group_name = str(task["model_group_name"])
    algorithm_name = str(task["algorithm_name"])
    runner_key = str(task["runner_key"])
    runner_script = Path(task["runner_script"])
    env_name = str(task["env_name"])
    task_index = int(task["task_index"])
    run_dir = checkpoint.parent.parent.resolve()

    result_payload: Dict[str, object] = {
        "task_index": task_index,
        "checkpoint": checkpoint,
        "model_name": model_name,
        "model_group_name": model_group_name,
        "algorithm_name": algorithm_name,
        "runner_key": runner_key,
        "runner_script": runner_script,
        "env_name": env_name,
        "device": device,
        "run_dir": run_dir,
        "passed": False,
        "failure": None,
        "rollout_dir": None,
        "seed_render_path": None,
        "high_parallel": False,
    }

    rollout_dir_name = build_rollout_dir_name(checkpoint=checkpoint, task_index=task_index, device=device)
    expected_rollout_dir = rollout_output_root / rollout_dir_name
    seed_render_label = build_seed_render_label(
        task_index=task_index,
        algorithm_name=algorithm_name,
        model_name=model_name,
        checkpoint=checkpoint,
    )

    cmd = build_test_command(
        python_exe=args.python_exe,
        runner_script=runner_script,
        runner_key=runner_key,
        env_name=env_name,
        checkpoint=checkpoint,
        car_preset=args.car_preset,
        mlp_size_last=args.mlp_size_last,
        device=device,
        extra_run5_args=extra_runner_args,
    )
    cmd_str = format_command_for_log(cmd)
    print_with_lock(
        print_lock,
        (
            f"\n[RUN {task_index}] {checkpoint.name} "
            f"({redact_disturbance_scale_tags(model_name)}, algo={algorithm_name}, "
            f"env={env_name}, device={device}, runner={runner_script.name})\n"
            f"{cmd_str}"
        ),
    )

    run_env = os.environ.copy()
    run_env["XCAR_ROLLOUT_NAME"] = rollout_dir_name
    run_env["XCAR_ROLLOUT_ROOT"] = str(rollout_output_root)
    run_env["XCAR_RUN_ROOT_DIR"] = str(runner_log_root)
    result = subprocess.run(
        cmd,
        cwd=str(project_root),
        env=run_env,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        result_payload["failure"] = result.returncode
        combined_tail = redact_disturbance_scale_tags(
            tail_text("\n".join([result.stdout or "", result.stderr or ""]))
        )
        if combined_tail:
            print_with_lock(
                print_lock,
                f"[FAIL {task_index}] {checkpoint.name} on {device}\n{combined_tail}",
            )
        return result_payload

    result_payload["passed"] = True
    combined_output = "\n".join([result.stdout or "", result.stderr or ""])
    rollout_dir = extract_rollout_dir_from_output(project_root=project_root, output_text=combined_output)
    if rollout_dir is None:
        rollout_dir = locate_rollout_dir(
            runs_dir=runs_dir,
            project_root=project_root,
            preferred_run_dir=run_dir,
            rollout_dir_name=rollout_dir_name,
            preferred_rollout_roots=(rollout_output_root,),
        )
    if rollout_dir is None or not rollout_dir.exists():
        result_payload["failure"] = "rollout-missing"
        print_with_lock(
            print_lock,
            (
                f"[WARN] no rollout directory detected for {checkpoint.name}\n"
                f"Expected preferred path: {redact_disturbance_scale_tags(expected_rollout_dir)}\n"
                f"Recent output tail:\n{redact_disturbance_scale_tags(tail_text(combined_output))}"
            ),
        )
        return result_payload
    if not rollout_dir_has_render_data(rollout_dir):
        result_payload["failure"] = "rollout-data-missing"
        print_with_lock(
            print_lock,
            (
                f"[WARN] rollout directory has no render_rollout.npz or CSV files for {checkpoint.name}: "
                f"{redact_disturbance_scale_tags(rollout_dir)}"
            ),
        )
        return result_payload

    result_payload["rollout_dir"] = rollout_dir
    trajectory_count = rollout_dir_trajectory_count(rollout_dir)
    high_parallel = trajectory_count > HIGH_PARALLEL_RENDER_THRESHOLD
    result_payload["high_parallel"] = high_parallel
    if rollout_dir.parent.resolve() != rollout_output_root.resolve():
        print_with_lock(
            print_lock,
            f"[ROLLUP FOUND {task_index}] using fallback rollout folder: {redact_disturbance_scale_tags(rollout_dir)}",
        )

    if args.skip_render:
        return result_payload
    if high_parallel:
        print_with_lock(
            print_lock,
            (
                f"[SEED RENDER SKIP {task_index}] {checkpoint.name}: "
                f"{trajectory_count} trajectories > {HIGH_PARALLEL_RENDER_THRESHOLD}; "
                "keeping total summary table/radar only."
            ),
        )
        return result_payload

    seed_render_cmd = build_render_command(
        python_exe=args.python_exe,
        render_script=render_script,
        rollout_dirs=[rollout_dir],
        extra_render_args=args.extra_render_args,
        model_label=seed_render_label,
        output_dir=seed_figure_dir,
        skip_summary_table=True,
        split_panels=args.split_panels,
    )
    seed_render_cmd_str = " ".join(shlex.quote(part) for part in seed_render_cmd)
    print_with_lock(print_lock, f"[SEED RENDER {task_index}] {seed_render_cmd_str}")
    seed_render_result = subprocess.run(
        seed_render_cmd,
        cwd=str(project_root),
        capture_output=True,
        text=True,
    )
    if seed_render_result.returncode != 0:
        result_payload["failure"] = f"seed render failed with exit code {seed_render_result.returncode}"
        combined_tail = redact_disturbance_scale_tags(
            tail_text("\n".join([seed_render_result.stdout or "", seed_render_result.stderr or ""]))
        )
        if combined_tail:
            print_with_lock(
                print_lock,
                f"[SEED RENDER FAIL {task_index}] {checkpoint.name}\n{combined_tail}",
            )
        return result_payload

    seed_output_image = expected_render_output_path(
        primary_rollout_dir=rollout_dir,
        output_name=str(render_args.output_name),
        model_label=seed_render_label,
        output_dir=seed_figure_dir,
        split_panels=bool(render_args.split_panels),
    )
    if not render_output_exists(
        seed_output_image,
        split_panels=bool(render_args.split_panels),
        split_panel_filenames=split_panel_filenames,
    ):
        result_payload["failure"] = f"seed render output missing: {seed_output_image}"
        print_with_lock(print_lock, f"[WARN] seed render output missing: {seed_output_image}")
        return result_payload

    result_payload["seed_render_path"] = seed_output_image
    print_with_lock(print_lock, f"[SEED FIGURE {task_index}] {seed_output_image}")
    return result_payload


def run_device_queue(
    device: str,
    task_queue: Queue,
    project_root: Path,
    runs_dir: Path,
    render_script: Path,
    split_panel_filenames: Sequence[str],
    rollout_output_root: Path,
    seed_figure_dir: Path,
    runner_log_root: Path,
    args: argparse.Namespace,
    render_args: Optional[argparse.Namespace],
    extra_runner_args: List[str],
    print_lock: threading.Lock,
    failure_event: threading.Event,
) -> List[Dict[str, object]]:
    device_results: List[Dict[str, object]] = []
    while True:
        if args.stop_on_error and failure_event.is_set():
            break
        try:
            task = task_queue.get_nowait()
        except Empty:
            break
        try:
            result_payload = run_single_checkpoint_task(
                task=task,
                device=device,
                project_root=project_root,
                runs_dir=runs_dir,
                render_script=render_script,
                split_panel_filenames=split_panel_filenames,
                rollout_output_root=rollout_output_root,
                seed_figure_dir=seed_figure_dir,
                runner_log_root=runner_log_root,
                args=args,
                render_args=render_args,
                extra_runner_args=extra_runner_args,
                print_lock=print_lock,
            )
            device_results.append(result_payload)
            if args.stop_on_error and not bool(result_payload["passed"]):
                failure_event.set()
                print_with_lock(
                    print_lock,
                    f"[STOP] stop-on-error triggered on {device}; no new tasks will be scheduled.",
                )
        finally:
            task_queue.task_done()
    return device_results


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch test all matching checkpoints under runs/")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Runs directory to scan")
    parser.add_argument(
        "--run5-script",
        type=str,
        default="run5.py",
        help="Path to runner script for PPO/EPPO/PPPO/RPPO checkpoints",
    )
    parser.add_argument(
        "--run-rarl-script",
        type=str,
        default="run_rarl.py",
        help="Path to runner script for RARL/K-RARL checkpoints",
    )
    parser.add_argument(
        "--run-sac-script",
        type=str,
        default="run_sac.py",
        help="Path to runner script for SAC checkpoints",
    )
    parser.add_argument(
        "--run-td3-script",
        type=str,
        default="run_TD3.py",
        help="Path to runner script for TD3 checkpoints",
    )
    parser.add_argument(
        "--render-script",
        type=str,
        default="auto",
        help=(
            "Path to a render entry script. Use 'auto' to select render_moose.py, "
            "render_singlelane.py, render_fixed_circle_iwd.py, or fall back to render5.py."
        ),
    )
    parser.add_argument(
        "--env",
        type=str,
        default="auto",
        help="Environment name passed to selected runner script. Use 'auto' to infer from each checkpoint.",
    )
    parser.add_argument("--car-preset", type=str, default="tesla_model_3")
    parser.add_argument("--mlp-size-last", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--devices",
        nargs="+",
        default=None,
        help="Optional device pool for parallel testing, e.g. --devices cuda:0 cuda:1 cuda:2 cuda:3",
    )
    parser.add_argument(
        "--checkpoint-names",
        type=str,
        nargs="+",
        default=list(DEFAULT_CHECKPOINT_NAMES),
        help=(
            "Checkpoint file names to scan for under runs/**/nn. Only moose.pth, "
            "fixed_circle_iwd.pth, and singlelane.pth are allowed."
        ),
    )
    parser.add_argument(
        "--python-exe",
        type=str,
        default=sys.executable,
        help="Python executable to run selected runner script",
    )
    parser.add_argument("--max-checkpoints", type=int, default=0, help="0 means no limit")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help=(
            "Directory for all batch-test generated files. Defaults to "
            "_batch_test_outputs/batchtest_<timestamp>."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Only print commands, do not execute")
    parser.add_argument("--skip-render", action="store_true", help="Skip auto rendering after each test")
    parser.add_argument(
        "--max-plot-trajectories",
        type=int,
        default=None,
        help="Maximum individual trajectories drawn per render angle panel. Use 0 to draw all.",
    )
    parser.add_argument(
        "--split-panels",
        dest="split_panels",
        action="store_true",
        default=True,
        help="Render trajectory analysis into a folder containing separate panel images.",
    )
    parser.add_argument(
        "--no-split-panels",
        dest="split_panels",
        action="store_false",
        help="Keep trajectory analysis as one combined figure image.",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop immediately if any test process exits with non-zero code",
    )
    parser.add_argument(
        "--extra-run5-args",
        nargs="*",
        default=[],
        help="Extra args appended to each runner command",
    )
    parser.add_argument(
        "--extra-render-args",
        nargs="*",
        default=[],
        help="Extra args appended to each render5.py command",
    )
    args, passthrough_args = parser.parse_known_args()
    extra_runner_args = list(args.extra_run5_args)
    if passthrough_args:
        # Allow users to append runner args directly at the end, e.g. --disturbed.
        extra_runner_args.extend(passthrough_args)
    extra_runner_args = [arg for arg in extra_runner_args if arg != "--"]
    extra_runner_args, dropped_passthrough_args = strip_batch_only_passthrough_args(extra_runner_args)
    if args.max_plot_trajectories is not None:
        args.extra_render_args = [
            *args.extra_render_args,
            "--max-plot-trajectories",
            str(args.max_plot_trajectories),
        ]
    if dropped_passthrough_args:
        print(
            "[WARN] Ignored batch-only args that leaked into runner passthrough: "
            + " ".join(shlex.quote(part) for part in dropped_passthrough_args)
        )
    args.checkpoint_names, ignored_checkpoint_names = filter_allowed_checkpoint_names(args.checkpoint_names)
    if ignored_checkpoint_names:
        print(
            "[WARN] Ignored unsupported checkpoint names: "
            + ", ".join(ignored_checkpoint_names)
        )

    project_root = Path(__file__).resolve().parent
    batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_output_root = resolve_batch_output_root(project_root, args.output_dir, batch_timestamp)
    rollout_output_root = batch_output_root / "rollouts"
    seed_figure_dir = batch_output_root / "seed_figures"
    runner_log_root = batch_output_root / "runner_logs"
    runs_dir = as_abs_path(project_root, args.runs_dir)
    run5_script = as_abs_path(project_root, args.run5_script)
    run_rarl_script = as_abs_path(project_root, args.run_rarl_script)
    run_sac_script = as_abs_path(project_root, args.run_sac_script)
    run_td3_script = as_abs_path(project_root, args.run_td3_script)
    runner_scripts_by_key = {
        "run5": run5_script,
        "run_rarl": run_rarl_script,
        "run_sac": run_sac_script,
        "run_td3": run_td3_script,
    }

    if not runs_dir.exists():
        print(f"[ERROR] runs dir not found: {runs_dir}")
        return 1

    checkpoints = collect_checkpoints(runs_dir, args.checkpoint_names)
    if args.max_checkpoints > 0:
        checkpoints = checkpoints[: args.max_checkpoints]

    render_script: Optional[Path] = None
    render_module = None
    render_args = None
    split_panel_filenames = list(DEFAULT_SPLIT_PANEL_EXPORT_FILENAMES)
    if not args.skip_render:
        render_script, detected_render_env_name, render_detection_reason = resolve_render_script(
            project_root=project_root,
            raw_render_script=args.render_script,
            runs_dir=runs_dir,
            checkpoints=checkpoints,
            requested_env_name=args.env,
        )
        if not render_script.exists():
            print(f"[ERROR] render script not found: {render_script}")
            return 1
        if args.render_script.lower() == "auto":
            if detected_render_env_name is None:
                print(
                    f"[INFO] Auto-selected generic render script: {render_script.name} "
                    "(no unique env hint found)."
                )
            else:
                print(
                    f"[INFO] Auto-selected render script: {render_script.name} "
                    f"(env={detected_render_env_name}, source={render_detection_reason})."
                )
        render_module, render_args = load_render_module_and_args(
            render_script=render_script,
            extra_render_args=args.extra_render_args,
            split_panels=args.split_panels,
        )
        split_panel_filenames = get_render_split_panel_filenames(render_module)

    if not checkpoints:
        print("[INFO] No checkpoints found.")
        print(f"[INFO] runs dir: {runs_dir}")
        print(f"[INFO] names   : {args.checkpoint_names}")
        return 0

    print("=" * 80)
    print(f"Found checkpoints: {len(checkpoints)}")
    print(f"Batch output root: {batch_output_root}")
    print(f"Rollouts: {rollout_output_root}")
    if not args.skip_render:
        print(f"Seed figures: {seed_figure_dir}")
        print(f"Merged figures: {batch_output_root / 'merged_figures'}")
    print(f"Runner logs: {runner_log_root}")
    for idx, ckpt in enumerate(checkpoints, 1):
        print(f"  [{idx:03d}] {redact_disturbance_scale_tags(ckpt)}")
    checkpoint_algorithm_lookup: Dict[Path, str] = {}
    checkpoint_runner_lookup: Dict[Path, str] = {}
    unsupported_checkpoints: List[Path] = []
    grouped_checkpoints: Dict[str, List[Path]] = {}
    for checkpoint in checkpoints:
        algorithm_name = infer_algorithm_name_from_checkpoint(checkpoint, runs_dir)
        if algorithm_name is None:
            unsupported_checkpoints.append(checkpoint)
            continue
        runner_key = resolve_runner_key_from_algorithm_name(algorithm_name)
        if runner_key is None:
            unsupported_checkpoints.append(checkpoint)
            continue
        checkpoint_algorithm_lookup[checkpoint] = algorithm_name
        checkpoint_runner_lookup[checkpoint] = runner_key
        grouped_checkpoints.setdefault(algorithm_name, []).append(checkpoint)

    print("-" * 80)
    print(f"Detected algorithm groups: {len(grouped_checkpoints)}")
    for idx, (group_name, grouped_paths) in enumerate(grouped_checkpoints.items(), 1):
        runner_key = checkpoint_runner_lookup[grouped_paths[0]]
        runner_script_name = runner_scripts_by_key[runner_key].name
        print(
            f"  [G{idx:02d}] {group_name} ({len(grouped_paths)} checkpoints, runner={runner_script_name})"
        )
    if unsupported_checkpoints:
        print(
            f"[WARN] {len(unsupported_checkpoints)} checkpoints are outside supported algorithm folders "
            f"({SUPPORTED_ALGORITHM_FOLDER_HINT})."
        )
    print("=" * 80)

    required_runner_keys = sorted(set(checkpoint_runner_lookup.values()))
    for runner_key in required_runner_keys:
        runner_script = runner_scripts_by_key[runner_key]
        if not runner_script.exists():
            print(f"[ERROR] runner script not found for {runner_key}: {runner_script}")
            return 1

    passed = []
    failed = []
    skipped_due_to_stop = []
    skipped_unsupported_algorithm = []
    seed_render_succeeded = []
    merged_render_succeeded = []
    summary_paths: List[Path] = []
    summary_mean_plot_paths: List[Path] = []
    render_succeeded = []
    render_failed = []
    group_rollout_dirs: Dict[str, List[Path]] = {group_name: [] for group_name in grouped_checkpoints}
    aggregate_dir = None
    aggregate_dir_name = "merged_figures"
    devices = list(dict.fromkeys(args.devices or [args.device]))
    if not devices:
        devices = [args.device]
    print(f"Device pool: {', '.join(devices)}")
    print(f"Parallel workers: {len(devices)}")

    prepared_tasks: List[Dict[str, object]] = []
    prepared_group_counts: Dict[str, int] = defaultdict(int)
    for idx, checkpoint in enumerate(checkpoints, 1):
        model_name = model_run_dir_name_from_checkpoint(checkpoint)
        algorithm_name = checkpoint_algorithm_lookup.get(checkpoint)
        runner_key = checkpoint_runner_lookup.get(checkpoint)
        if algorithm_name is None or runner_key is None:
            print(
                f"\n[SKIP {idx}/{len(checkpoints)}] {checkpoint.name} "
                f"({redact_disturbance_scale_tags(model_name)})"
            )
            print("[WARN] Unsupported algorithm folder under --runs-dir")
            skipped_unsupported_algorithm.append(checkpoint)
            continue
        runner_script = runner_scripts_by_key[runner_key]
        env_name = args.env
        if env_name == "auto":
            env_name = infer_env_name_from_checkpoint(checkpoint)
            if env_name is None:
                print(
                    f"\n[SKIP {idx}/{len(checkpoints)}] {checkpoint.name} "
                    f"({redact_disturbance_scale_tags(model_name)})"
                )
                print("[WARN] Could not infer env name from checkpoint path/name")
                failed.append((checkpoint, "env-inference"))
                continue

        prepared_tasks.append(
            {
                "task_index": len(prepared_tasks) + 1,
                "source_index": idx,
                "checkpoint": checkpoint,
                "model_name": model_name,
                "model_group_name": algorithm_name,
                "algorithm_name": algorithm_name,
                "runner_key": runner_key,
                "runner_script": runner_script,
                "env_name": env_name,
            }
        )
        prepared_group_counts[algorithm_name] += 1

    if prepared_tasks and not args.dry_run:
        rollout_output_root.mkdir(parents=True, exist_ok=True)
        runner_log_root.mkdir(parents=True, exist_ok=True)
        if not args.skip_render:
            seed_figure_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("\n" + "-" * 80)
        print(
            f"[DRY-RUN] {len(prepared_tasks)} runnable checkpoints will be scheduled "
            f"across {len(devices)} worker(s)."
        )
        for task in prepared_tasks:
            checkpoint = Path(task["checkpoint"])
            model_name = str(task["model_name"])
            algorithm_name = str(task["algorithm_name"])
            runner_key = str(task["runner_key"])
            runner_script = Path(task["runner_script"])
            env_name = str(task["env_name"])
            task_index = int(task["task_index"])
            preview_device = devices[(task_index - 1) % len(devices)]
            cmd = build_test_command(
                python_exe=args.python_exe,
                runner_script=runner_script,
                runner_key=runner_key,
                env_name=env_name,
                checkpoint=checkpoint,
                car_preset=args.car_preset,
                mlp_size_last=args.mlp_size_last,
                device=preview_device,
                extra_run5_args=extra_runner_args,
            )
            rollout_dir_name = build_rollout_dir_name(
                checkpoint=checkpoint,
                task_index=task_index,
                device=preview_device,
            )
            cmd_str = format_command_for_log(cmd)
            seed_render_label = build_seed_render_label(
                task_index=task_index,
                algorithm_name=algorithm_name,
                model_name=model_name,
                checkpoint=checkpoint,
            )
            print(
                f"\n[DRY-RUN {task_index}/{len(prepared_tasks)}] {checkpoint.name} "
                f"({redact_disturbance_scale_tags(model_name)}, algo={algorithm_name}, "
                f"env={env_name}, device~={preview_device}, "
                f"runner={runner_script.name})"
            )
            print(cmd_str)
            print(f"[DRY-RUN rollout] {rollout_output_root / rollout_dir_name}")
            print(f"[DRY-RUN runner-log-root] {runner_log_root}")
            if not args.skip_render:
                planned_seed_output = expected_render_output_path(
                    primary_rollout_dir=rollout_output_root / rollout_dir_name,
                    output_name=str(render_args.output_name),
                    model_label=seed_render_label,
                    output_dir=seed_figure_dir,
                    split_panels=bool(render_args.split_panels),
                )
                print(
                    f"[DRY-RUN seed-figure] "
                    f"{planned_seed_output}"
                )
    else:
        if prepared_tasks:
            print("\n" + "-" * 80)
            print(
                f"Launching {len(prepared_tasks)} checkpoint tests with {len(devices)} worker(s): "
                f"{', '.join(devices)}"
            )
        task_queue: Queue = Queue()
        for task in prepared_tasks:
            task_queue.put(task)

        print_lock = threading.Lock()
        failure_event = threading.Event()
        worker_results: List[Dict[str, object]] = []
        with ThreadPoolExecutor(max_workers=len(devices)) as executor:
            futures = [
                executor.submit(
                    run_device_queue,
                    device=device,
                    task_queue=task_queue,
                    project_root=project_root,
                    runs_dir=runs_dir,
                    render_script=render_script,
                    split_panel_filenames=split_panel_filenames,
                    rollout_output_root=rollout_output_root,
                    seed_figure_dir=seed_figure_dir,
                    runner_log_root=runner_log_root,
                    args=args,
                    render_args=render_args,
                    extra_runner_args=extra_runner_args,
                    print_lock=print_lock,
                    failure_event=failure_event,
                )
                for device in devices
            ]
            for future in futures:
                worker_results.extend(future.result())

        if args.stop_on_error and failure_event.is_set():
            while True:
                try:
                    skipped_task = task_queue.get_nowait()
                except Empty:
                    break
                skipped_due_to_stop.append(Path(skipped_task["checkpoint"]))

        worker_results.sort(key=lambda item: int(item["task_index"]))
        for result_payload in worker_results:
            checkpoint = Path(result_payload["checkpoint"])
            if bool(result_payload["passed"]):
                passed.append(checkpoint)
            else:
                failed.append((checkpoint, result_payload["failure"]))
                continue

            rollout_dir_raw = result_payload["rollout_dir"]
            if rollout_dir_raw is not None:
                rollout_dir = Path(rollout_dir_raw)
                model_group_name = str(result_payload["model_group_name"])
                if rollout_dir not in group_rollout_dirs[model_group_name]:
                    group_rollout_dirs[model_group_name].append(rollout_dir)

            seed_render_path_raw = result_payload["seed_render_path"]
            if seed_render_path_raw is not None:
                seed_render_succeeded.append((checkpoint, Path(seed_render_path_raw)))

            if result_payload["failure"] is not None:
                render_failed.append((str(checkpoint), str(result_payload["failure"])))

    if not args.skip_render:
        if args.dry_run:
            print("\n" + "-" * 80)
            print("Planned outputs:")
            for group_name, grouped_paths in grouped_checkpoints.items():
                prepared_count = prepared_group_counts.get(group_name, 0)
                if prepared_count <= 0:
                    continue
                print(
                    f"[DRY-RUN] per-seed figures will be rendered under {seed_figure_dir} for group {group_name}"
                )
                print(
                    f"[DRY-RUN] {group_name} -> "
                    f"{expected_render_output_path( primary_rollout_dir=Path('.'), output_name=render_args.output_name, model_label=image_stem_from_model_group(group_name), output_dir=batch_output_root / aggregate_dir_name, split_panels=bool(render_args.split_panels))} "
                    f"(merge {prepared_count} checkpoints)"
                )
        else:
            render_groups = [
                (group_name, rollout_dirs)
                for group_name, rollout_dirs in group_rollout_dirs.items()
                if rollout_dirs
            ]
            if render_groups:
                aggregate_dir = batch_output_root / aggregate_dir_name
                aggregate_dir.mkdir(parents=True, exist_ok=True)
            for render_idx, (group_name, rollout_dirs) in enumerate(render_groups, 1):
                print(
                    f"\n[RENDER {render_idx}/{len(render_groups)}] "
                    f"{group_name} (merge {len(rollout_dirs)} rollout folders)"
                )
                if has_high_parallel_rollout(rollout_dirs):
                    counts = [rollout_dir_trajectory_count(path) for path in rollout_dirs]
                    print(
                        f"[RENDER SKIP] {group_name}: max trajectories={max(counts) if counts else 0} "
                        f"> {HIGH_PARALLEL_RENDER_THRESHOLD}; keeping total summary table/radar only."
                    )
                    continue
                render_cmd = build_render_command(
                    python_exe=args.python_exe,
                    render_script=render_script,
                    rollout_dirs=rollout_dirs,
                    extra_render_args=args.extra_render_args,
                    model_label=group_name,
                    output_dir=aggregate_dir,
                    skip_summary_table=True,
                    split_panels=args.split_panels,
                    hide_panel_titles=True,
                )
                render_cmd_str = " ".join(shlex.quote(part) for part in render_cmd)
                print(f"[RENDER CMD] {render_cmd_str}")
                render_result = subprocess.run(render_cmd, cwd=str(project_root))

                if render_result.returncode != 0:
                    msg = f"render failed with exit code {render_result.returncode}"
                    print(f"[WARN] {msg}")
                    render_failed.append((group_name, msg))
                    continue

                primary_rollout_dir = rollout_dirs[0]
                merged_image = expected_render_output_path(
                    primary_rollout_dir=primary_rollout_dir,
                    output_name=render_args.output_name,
                    model_label=group_name,
                    output_dir=aggregate_dir,
                    split_panels=bool(render_args.split_panels),
                )
                if not render_output_exists(
                    merged_image,
                    split_panels=bool(render_args.split_panels),
                    split_panel_filenames=split_panel_filenames,
                ):
                    msg = f"render output missing: {merged_image}"
                    print(f"[WARN] {msg}")
                    render_failed.append((group_name, msg))
                    continue

                print(f"[MERGED FIGURE] {merged_image}")
                merged_render_succeeded.append((group_name, merged_image))
                render_succeeded.append((group_name, merged_image))

            if aggregate_dir is not None:
                summary_records = []
                for group_name, rollout_dirs in render_groups:
                    env_names = {
                        inferred_env
                        for inferred_env in (
                            infer_env_name_from_checkpoint(checkpoint)
                            for checkpoint in grouped_checkpoints.get(group_name, [])
                        )
                        if inferred_env is not None
                    }
                    group_env_name = next(iter(env_names)) if len(env_names) == 1 else None
                    model_source = {
                        "label": group_name,
                        "display_name": group_name,
                        "algorithm_name": group_name,
                        "model_dirs": [str(path) for path in rollout_dirs],
                        "primary_dir": str(rollout_dirs[0]),
                        "env_name": group_env_name,
                    }
                    summary = render_module.summarize_model_source(model_source, render_args)
                    if summary is not None:
                        summary["display_name"] = group_name
                        summary["algorithm_name"] = group_name
                        summary_records.append(summary)

                if summary_records:
                    render_module.apply_publication_style()
                    set_titles_visible = getattr(render_module, "set_panel_titles_visible", None)
                    if callable(set_titles_visible):
                        set_titles_visible(False)
                    exported_summary_paths = render_module.export_summary_table_figure(
                        summary_records=summary_records,
                        output_dir=str(aggregate_dir),
                        output_name=render_args.output_name,
                        dpi=render_args.dpi,
                        save_pdf=render_args.save_pdf,
                    )
                    summary_paths = [Path(path) for path in exported_summary_paths]
                    for summary_path in summary_paths:
                        print(f"[TOTAL SUMMARY] {redact_disturbance_scale_tags(summary_path)}")

                    exported_summary_mean_paths = render_module.export_summary_mean_radar_figure(
                        summary_records=summary_records,
                        output_dir=str(aggregate_dir),
                        output_name=render_args.output_name,
                        dpi=render_args.dpi,
                        save_pdf=render_args.save_pdf,
                    )
                    summary_mean_plot_paths = [Path(path) for path in exported_summary_mean_paths]
                    for summary_mean_path in summary_mean_plot_paths:
                        print(f"[TOTAL SUMMARY MEAN] {redact_disturbance_scale_tags(summary_mean_path)}")

    print("\n" + "=" * 80)
    print("Batch test finished")
    print(f"Total   : {len(checkpoints)}")
    print(f"Passed  : {len(passed)}")
    print(f"Failed  : {len(failed)}")
    print(f"Skipped : {len(skipped_due_to_stop)}")
    print(f"Ignored : {len(skipped_unsupported_algorithm)}")
    print(f"Output  : {batch_output_root}")
    if not args.skip_render:
        print(f"SeedFig : {len(seed_render_succeeded)}")
        print(f"Merged  : {len(merged_render_succeeded)}")
        print(f"Summary : {len(summary_paths)}")
        print(f"MeanPlot: {len(summary_mean_plot_paths)}")
        print(f"Rendered: {len(render_succeeded)}")
        print(f"R-Failed: {len(render_failed)}")
        if aggregate_dir is not None:
            print(f"Merge dir: {aggregate_dir}")

    if failed:
        print("-" * 80)
        for path, code in failed:
            print(f"[FAIL code={code}] {redact_disturbance_scale_tags(path)}")
    if skipped_due_to_stop:
        print("-" * 80)
        for path in skipped_due_to_stop:
            print(f"[SKIPPED stop-on-error] {redact_disturbance_scale_tags(path)}")
    if skipped_unsupported_algorithm:
        print("-" * 80)
        for path in skipped_unsupported_algorithm:
            print(f"[SKIPPED unsupported-algorithm-folder] {redact_disturbance_scale_tags(path)}")
    if render_failed:
        print("-" * 80)
        for label, msg in render_failed:
            print(f"[RENDER FAIL] {redact_disturbance_scale_tags(label)} -> {msg}")

    print("=" * 80)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
