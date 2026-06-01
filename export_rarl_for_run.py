import os
from pathlib import Path
from typing import List, Optional, Tuple

import torch


SUPPORTED_ENV_NAMES = ("singlelane", "moose", "fixed_circle_iwd")
EXPORT_SUFFIX = "_pro_for_runpy.pth"
RUNS_DIR_NAMES = (
    "runs",
    "runs_moose",
    "runs_singlelane",
    "runs_fixed_circle_iwd",
)


def is_rarl_run_dir(run_dir: Path) -> bool:
    """Return True for RARL/K-RARL run folders, including grouped layouts."""
    name = run_dir.name.lower()
    path_hint = str(run_dir).lower()
    return name.startswith("rarl_") or "rarl" in path_hint


def iter_candidate_run_dirs(runs_dir: Path) -> List[Path]:
    """Find run directories that contain an nn folder.

    Older runs were stored directly under runs_moose/<run>/nn, while newer
    batch exports can be grouped as runs_moose/K-RARL_dist/<run>/nn.
    """
    candidates = []
    for root, dirs, _files in os.walk(runs_dir):
        if "nn" not in dirs:
            continue

        run_dir = Path(root)
        if is_rarl_run_dir(run_dir):
            candidates.append(run_dir)

    return sorted(candidates)


def convert_rarl_checkpoint(ckpt_path: Path, out_path: Path) -> bool:
    """Convert one RARL checkpoint into a run5.py-compatible checkpoint."""
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")

        if "pro_model" not in ckpt:
            print(f"  [SKIP] {ckpt_path}: not a RARL checkpoint with pro_model")
            return False

        new_ckpt = {
            "model": ckpt["pro_model"],
        }

        if "pro_running_mean_std" in ckpt:
            new_ckpt["running_mean_std"] = ckpt["pro_running_mean_std"]

        if "env_state" in ckpt:
            new_ckpt["env_state"] = ckpt["env_state"]

        os.makedirs(out_path.parent, exist_ok=True)
        torch.save(new_ckpt, out_path)

        print(f"  [OK] converted: {out_path}")
        return True
    except Exception as exc:
        print(f"  [ERR] failed to convert {ckpt_path}: {exc}")
        return False


def infer_env_name(run_dir: Path, nn_dir: Path) -> Optional[str]:
    for env_name in SUPPORTED_ENV_NAMES:
        if (nn_dir / f"{env_name}.pth").exists():
            return env_name

    run_dir_name = run_dir.name.lower()
    for env_name in SUPPORTED_ENV_NAMES:
        if env_name in run_dir_name:
            return env_name

    return None


def find_named_env_checkpoint(nn_dir: Path, env_name: Optional[str]) -> Optional[Path]:
    if env_name is None:
        return None

    ckpt_path = nn_dir / f"{env_name}.pth"
    if ckpt_path.exists():
        return ckpt_path
    return None


def build_conversion_tasks(run_dir: Path, nn_dir: Path) -> List[Tuple[Path, Path]]:
    env_name = infer_env_name(run_dir, nn_dir)
    tasks = []

    env_ckpt = find_named_env_checkpoint(nn_dir, env_name)
    if env_ckpt is not None and env_name is not None:
        tasks.append((env_ckpt, nn_dir / f"{env_name}{EXPORT_SUFFIX}"))

    return tasks


def main() -> None:
    runs_dirs = [Path(name) for name in RUNS_DIR_NAMES]
    existing_runs_dirs = [runs_dir for runs_dir in runs_dirs if runs_dir.exists()]

    if not existing_runs_dirs:
        print("Error: no runs directories exist")
        print("Checked:", ", ".join(str(path) for path in runs_dirs))
        return

    converted_count = 0
    skipped_count = 0
    error_count = 0

    for runs_dir in existing_runs_dirs:
        print(f"[SCAN] {runs_dir}")
        for run_dir in iter_candidate_run_dirs(runs_dir):
            nn_dir = run_dir / "nn"
            if not nn_dir.exists():
                continue

            tasks = build_conversion_tasks(run_dir, nn_dir)
            if not tasks:
                continue

            for ckpt_path, out_path in tasks:
                if out_path.exists():
                    print(f"[SKIP] {runs_dir / run_dir.name}: already exists -> {out_path.name}")
                    skipped_count += 1
                    continue

                print(f"[PROCESS] {runs_dir / run_dir.name}: {ckpt_path.name} -> {out_path.name}")
                success = convert_rarl_checkpoint(ckpt_path, out_path)

                if success:
                    converted_count += 1
                else:
                    error_count += 1

    print("\n" + "=" * 60)
    print("Conversion finished")
    print(f"  scanned   : {len(existing_runs_dirs)} runs roots")
    print(f"  converted : {converted_count}")
    print(f"  skipped   : {skipped_count}")
    print(f"  failed    : {error_count}")
    print("=" * 60)


if __name__ == "__main__":
    main()
