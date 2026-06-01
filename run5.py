import argparse
import glob
import os
import pprint
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime

import yaml
import torch
from icecream import ic

file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from utils.rlgame_utils import RLGPUAlgoObserver, RLGPUEnv
from utils.disturbance_utils import (
    build_disturbance_range_from_args,
    build_rpi_q_disturbance_scale_vector_from_args,
    format_disturbance_name_tag,
    redact_disturbance_scale_tags,
)
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
from envs.eight_drift_iwd import EightDriftIWDEnv
from envs.fixed_circle_iwd import FixedCircleIWDEnv
from envs.lane import LaneChangeStabilityIWDEnv
from envs.moose import MooseIWDEnv
from envs.singlelane import SingleLaneChangeStabilityIWDEnv
from envs.state_tracker_iwd import StateTrackerIWDEnv


@contextmanager
def suppress_stdout_stderr():
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull), redirect_stdout(fnull):
            yield


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


@contextmanager
def redirect_output_to_log(log_file_path):
    os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    with open(log_file_path, "a", encoding="utf-8", buffering=1) as log_file:
        log_file.write(f"\n===== session start: {datetime.now().isoformat()} =====\n")
        with redirect_stdout(TeeStream(sys.stdout, log_file)), redirect_stderr(TeeStream(sys.stderr, log_file)):
            yield
        log_file.write(f"\n===== session end: {datetime.now().isoformat()} =====\n")


def _format_name_number(value):
    numeric = float(value)
    rounded = round(numeric)
    if abs(numeric - rounded) < 1e-6:
        return str(int(rounded))
    return f"{numeric:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _get_disturbance_name_tag():
    return format_disturbance_name_tag(args, prefix="distured")


def _build_disturbance_range_from_args():
    return build_disturbance_range_from_args(args)


def _build_rpi_q_disturbance_scale_vector_from_args():
    return build_rpi_q_disturbance_scale_vector_from_args(args)


def _get_default_rpi_q_dynamic_obs_indices():
    # Align robust dimensions with the state components most affected by dynamics disturbances.
    if args.env in {"moose", "singlelane", "lane", "state_tracker_iwd"}:
        return [8, 9, 10, 11, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    if args.env in {"fixed_circle_iwd", "continuous_drift_iwd", "eight_drift_iwd"}:
        return [2, 3, 4, 11, 12, 13, 14, 15, 16, 17, 18]
    return None


def _resolve_rpi_q_robust_mode():
    mode = str(args.rpi_q_robust_mode).lower()
    if mode == "auto":
        return "dynamics" if args.disturbed else "obs"
    return mode


def _get_rpi_q_name_tag():
    # Always include this flag in run naming so rpi-q and baseline outputs do not mix.
    return f"_rpiq_{int(bool(args.use_rpi_q))}"


RUN_TIMESTAMP = datetime.now().strftime("%m%d_%H%M%S")
RUNNER_CONFIG_FILES = {
    "sac": "sac_config.yaml",
    "rarl_continuous": "rarl_config.yaml",
}
_CHECKPOINT_METADATA_CACHE = {}


def _build_experiment_name_prefix(algo_name):
    return f"{algo_name}_{args.env}_{args.num_parallel}" + _get_rpi_q_name_tag() + _get_disturbance_name_tag()


def _build_legacy_experiment_name(algo_name):
    return (
        f"{algo_name}_{args.env}_{args.seed}_{args.num_parallel}" + _get_rpi_q_name_tag() + _get_disturbance_name_tag()
    )


def _build_experiment_name_prefix_without_rpi(algo_name):
    # Backward compatibility for legacy runs created before the rpi-q flag was added to naming.
    return f"{algo_name}_{args.env}_{args.num_parallel}" + _get_disturbance_name_tag()


def _build_legacy_experiment_name_without_rpi(algo_name):
    return f"{algo_name}_{args.env}_{args.seed}_{args.num_parallel}" + _get_disturbance_name_tag()


def get_run_root_name():
    env_run_roots = {
        "fixed_circle_iwd": "runs_fixed_circle_iwd",
        "moose": "runs_moose",
        "singlelane": "runs_singlelane",
    }
    return env_run_roots.get(args.env, "runs")


def get_run_root_dir():
    env_run_root = os.environ.get("XCAR_RUN_ROOT_DIR")
    if env_run_root:
        return os.path.abspath(os.path.expanduser(env_run_root))
    return os.path.join(file_path, get_run_root_name())


def build_full_experiment_name(algo_name, add_timestamp=False):
    name_prefix = _build_experiment_name_prefix(algo_name)
    if add_timestamp:
        return f"{name_prefix}_{RUN_TIMESTAMP}_{args.seed}"
    return f"{name_prefix}_{args.seed}"


def _get_primary_checkpoint_path():
    if not args.checkpoint:
        return None
    return args.checkpoint[0] if isinstance(args.checkpoint, list) else args.checkpoint


def _load_checkpoint_metadata(checkpoint_path):
    if not checkpoint_path:
        return None

    normalized_path = os.path.abspath(checkpoint_path)
    if normalized_path in _CHECKPOINT_METADATA_CACHE:
        return _CHECKPOINT_METADATA_CACHE[normalized_path]

    try:
        checkpoint = torch.load(normalized_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: failed to inspect checkpoint metadata for {normalized_path}: {exc}")
        _CHECKPOINT_METADATA_CACHE[normalized_path] = None
        return None

    metadata = checkpoint if isinstance(checkpoint, dict) else None
    _CHECKPOINT_METADATA_CACHE[normalized_path] = metadata
    return metadata


def infer_algo_name_from_checkpoint(checkpoint_path):
    metadata = _load_checkpoint_metadata(checkpoint_path)
    if not metadata:
        return None

    metadata_keys = set(metadata.keys())
    if {"actor", "critic", "critic_target"}.issubset(metadata_keys):
        return "sac"
    if "pro_model" in metadata_keys:
        return "rarl_continuous"
    if "model" in metadata_keys:
        return "a2c_continuous"
    return None


def infer_experiment_name_from_checkpoint(checkpoint_path):
    if not checkpoint_path:
        return None

    normalized_path = os.path.abspath(checkpoint_path)
    nn_dir = os.path.dirname(normalized_path)
    if os.path.basename(nn_dir) != "nn":
        return None

    experiment_dir = os.path.dirname(nn_dir)
    experiment_name = os.path.basename(experiment_dir)
    return experiment_name or None


def get_effective_algo_name(default_algo_name):
    checkpoint_path = _get_primary_checkpoint_path()
    if not checkpoint_path:
        return default_algo_name

    inferred_algo_name = infer_algo_name_from_checkpoint(checkpoint_path)
    if inferred_algo_name:
        return inferred_algo_name
    return default_algo_name


def _collect_matching_experiment_dirs(run_root, exact_name, wildcard_pattern):
    candidate_dirs = []

    exact_dir = os.path.join(run_root, exact_name)
    if os.path.isdir(exact_dir):
        candidate_dirs.append(exact_dir)

    wildcard_candidates = glob.glob(os.path.join(run_root, wildcard_pattern))
    candidate_dirs.extend(d for d in wildcard_candidates if os.path.isdir(d))
    return candidate_dirs


def get_latest_experiment_name(algo_name):
    run_root = get_run_root_dir()
    candidate_dirs = []

    current_name = build_full_experiment_name(algo_name, add_timestamp=False)
    current_prefix = _build_experiment_name_prefix(algo_name)
    candidate_dirs.extend(
        _collect_matching_experiment_dirs(
            run_root,
            current_name,
            f"{current_prefix}_*_{args.seed}",
        )
    )

    legacy_name = _build_legacy_experiment_name(algo_name)
    candidate_dirs.extend(_collect_matching_experiment_dirs(run_root, legacy_name, f"{legacy_name}_*"))

    # Also match old naming convention (without explicit rpi-q tag) so resume keeps working.
    compatibility_prefix = _build_experiment_name_prefix_without_rpi(algo_name)
    candidate_dirs.extend(
        _collect_matching_experiment_dirs(
            run_root,
            f"{compatibility_prefix}_{args.seed}",
            f"{compatibility_prefix}_*_{args.seed}",
        )
    )

    compatibility_legacy = _build_legacy_experiment_name_without_rpi(algo_name)
    candidate_dirs.extend(
        _collect_matching_experiment_dirs(run_root, compatibility_legacy, f"{compatibility_legacy}_*")
    )

    if not candidate_dirs:
        return current_name
    return os.path.basename(max(candidate_dirs, key=os.path.getctime))


def resolve_experiment_name(algo_name):
    if args.train_or_test == "train":
        if args.resume and not args.checkpoint:
            return get_latest_experiment_name(algo_name)
        return build_full_experiment_name(algo_name, add_timestamp=True)

    if args.checkpoint:
        inferred_name = infer_experiment_name_from_checkpoint(_get_primary_checkpoint_path())
        if inferred_name:
            return inferred_name
        return build_full_experiment_name(algo_name, add_timestamp=False)
    return get_latest_experiment_name(algo_name)


def load_runner_config(algo_name=None):
    config_name = RUNNER_CONFIG_FILES.get(algo_name, "runner_config.yaml")
    with open(os.path.join(file_path, config_name), encoding="utf-8") as f:
        return yaml.safe_load(f)


ic.configureOutput(argToStringFunction=lambda x: pprint.pformat(x, sort_dicts=False))

parser = argparse.ArgumentParser()
parser.add_argument("train_or_test", type=str, help="Train or test")
parser.add_argument("env", type=str)
parser.add_argument("--car-preset", type=str, default="tesla_model_3")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--rnn", action="store_true")
parser.add_argument("-disturbed", "--disturbed", "--distured", dest="disturbed", action="store_true")
parser.add_argument("--disturbance-rho", type=float, default=0.50, help="AR(1) coefficient for random disturbance")
parser.add_argument("--disturbance-every", type=int, default=1, help="Refresh random disturbance every N steps")
parser.add_argument(
    "--disturbance-scale",
    type=float,
    default=1.0,
    help="Global multiplier applied to all --w-* disturbance bounds before injecting them into the environment",
)
parser.add_argument(
    "--disturbance-scales",
    type=float,
    nargs=15,
    default=None,
    metavar="S",
    help=(
        "15 per-dimension multipliers in layout "
        "fx1 fx2 fx3 fx4 fy1 fy2 fy3 fy4 M_phi M_theta m K21 C21 K23 C23"
    ),
)
parser.add_argument("--w-force", type=float, default=2500.0, help="Uniform disturbance abs bound for tyre force dims (N)")
parser.add_argument("--w-phi", type=float, default=25000.0, help="Uniform disturbance abs bound for roll moment (N·m)")
parser.add_argument("--w-theta", type=float, default=22000.0, help="Uniform disturbance abs bound for pitch moment (N·m)")
parser.add_argument("--w-m", type=float, default=0.0, help="Uniform disturbance abs bound for mass offset (kg)")
parser.add_argument("--w-k21", type=float, default=0.0, help="Uniform disturbance abs bound for K21 offset (N/m)")
parser.add_argument("--w-c21", type=float, default=0.0, help="Uniform disturbance abs bound for C21 offset (N·s/m)")
parser.add_argument("--w-k23", type=float, default=0.0, help="Uniform disturbance abs bound for K23 offset (N/m)")
parser.add_argument("--w-c23", type=float, default=0.0, help="Uniform disturbance abs bound for C23 offset (N·s/m)")
parser.add_argument("--randomize-tyre", nargs="?", const="big", default=None, type=str)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--exp-name", type=str, default="default")
parser.add_argument("--epochs", type=int, default=1500)
parser.add_argument("--num-parallel", type=int, default=30000)
parser.add_argument("--lr-schedule", type=str, default="linear")
parser.add_argument("--mini-epochs", type=int, default=5)
parser.add_argument("--mlp-size-last", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--score-to-win", type=int, default=20000)
parser.add_argument("--save-freq", type=int, default=50)
parser.add_argument("--epoch-index", type=int, default=-1, help="For test only, -1 for using latest")
parser.add_argument(
    "--checkpoint",
    type=str,
    nargs="+",
    default=None,
    help="Direct path(s) to checkpoint file(s) (for test or resume training). Can specify multiple paths.",
)
parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
parser.add_argument("--latent-size", type=int, default=8)
parser.add_argument("--quiet", action="store_true")
parser.add_argument("--aux-reward-decay-steps", type=int, default=0)
parser.add_argument("--aux-reward-coef", type=float, default=1.0)
parser.add_argument("--env-variant", type=str, default="")
parser.add_argument("--ref-mode", type=str, default="hybrid")
parser.add_argument("--obs-noise", action="store_true", help="Add 5% dynamic noise to observations")
parser.add_argument("--use-rpi-q", action="store_true", help="Enable robust Q critic target (RPI-Q style)")
parser.add_argument("--rpi-q-scale-ucc", type=float, default=1.0, help="Scale for robust Q uncertainty correction term")
parser.add_argument("--rpi-q-obs-radius", type=float, default=0.01, help="State perturbation radius used in robust Q target")
parser.add_argument(
    "--rpi-q-robust-mode",
    type=str,
    default="auto",
    choices=["auto", "obs", "dynamics"],
    help="Robust UCC mode. auto=dynamics when --disturbed else obs.",
)
parser.add_argument(
    "--rpi-q-dyn-obs-indices",
    type=int,
    nargs="+",
    default=None,
    help="Optional observation indices for dynamics-aware robust UCC (overrides defaults by env).",
)
parser.add_argument(
    "--rpi-q-dyn-obs-weights",
    type=float,
    nargs="+",
    default=None,
    help="Optional weights aligned with --rpi-q-dyn-obs-indices.",
)
parser.add_argument(
    "--rpi-q-disable-auto-scale-ucc",
    action="store_true",
    help="Disable adaptive UCC scaling based on current Q/UCC magnitudes.",
)
args = parser.parse_args()

if args.train_or_test == "test":
    args.randomize_tyre = None


def get_num_parallel():
    if args.train_or_test == "train":
        return args.num_parallel
    return 20


def get_env_creator():
    return {
        "fixed_circle_iwd": lambda **kwargs: FixedCircleIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
        "eight_drift_iwd": lambda **kwargs: EightDriftIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
        "state_tracker_iwd": lambda **kwargs: StateTrackerIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
        "continuous_drift_iwd": lambda **kwargs: ContinuousDriftIWDEnv(
            args.car_preset, get_num_parallel(), args.device, **kwargs
        ),
        "singlelane": lambda **kwargs: SingleLaneChangeStabilityIWDEnv(
            args.car_preset, get_num_parallel(), args.device, **kwargs
        ),
        "lane": lambda **kwargs: LaneChangeStabilityIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
        "moose": lambda **kwargs: MooseIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    }


def get_default_env_config():
    return {
        "dt": args.dt,
        "disturbance_param": (args.disturbance_rho, max(1, args.disturbance_every)) if args.disturbed else None,
        "disturbance_range": _build_disturbance_range_from_args() if args.disturbed else None,
        "randomize_param": {
            None: {},
            "big": {
                "B": [8.3, 11.7],
                "C": [1.45, 2.38],
                "D": [0.75, 1.25],
                "m": [1612.0, 1930.0],
                "k_phi": [10350.0, 14650.0],
                "c_phi": [1040.0, 1460.0],
                "k_theta": [8800.0, 11500.0],
                "c_theta": [1610.0, 2090.0],
            },
            "small": {
                "B": [9.0, 11.0],
                "C": [1.6, 2.2],
                "D": [0.9, 1.1],
                "m": [1672.0, 1848.0],
                "k_phi": [11350.0, 14650.0],
                "c_phi": [1140.0, 1460.0],
                "k_theta": [10000.0, 11500.0],
                "c_theta": [1710.0, 1990.0],
            },
            "xuanjia": {
                "k_phi": [10350.0, 15650.0],
                "c_phi": [1040.0, 1560.0],
                "k_theta": [8500.0, 12500.0],
                "c_theta": [1710.0, 2190.0],
            },
        }[args.randomize_tyre],
        "random_seed": args.seed if args.train_or_test == "train" else None,
        "quiet": args.quiet,
        "aux_reward_decay_steps": args.aux_reward_decay_steps,
        "aux_reward_coef": args.aux_reward_coef,
        "gamma": args.gamma,
        "train": args.train_or_test == "train",
        "obs_noise": args.obs_noise,
    }


def setup_env_config(default_env_config):
    if args.env != "state_tracker_iwd":
        return

    if args.train_or_test == "train":
        default_env_config["ref_mode"] = "hybrid"
        return

    env_variant_to_ref_mode = {
        "hybrid": "hybrid",
        "fixed": "fixed_circle_ccw",
        "fixed_cw": "fixed_circle_cw",
        "eight": "eight_drift",
    }
    if args.env_variant in env_variant_to_ref_mode:
        default_env_config["ref_mode"] = env_variant_to_ref_mode[args.env_variant]
    else:
        raise ValueError(f"Unknown env variant {args.env_variant}")


def create_and_load_runner(full_experiment_name, algo_name=None):
    default_env_config = get_default_env_config()
    setup_env_config(default_env_config)

    envs = get_env_creator()
    blacklist_keys = lambda d, blacklist: {k: d[k] for k in d if k not in blacklist}

    vecenv.register("RLGPU", lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
    env_configurations.register(
        "rlgpu",
        {
            "vecenv_type": "RLGPU",
            "env_creator": lambda **env_config: envs[args.env](
                **blacklist_keys(default_env_config, env_config.keys()),
                **env_config,
            ),
        },
    )

    runner = Runner(RLGPUAlgoObserver())
    runner_config = load_runner_config(algo_name=algo_name)

    default_env_config["full_experiment_name"] = full_experiment_name
    if os.environ.get("XCAR_ROLLOUT_ROOT"):
        default_env_config["rollout_root_dir"] = os.environ["XCAR_ROLLOUT_ROOT"]
    runner_config["params"]["seed"] = args.seed
    runner_config["params"]["config"]["num_actors"] = args.num_parallel
    runner_config["params"]["config"]["max_epochs"] = args.epochs
    runner_config["params"]["config"]["minibatch_size"] = args.num_parallel
    runner_config["params"]["config"]["games_to_track"] = args.num_parallel
    runner_config["params"]["config"]["mini_epochs"] = args.mini_epochs
    runner_config["params"]["config"]["lr_schedule"] = args.lr_schedule
    runner_config["params"]["config"]["gamma"] = args.gamma
    runner_config["params"]["config"]["horizon_length"] = args.horizon
    runner_config["params"]["config"]["score_to_win"] = args.score_to_win
    runner_config["params"]["config"]["name"] = args.env
    runner_config["params"]["config"]["full_experiment_name"] = full_experiment_name
    runner_config["params"]["config"]["train_dir"] = get_run_root_dir()
    runner_config["params"]["network"]["mlp"]["units"] = [args.mlp_size_last * i for i in (4, 2, 1)]
    runner_config["params"]["config"]["save_frequency"] = args.save_freq
    runner_config["params"]["config"]["device_name"] = args.device
    runner_config["params"]["config"]["device"] = args.device
    runner_config["params"]["config"]["use_rpi_q"] = args.use_rpi_q
    runner_config["params"]["config"]["rpi_q_scale_ucc"] = args.rpi_q_scale_ucc
    runner_config["params"]["config"]["rpi_q_obs_radius"] = args.rpi_q_obs_radius
    runner_config["params"]["config"]["rpi_q_robust_mode"] = _resolve_rpi_q_robust_mode()
    runner_config["params"]["config"]["rpi_q_normalize_obs"] = True
    runner_config["params"]["config"]["rpi_q_auto_scale_ucc"] = not args.rpi_q_disable_auto_scale_ucc
    dyn_indices = args.rpi_q_dyn_obs_indices
    if dyn_indices is None and runner_config["params"]["config"]["rpi_q_robust_mode"] == "dynamics":
        dyn_indices = _get_default_rpi_q_dynamic_obs_indices()
    if dyn_indices is not None:
        runner_config["params"]["config"]["rpi_q_dynamic_obs_indices"] = list(dyn_indices)
    if args.rpi_q_dyn_obs_weights is not None:
        runner_config["params"]["config"]["rpi_q_dynamic_obs_weights"] = list(args.rpi_q_dyn_obs_weights)
    if args.disturbed:
        runner_config["params"]["config"]["rpi_q_disturbance_scale_vector"] = (
            _build_rpi_q_disturbance_scale_vector_from_args()
        )
    if not args.rnn:
        runner_config["params"]["network"].pop("rnn", None)

    if args.quiet:
        with suppress_stdout_stderr():
            runner.load(runner_config)
    else:
        runner.load(runner_config)

    return runner, full_experiment_name


if __name__ == "__main__":
    default_algo_name = load_runner_config()["params"]["algo"]["name"]
    effective_algo_name = get_effective_algo_name(default_algo_name)
    final_experiment_name = resolve_experiment_name(effective_algo_name)
    run_dir = os.path.join(get_run_root_dir(), final_experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "run.log")

    with redirect_output_to_log(log_file_path):
        if effective_algo_name != default_algo_name:
            print(f"Detected checkpoint algorithm: {effective_algo_name} (default runner config: {default_algo_name})")

        if args.train_or_test == "train":
            runner, full_experiment_name = create_and_load_runner(final_experiment_name, algo_name=effective_algo_name)
            train_config = {"train": True}

            if args.resume or args.checkpoint:
                if args.checkpoint:
                    checkpoint_name = args.checkpoint[0] if isinstance(args.checkpoint, list) else args.checkpoint
                else:
                    checkpoint_dir = os.path.join(get_run_root_dir(), full_experiment_name, "nn")
                    checkpoint_name = os.path.join(checkpoint_dir, f"{args.env}.pth")

                print(f"Resuming training from checkpoint: {redact_disturbance_scale_tags(checkpoint_name)}")
                train_config["checkpoint"] = checkpoint_name

            runner.run(train_config)

        elif args.train_or_test == "test":
            checkpoint_list = []

            if args.checkpoint:
                checkpoint_list = args.checkpoint if isinstance(args.checkpoint, list) else [args.checkpoint]
            else:
                runner, full_experiment_name = create_and_load_runner(final_experiment_name)
                checkpoint_dir = os.path.join(get_run_root_dir(), full_experiment_name, "nn")
                if args.epoch_index == -1:
                    checkpoint_list = [os.path.join(checkpoint_dir, f"{args.env}.pth")]
                else:
                    list_of_files = glob.glob(
                        os.path.join(checkpoint_dir, f"last_{args.env}_ep_{args.epoch_index}_rew_*.pth")
                    )
                    checkpoint_list = [max(list_of_files, key=os.path.getctime)]

            print(f"\n{'=' * 80}")
            print(f"Total checkpoints to test: {len(checkpoint_list)}")
            print(f"{'=' * 80}\n")

            for idx, checkpoint_name in enumerate(checkpoint_list, 1):
                print(f"\n{'=' * 80}")
                print(
                    f"[{idx}/{len(checkpoint_list)}] Testing checkpoint: "
                    f"{redact_disturbance_scale_tags(checkpoint_name)}"
                )
                print(f"{'=' * 80}\n")

                if not os.path.exists(checkpoint_name):
                    print(
                        "Warning: checkpoint file not found, skipping: "
                        f"{redact_disturbance_scale_tags(checkpoint_name)}\n"
                    )
                    continue

                try:
                    checkpoint_algo_name = infer_algo_name_from_checkpoint(checkpoint_name) or effective_algo_name
                    checkpoint_experiment_name = (
                        infer_experiment_name_from_checkpoint(checkpoint_name) or final_experiment_name
                    )
                    if checkpoint_algo_name != effective_algo_name:
                        print(f"Detected checkpoint algorithm override: {checkpoint_algo_name}")

                    runner, full_experiment_name = create_and_load_runner(
                        checkpoint_experiment_name,
                        algo_name=checkpoint_algo_name,
                    )
                    try:
                        runner.run({"train": False, "play": True, "checkpoint": checkpoint_name})
                    except SystemExit as e:
                        print(f"Caught SystemExit: {e}")

                    print(
                        f"\nCheckpoint [{idx}/{len(checkpoint_list)}] test finished: "
                        f"{redact_disturbance_scale_tags(checkpoint_name)}\n"
                    )
                except KeyboardInterrupt:
                    print("\nUser interrupted, stop testing.")
                    break
                except Exception as e:
                    print(
                        f"\nError: failed while testing "
                        f"{redact_disturbance_scale_tags(checkpoint_name)}: "
                        f"{redact_disturbance_scale_tags(str(e))}\n"
                    )
                    traceback.print_exc()

            print(f"\n{'=' * 80}")
            print(f"All tests complete. Tested {len(checkpoint_list)} checkpoints.")
            print(f"{'=' * 80}\n")
