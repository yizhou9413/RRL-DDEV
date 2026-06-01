import argparse
import glob
import os
import pprint
import sys
import traceback
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime

import yaml
try:
    from icecream import ic
except Exception:  # pragma: no cover - optional debug dependency
    def ic(*args, **kwargs):
        return args[0] if len(args) == 1 else args

file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from utils.rlgame_utils import RLGPUAlgoObserver, RLGPUEnv
from utils.disturbance_utils import (
    build_disturbance_range_from_args,
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
    return format_disturbance_name_tag(args, prefix="disturbed")


def _build_disturbance_range_from_args():
    return build_disturbance_range_from_args(args)


RUN_TIMESTAMP = datetime.now().strftime("%m%d_%H%M%S")


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


def _build_experiment_name_prefix(algo_name):
    prefix = f"{algo_name}_{args.env}_{args.num_parallel}"
    if args.exp_name and args.exp_name != "default":
        prefix = f"{prefix}_{args.exp_name}"
    return prefix + _get_disturbance_name_tag()


def build_full_experiment_name(algo_name, add_timestamp=False):
    name_prefix = _build_experiment_name_prefix(algo_name)
    if add_timestamp:
        return f"{name_prefix}_{RUN_TIMESTAMP}_{args.seed}"
    return f"{name_prefix}_{args.seed}"


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

    if not candidate_dirs:
        return current_name
    return os.path.basename(max(candidate_dirs, key=os.path.getctime))


def resolve_experiment_name(algo_name):
    if args.train_or_test == "train":
        if args.resume and not args.checkpoint:
            return get_latest_experiment_name(algo_name)
        return build_full_experiment_name(algo_name, add_timestamp=True)

    if args.checkpoint:
        return build_full_experiment_name(algo_name, add_timestamp=False)
    return get_latest_experiment_name(algo_name)


def load_runner_config():
    with open(os.path.join(file_path, "td3_config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


if hasattr(ic, "configureOutput"):
    ic.configureOutput(argToStringFunction=lambda x: pprint.pformat(x, sort_dicts=False))

parser = argparse.ArgumentParser()
parser.add_argument("train_or_test", type=str, help="Train or test")
parser.add_argument("env", type=str)
parser.add_argument("--car-preset", type=str, default="tesla_model_3")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--rnn", action="store_true")
parser.add_argument("--disturbed", "--distured", dest="disturbed", action="store_true")
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
parser.add_argument("--w-phi", type=float, default=25000.0, help="Uniform disturbance abs bound for roll moment (N*m)")
parser.add_argument("--w-theta", type=float, default=22000.0, help="Uniform disturbance abs bound for pitch moment (N*m)")
parser.add_argument("--w-m", type=float, default=0.0, help="Uniform disturbance abs bound for mass offset (kg)")
parser.add_argument("--w-k21", type=float, default=0.0, help="Uniform disturbance abs bound for K21 offset (N/m)")
parser.add_argument("--w-c21", type=float, default=0.0, help="Uniform disturbance abs bound for C21 offset (N*s/m)")
parser.add_argument("--w-k23", type=float, default=0.0, help="Uniform disturbance abs bound for K23 offset (N/m)")
parser.add_argument("--w-c23", type=float, default=0.0, help="Uniform disturbance abs bound for C23 offset (N*s/m)")
parser.add_argument("--randomize-tyre", nargs="?", const="big", default=None, type=str)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--exp-name", type=str, default="default")
parser.add_argument("--epochs", type=int, default=10000)
parser.add_argument("--num-parallel", type=int, default=30000)
parser.add_argument("--lr-schedule", type=str, default="linear")
parser.add_argument("--mini-epochs", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--num-steps-per-episode", type=int, default=None)
parser.add_argument("--max-env-steps", type=int, default=500)
parser.add_argument("--batch-size", type=int, default=8192)
parser.add_argument("--replay-buffer-size", type=int, default=15000000)
parser.add_argument("--num-warmup-steps", type=int, default=10000000)
parser.add_argument("--warmup-mode", type=str, default="steps", choices=["epochs", "steps"])
parser.add_argument("--updates-per-step", type=int, default=4)
parser.add_argument("--actor-lr", type=float, default=3e-4)
parser.add_argument("--critic-lr", type=float, default=3e-4)
parser.add_argument("--actor-update-interval", type=int, default=2)
parser.add_argument("--target-update-interval", type=int, default=2)
parser.add_argument("--critic-tau", type=float, default=0.005)
parser.add_argument("--exploration-noise", type=float, default=0.1)
parser.add_argument("--policy-noise", type=float, default=0.2)
parser.add_argument("--noise-clip", type=float, default=0.5)
parser.add_argument("--grad-norm", type=float, default=0.5)
parser.add_argument("--reward-scale", type=float, default=0.1)
parser.add_argument("--save-freq", type=int, default=50)
parser.add_argument("--save-best-after", type=int, default=20)
parser.add_argument("--score-to-win", type=int, default=20000)
parser.add_argument("--games-to-track", type=int, default=30000)
parser.add_argument("--mlp-size-last", type=int, default=64)
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
args = parser.parse_args()

if args.train_or_test == "test":
    args.randomize_tyre = None

if args.num_steps_per_episode is None:
    args.num_steps_per_episode = args.horizon

if args.updates_per_step is None:
    args.updates_per_step = max(2, min(4, args.mini_epochs))

if args.batch_size is None:
    args.batch_size = max(512, min(args.num_parallel, 1024))


def get_num_parallel():
    if args.train_or_test == "train":
        return args.num_parallel
    return 10


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


def create_and_load_runner(full_experiment_name):
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
    runner_config = load_runner_config()

    default_env_config["full_experiment_name"] = full_experiment_name
    if os.environ.get("XCAR_ROLLOUT_ROOT"):
        default_env_config["rollout_root_dir"] = os.environ["XCAR_ROLLOUT_ROOT"]

    runner_config["params"]["seed"] = args.seed
    runner_config["params"]["config"]["num_actors"] = args.num_parallel
    runner_config["params"]["config"]["max_epochs"] = args.epochs
    runner_config["params"]["config"]["gamma"] = args.gamma
    runner_config["params"]["config"]["lr_schedule"] = args.lr_schedule
    runner_config["params"]["config"]["mini_epochs"] = args.mini_epochs
    runner_config["params"]["config"]["horizon_length"] = args.horizon
    runner_config["params"]["config"]["name"] = args.env
    runner_config["params"]["config"]["full_experiment_name"] = full_experiment_name
    runner_config["params"]["config"]["train_dir"] = get_run_root_dir()
    runner_config["params"]["config"]["save_frequency"] = args.save_freq
    runner_config["params"]["config"]["save_best_after"] = args.save_best_after
    runner_config["params"]["config"]["score_to_win"] = args.score_to_win
    runner_config["params"]["config"]["games_to_track"] = args.games_to_track
    runner_config["params"]["config"]["device_name"] = args.device
    runner_config["params"]["config"]["device"] = args.device
    runner_config["params"]["config"]["num_steps_per_episode"] = args.num_steps_per_episode
    runner_config["params"]["config"]["max_env_steps"] = args.max_env_steps
    runner_config["params"]["config"]["batch_size"] = args.batch_size
    runner_config["params"]["config"]["replay_buffer_size"] = args.replay_buffer_size
    runner_config["params"]["config"]["num_warmup_steps"] = args.num_warmup_steps
    runner_config["params"]["config"]["warmup_mode"] = args.warmup_mode
    runner_config["params"]["config"]["updates_per_step"] = args.updates_per_step
    runner_config["params"]["config"]["actor_lr"] = args.actor_lr
    runner_config["params"]["config"]["critic_lr"] = args.critic_lr
    runner_config["params"]["config"]["critic_tau"] = args.critic_tau
    runner_config["params"]["config"]["grad_norm"] = args.grad_norm
    runner_config["params"]["config"]["actor_update_interval"] = args.actor_update_interval
    runner_config["params"]["config"]["target_update_interval"] = args.target_update_interval
    runner_config["params"]["config"]["exploration_noise"] = args.exploration_noise
    runner_config["params"]["config"]["policy_noise"] = args.policy_noise
    runner_config["params"]["config"]["noise_clip"] = args.noise_clip
    runner_config["params"]["config"]["reward_shaper"]["scale_value"] = args.reward_scale

    runner_config["params"]["network"]["mlp"]["units"] = [args.mlp_size_last * i for i in (4, 2, 1)]

    if args.quiet:
        with suppress_stdout_stderr():
            runner.load(runner_config)
    else:
        runner.load(runner_config)

    return runner, full_experiment_name


if __name__ == "__main__":
    initial_algo_name = load_runner_config()["params"]["algo"]["name"]
    final_experiment_name = resolve_experiment_name(initial_algo_name)
    run_dir = os.path.join(get_run_root_dir(), final_experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "run.log")

    with redirect_output_to_log(log_file_path):
        if args.train_or_test == "train":
            runner, full_experiment_name = create_and_load_runner(final_experiment_name)
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
                    if not list_of_files:
                        raise FileNotFoundError(
                            f"No checkpoint matched pattern last_{args.env}_ep_{args.epoch_index}_rew_*.pth"
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
                    runner, full_experiment_name = create_and_load_runner(final_experiment_name)
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
