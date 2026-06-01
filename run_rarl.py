"""
RARL training/test entry.

Usage:
    train: python run_rarl.py train <env_name> [options]
    test : python run_rarl.py test <env_name> [options]
"""

import argparse
import os
import pprint
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime

import yaml
from icecream import ic

file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))
sys.path.append(os.path.join(file_path, "xcar-simulation"))

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from utils.rlgame_utils import RLGPUAlgoObserver, RLGPUEnv
from utils.disturbance_utils import (
    build_disturbance_range_from_args,
    format_disturbance_bounds_for_display,
    redact_disturbance_scale_tags,
)
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
from envs.eight_drift_iwd import EightDriftIWDEnv
from envs.fixed_circle_iwd import FixedCircleIWDEnv
from envs.lane import LaneChangeStabilityIWDEnv
from envs.moose import MooseIWDEnv
from envs.singlelane import SingleLaneChangeStabilityIWDEnv
from envs.state_tracker_iwd import StateTrackerIWDEnv


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


def load_runner_config():
    with open(os.path.join(file_path, "rarl_config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def _format_name_number(value):
    value = abs(float(value))
    rounded = round(value)
    if abs(value - rounded) < 1e-6:
        return str(int(rounded))
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p")


def _build_disturbance_range_from_args():
    return build_disturbance_range_from_args(args)


def _coerce_positive_int(value, default=1):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return parsed if parsed > 0 else default


def _coerce_bool(value, default=False):
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on", "enable", "enabled"}:
            return True
        if text in {"0", "false", "no", "n", "off", "disable", "disabled"}:
            return False
    return default


def _get_adv_action_space_tag(runner_config):
    params = runner_config.get("params", {})
    adv_action_space = (
        params.get("config", {}).get("adv_action_space")
        or params.get("rarl", {}).get("adv_action_space")
    )
    if not adv_action_space:
        return ""

    low = adv_action_space.get("low")
    high = adv_action_space.get("high")
    if not isinstance(low, list) or not isinstance(high, list):
        return ""
    if len(low) != len(high) or len(low) < 15:
        return ""

    magnitudes = [max(abs(float(lo)), abs(float(hi))) for lo, hi in zip(low, high)]

    # Naming rule requested by user:
    # dims 0..7 -> use dim0 magnitude once, then dims 8..14 one by one.
    compact_values = [
        magnitudes[0],
        magnitudes[8],
        magnitudes[9],
        magnitudes[10],
        magnitudes[11],
        magnitudes[12],
        magnitudes[13],
        magnitudes[14],
    ]
    return "_" + "_".join(_format_name_number(v) for v in compact_values)


def _sanitize_name_token(value, default="unknown"):
    text = str(value).strip().lower()
    if not text:
        return default

    sanitized = "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "-" for ch in text)
    sanitized = sanitized.strip("-_")
    return sanitized or default


def _get_rarl_mode_tag(runner_config):
    params = runner_config.get("params", {})
    config = params.get("config", {})

    branch = _sanitize_name_token(config.get("rarl_branch", "original"), default="original")
    if branch == "sampled_adv_scale":
        sampled_cfg = config.get("sampled_adv_scale", {})
        dist = _sanitize_name_token(sampled_cfg.get("distribution", "normal"), default="normal")
        return f"_{branch}_{dist}"
    return f"_{branch}"


def _get_adv_output_freq_tag(runner_config):
    params = runner_config.get("params", {})
    config = params.get("config", {})
    fallback = params.get("rarl", {}).get("adv_output_freq", 1)
    freq = config.get("adv_output_freq", fallback)
    return f"_af{_coerce_positive_int(freq, default=1)}"


def _get_grad_reg_tag(runner_config):
    params = runner_config.get("params", {})
    config = params.get("config", {})
    enabled = _coerce_bool(config.get("grad_reg_enabled", False), default=False)
    return "_gron" if enabled else "_groff"


def _get_effective_risk_q_enabled(runner_config):
    params = runner_config.get("params", {})
    config = params.get("config", {})
    enabled_new = _coerce_bool(config.get("use_risk_q", False), default=False)
    enabled_legacy = _coerce_bool(config.get("use_rararl_q", False), default=False)
    return enabled_new or enabled_legacy


def _get_risk_q_tag(runner_config):
    enabled = _get_effective_risk_q_enabled(runner_config)
    return f"_riskq_{int(enabled)}"


def _get_legacy_rararl_q_tag(runner_config):
    enabled = _get_effective_risk_q_enabled(runner_config)
    return f"_rararlq_{int(enabled)}"


def _build_experiment_name_prefix(algo_name, runner_config=None):
    base_name = f"{algo_name}_{args.env}_{args.num_parallel}"
    if runner_config is None:
        return base_name
    return (
        base_name
        + _get_rarl_mode_tag(runner_config)
        + _get_risk_q_tag(runner_config)
        + _get_grad_reg_tag(runner_config)
        + _get_adv_output_freq_tag(runner_config)
        + _get_adv_action_space_tag(runner_config)
    )


def _build_legacy_experiment_name(algo_name, runner_config=None):
    base_name = f"{algo_name}_{args.env}_{args.seed}_{args.num_parallel}"
    if runner_config is None:
        return base_name
    return (
        base_name
        + _get_rarl_mode_tag(runner_config)
        + _get_risk_q_tag(runner_config)
        + _get_grad_reg_tag(runner_config)
        + _get_adv_output_freq_tag(runner_config)
        + _get_adv_action_space_tag(runner_config)
    )


def _build_experiment_name_prefix_without_rararl_q(algo_name, runner_config=None):
    base_name = f"{algo_name}_{args.env}_{args.num_parallel}"
    if runner_config is None:
        return base_name
    return (
        base_name
        + _get_rarl_mode_tag(runner_config)
        + _get_grad_reg_tag(runner_config)
        + _get_adv_output_freq_tag(runner_config)
        + _get_adv_action_space_tag(runner_config)
    )


def _build_legacy_experiment_name_without_rararl_q(algo_name, runner_config=None):
    base_name = f"{algo_name}_{args.env}_{args.seed}_{args.num_parallel}"
    if runner_config is None:
        return base_name
    return (
        base_name
        + _get_rarl_mode_tag(runner_config)
        + _get_grad_reg_tag(runner_config)
        + _get_adv_output_freq_tag(runner_config)
        + _get_adv_action_space_tag(runner_config)
    )


def _build_experiment_name_prefix_with_legacy_rararl_q(algo_name, runner_config=None):
    base_name = f"{algo_name}_{args.env}_{args.num_parallel}"
    if runner_config is None:
        return base_name
    return (
        base_name
        + _get_rarl_mode_tag(runner_config)
        + _get_legacy_rararl_q_tag(runner_config)
        + _get_grad_reg_tag(runner_config)
        + _get_adv_output_freq_tag(runner_config)
        + _get_adv_action_space_tag(runner_config)
    )


def _build_legacy_experiment_name_with_legacy_rararl_q(algo_name, runner_config=None):
    base_name = f"{algo_name}_{args.env}_{args.seed}_{args.num_parallel}"
    if runner_config is None:
        return base_name
    return (
        base_name
        + _get_rarl_mode_tag(runner_config)
        + _get_legacy_rararl_q_tag(runner_config)
        + _get_grad_reg_tag(runner_config)
        + _get_adv_output_freq_tag(runner_config)
        + _get_adv_action_space_tag(runner_config)
    )


def build_full_experiment_name(algo_name, runner_config=None):
    return f"{_build_experiment_name_prefix(algo_name, runner_config)}_{args.seed}"


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


def resolve_experiment_name(algo_name, runner_config=None):
    current_name = build_full_experiment_name(algo_name, runner_config)
    if args.train_or_test == "train" and not args.resume and not args.checkpoint:
        return current_name

    run_root = get_run_root_dir()
    current_dir = os.path.join(run_root, current_name)
    if os.path.isdir(current_dir):
        return current_name

    candidate_names = [
        _build_legacy_experiment_name(algo_name, runner_config),
        f"{_build_experiment_name_prefix_without_rararl_q(algo_name, runner_config)}_{args.seed}",
        _build_legacy_experiment_name_without_rararl_q(algo_name, runner_config),
        f"{_build_experiment_name_prefix_with_legacy_rararl_q(algo_name, runner_config)}_{args.seed}",
        _build_legacy_experiment_name_with_legacy_rararl_q(algo_name, runner_config),
    ]
    for candidate in candidate_names:
        if os.path.isdir(os.path.join(run_root, candidate)):
            return candidate

    return current_name


def _apply_risk_q_overrides(config):
    if args.use_risk_q is not None:
        enabled = bool(args.use_risk_q)
        config["use_risk_q"] = enabled
        # Backward compatibility for historical checkpoints/configs.
        config["use_rararl_q"] = enabled

    if args.risk_q_lambda_protagonist is not None:
        value = float(args.risk_q_lambda_protagonist)
        config["risk_lambda_protagonist"] = value
        config["rararl_q_lambda_protagonist"] = value

    if args.risk_q_lambda_adversary is not None:
        value = float(args.risk_q_lambda_adversary)
        config["risk_lambda_adversary"] = value
        config["rararl_q_lambda_adversary"] = value

    if args.risk_q_ensemble_size is not None:
        value = max(2, int(args.risk_q_ensemble_size))
        config["q_ensemble_size"] = value
        config["rararl_q_num_heads"] = value

    if args.risk_q_adv_chunk_size is not None:
        value = max(1, int(args.risk_q_adv_chunk_size))
        config["risk_q_advantage_chunk_size"] = value
        config["rararl_q_advantage_chunk_size"] = value

    if args.q_loss_coef is not None:
        config["q_loss_coef"] = max(float(args.q_loss_coef), 0.0)

    if args.q_bootstrap_p is not None:
        config["q_bootstrap_p"] = min(max(float(args.q_bootstrap_p), 0.0), 1.0)

    if args.risk_q_obs_normalize is not None:
        enabled = bool(args.risk_q_obs_normalize)
        config["risk_q_obs_normalize"] = enabled
        config["rararl_q_obs_normalize"] = enabled

    if args.risk_q_blend_start_epoch is not None:
        value = max(int(args.risk_q_blend_start_epoch), 0)
        config["risk_q_adv_blend_start_epoch"] = value
        config["rararl_q_adv_blend_start_epoch"] = value

    if args.risk_q_blend_end_epoch is not None:
        value = max(int(args.risk_q_blend_end_epoch), 0)
        config["risk_q_adv_blend_end_epoch"] = value
        config["rararl_q_adv_blend_end_epoch"] = value

    if args.risk_q_blend_lambda_ref is not None:
        value = max(float(args.risk_q_blend_lambda_ref), 1e-12)
        config["risk_q_blend_lambda_ref"] = value
        config["rararl_q_blend_lambda_ref"] = value

    if args.risk_q_force_ppo_when_zero_lambda is not None:
        enabled = bool(args.risk_q_force_ppo_when_zero_lambda)
        config["risk_q_force_ppo_when_zero_lambda"] = enabled
        config["rararl_q_force_ppo_when_zero_lambda"] = enabled


def get_num_parallel():
    if args.train_or_test == "train":
        return args.num_parallel
    return 40


def get_env_creator():
    return {
        "fixed_circle_iwd": lambda **kwargs: FixedCircleIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
        "eight_drift_iwd": lambda **kwargs: EightDriftIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
        "state_tracker_iwd": lambda **kwargs: StateTrackerIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
        "continuous_drift_iwd": lambda **kwargs: ContinuousDriftIWDEnv(
            args.car_preset, get_num_parallel(), args.device, "hybrid", **kwargs
        ),
        "singlelane": lambda **kwargs: SingleLaneChangeStabilityIWDEnv(
            args.car_preset, get_num_parallel(), args.device, **kwargs
        ),
        "lane": lambda **kwargs: LaneChangeStabilityIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
        "moose": lambda **kwargs: MooseIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    }


def get_default_env_config():
    use_random_disturbance = args.train_or_test == "test" and args.disturbed
    use_rarl_mode = args.train_or_test == "train"
    return {
        "dt": args.dt,
        "disturbance_param": (
            (args.disturbance_rho, max(1, args.disturbance_every))
            if use_random_disturbance
            else None
        ),
        "disturbance_range": _build_disturbance_range_from_args() if use_random_disturbance else None,
        "randomize_param": {},
        "random_seed": args.seed,
        "train": args.train_or_test == "train",
        "rarl_mode": use_rarl_mode,
    }


def create_and_load_runner(full_experiment_name=None):
    default_env_config = get_default_env_config()
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
    adv_output_freq = _coerce_positive_int(args.adv_output_freq, default=1)
    runner_config["params"]["config"]["adv_output_freq"] = adv_output_freq
    runner_config["params"]["config"]["grad_reg_enabled"] = bool(args.grad_reg)
    _apply_risk_q_overrides(runner_config["params"]["config"])
    if args.grad_reg_coef is not None:
        runner_config["params"]["config"]["grad_reg_coef"] = max(float(args.grad_reg_coef), 0.0)
    algo_name = runner_config["params"]["algo"]["name"]
    if full_experiment_name is None:
        full_experiment_name = build_full_experiment_name(algo_name, runner_config)

    runner_config["params"]["seed"] = args.seed
    runner_config["params"]["config"]["num_actors"] = args.num_parallel
    runner_config["params"]["config"]["max_epochs"] = args.epochs
    runner_config["params"]["config"]["minibatch_size"] = args.num_parallel
    runner_config["params"]["config"]["games_to_track"] = args.num_parallel
    runner_config["params"]["config"]["mini_epochs"] = args.mini_epochs
    runner_config["params"]["config"]["lr_schedule"] = args.lr_schedule
    runner_config["params"]["config"]["gamma"] = args.gamma
    runner_config["params"]["config"]["horizon_length"] = args.horizon
    runner_config["params"]["config"]["name"] = args.env
    runner_config["params"]["config"]["full_experiment_name"] = full_experiment_name
    runner_config["params"]["config"]["train_dir"] = get_run_root_dir()
    runner_config["params"]["config"]["save_frequency"] = args.save_freq
    runner_config["params"]["config"]["device_name"] = args.device
    runner_config["params"]["config"]["device"] = args.device

    runner_config["params"]["config"]["n_pro_itr"] = args.n_pro_itr
    runner_config["params"]["config"]["n_adv_itr"] = args.n_adv_itr
    runner_config["params"]["config"]["adv_reward_scale"] = args.adv_scale
    runner_config["params"]["config"]["adv_output_freq"] = adv_output_freq
    runner_config["params"]["config"]["grad_reg_enabled"] = bool(args.grad_reg)
    if args.grad_reg_coef is not None:
        runner_config["params"]["config"]["grad_reg_coef"] = max(float(args.grad_reg_coef), 0.0)
    runner_config["params"]["config"]["rarl_mode"] = args.train_or_test == "train"

    if "rarl" in runner_config["params"] and "adv_action_space" in runner_config["params"]["rarl"]:
        runner_config["params"]["config"]["adv_action_space"] = runner_config["params"]["rarl"]["adv_action_space"]

    default_env_config["full_experiment_name"] = full_experiment_name
    if os.environ.get("XCAR_ROLLOUT_ROOT"):
        default_env_config["rollout_root_dir"] = os.environ["XCAR_ROLLOUT_ROOT"]
    if not args.rnn:
        runner_config["params"]["network"].pop("rnn", None)

    runner.load(runner_config)
    return runner, full_experiment_name


ic.configureOutput(argToStringFunction=lambda x: pprint.pformat(x, sort_dicts=False))

parser = argparse.ArgumentParser(description="RARL train/test runner")
parser.add_argument("train_or_test", type=str, help="train or test")
parser.add_argument("env", type=str, help="environment name")
parser.add_argument("--car-preset", type=str, default="tesla_model_3")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--rnn", action="store_true", help="use RNN network")
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--exp-name", type=str, default="default")
parser.add_argument("--epochs", type=int, default=650)
parser.add_argument("--num-parallel", type=int, default=30000)
parser.add_argument("--lr-schedule", type=str, default="linear")
parser.add_argument("--mini-epochs", type=int, default=5)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--save-freq", type=int, default=50)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument(
    "-disturbed",
    "--disturbed",
    "--distured",
    dest="disturbed",
    action="store_true",
    help="enable run5-style random AR disturbance during test",
)
parser.add_argument("--disturbance-rho", type=float, default=0.50, help="AR(1) coefficient for random disturbance")
parser.add_argument("--disturbance-every", type=int, default=1, help="refresh random disturbance every N steps")
parser.add_argument(
    "--disturbance-scale",
    type=float,
    default=1.0,
    help="global multiplier applied to all --w-* disturbance bounds before injecting them into the environment",
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
parser.add_argument("--w-force", type=float, default=2500.0, help="uniform disturbance abs bound for tyre force dims (N)")
parser.add_argument("--w-phi", type=float, default=25000.0, help="uniform disturbance abs bound for roll moment (N*m)")
parser.add_argument("--w-theta", type=float, default=22000.0, help="uniform disturbance abs bound for pitch moment (N*m)")
parser.add_argument("--w-m", type=float, default=0.0, help="uniform disturbance abs bound for mass offset (kg)")
parser.add_argument("--w-k21", type=float, default=0.0, help="uniform disturbance abs bound for K21 offset (N/m)")
parser.add_argument("--w-c21", type=float, default=0.0, help="uniform disturbance abs bound for C21 offset (N*s/m)")
parser.add_argument("--w-k23", type=float, default=0.0, help="uniform disturbance abs bound for K23 offset (N/m)")
parser.add_argument("--w-c23", type=float, default=0.0, help="uniform disturbance abs bound for C23 offset (N*s/m)")
parser.add_argument("--n-pro-itr", type=int, default=1, help="protagonist iters per cycle")
parser.add_argument("--n-adv-itr", type=int, default=1, help="adversary iters per cycle")
parser.add_argument("--adv-scale", type=float, default=1.0, help="adversarial reward scale")
parser.add_argument(
    "--adv-output-freq",
    type=int,
    default=1,
    help="adversary disturbance output interval in steps (default: 1, emit every step)",
)
parser.add_argument("--grad-reg", action="store_true", help="enable gradient regularization term in RL loss")
parser.add_argument(
    "--grad-reg-coef",
    type=float,
    default=None,
    help="gradient regularization coefficient (used with --grad-reg, default from config)",
)
parser.add_argument(
    "--use-risk-q",
    "--use-rararl-q",
    dest="use_risk_q",
    action="store_true",
    default=None,
    help="enable risk-adjusted ensemble Q for PPO-compatible actor advantage (legacy alias: --use-rararl-q)",
)
parser.add_argument(
    "--risk-q-lambda-protagonist",
    "--rararl-q-lambda-protagonist",
    dest="risk_q_lambda_protagonist",
    type=float,
    default=None,
    help="risk coefficient lambda_P for protagonist: Q_hat = Q_bar - lambda_P * Var(Q_i)",
)
parser.add_argument(
    "--risk-q-lambda-adversary",
    "--rararl-q-lambda-adversary",
    dest="risk_q_lambda_adversary",
    type=float,
    default=None,
    help="risk coefficient lambda_A for adversary: Q_hat = Q_bar + lambda_A * Var(Q_i)",
)
parser.add_argument(
    "--risk-q-ensemble-size",
    "--rararl-q-num-heads",
    dest="risk_q_ensemble_size",
    type=int,
    default=None,
    help="number of ensemble Q heads used by risk-adjusted Q",
)
parser.add_argument(
    "--risk-q-adv-chunk-size",
    "--rararl-q-adv-chunk-size",
    dest="risk_q_adv_chunk_size",
    type=int,
    default=None,
    help="chunk size for risk-adjusted Q advantage computation to reduce peak VRAM",
)
parser.add_argument(
    "--q-loss-coef",
    type=float,
    default=None,
    help="coefficient for auxiliary ensemble Q critic loss",
)
parser.add_argument(
    "--q-bootstrap-p",
    type=float,
    default=None,
    help="bootstrap mask Bernoulli probability for each Q head/sample",
)
parser.add_argument(
    "--risk-q-obs-normalize",
    dest="risk_q_obs_normalize",
    action="store_true",
    default=None,
    help="normalize observations for risk-Q critic using actor running_mean_std",
)
parser.add_argument(
    "--no-risk-q-obs-normalize",
    dest="risk_q_obs_normalize",
    action="store_false",
    help="disable observation normalization for risk-Q critic",
)
parser.add_argument(
    "--risk-q-blend-start-epoch",
    type=int,
    default=None,
    help="epoch to start blending PPO advantage -> risk-Q advantage",
)
parser.add_argument(
    "--risk-q-blend-end-epoch",
    type=int,
    default=None,
    help="epoch to finish blending PPO advantage -> risk-Q advantage",
)
parser.add_argument(
    "--risk-q-blend-lambda-ref",
    type=float,
    default=None,
    help="reference lambda for scaling risk advantage blend strength",
)
parser.add_argument(
    "--risk-q-force-ppo-when-zero-lambda",
    dest="risk_q_force_ppo_when_zero_lambda",
    action="store_true",
    default=None,
    help="fallback to PPO advantage when current role risk lambda is zero",
)
parser.add_argument(
    "--no-risk-q-force-ppo-when-zero-lambda",
    dest="risk_q_force_ppo_when_zero_lambda",
    action="store_false",
    help="disable zero-lambda fallback and always allow pure risk-Q advantage",
)
args = parser.parse_args()
args.adv_output_freq = _coerce_positive_int(args.adv_output_freq, default=1)


if __name__ == "__main__":
    initial_runner_config = load_runner_config()
    initial_runner_config["params"]["config"]["adv_output_freq"] = args.adv_output_freq
    initial_runner_config["params"]["config"]["grad_reg_enabled"] = bool(args.grad_reg)
    _apply_risk_q_overrides(initial_runner_config["params"]["config"])
    if args.grad_reg_coef is not None:
        initial_runner_config["params"]["config"]["grad_reg_coef"] = max(float(args.grad_reg_coef), 0.0)
    effective_use_risk_q = _get_effective_risk_q_enabled(initial_runner_config)
    initial_algo_name = initial_runner_config["params"]["algo"]["name"]
    final_experiment_name = resolve_experiment_name(initial_algo_name, initial_runner_config)
    run_dir = os.path.join(get_run_root_dir(), final_experiment_name)
    os.makedirs(run_dir, exist_ok=True)
    log_file_path = os.path.join(run_dir, "run.log")

    with redirect_output_to_log(log_file_path):
        runner, full_experiment_name = create_and_load_runner(final_experiment_name)
        final_experiment_name = full_experiment_name

        print("\n" + "=" * 60)
        print("RARL (Robust Adversarial Reinforcement Learning)")
        print("=" * 60)
        print(f"Env: {args.env}")
        print(f"Mode: {args.train_or_test}")
        if args.train_or_test == "test" and args.disturbed:
            print("Test mode: protagonist + run5-style random AR disturbance")
            print(f"Disturbance rho: {args.disturbance_rho}")
            print(f"Disturbance refresh interval: {max(1, args.disturbance_every)}")
            print("Injected disturbance bounds: " + format_disturbance_bounds_for_display(args))
        elif args.train_or_test == "test":
            print("Test mode: protagonist only (no adversary or random disturbance)")
        else:
            print(f"Protagonist iters/cycle: {args.n_pro_itr}")
            print(f"Adversary iters/cycle: {args.n_adv_itr}")
            print(f"Adversary output frequency (steps): {args.adv_output_freq}")
            print(f"Gradient regularization: {'ON' if args.grad_reg else 'OFF'}")
            print(f"Risk-adjusted Q: {'ON' if effective_use_risk_q else 'OFF'}")
        print(f"Num parallel envs: {args.num_parallel}")
        print("=" * 60 + "\n")

        if args.train_or_test == "train":
            train_config = {"train": True}
            if args.resume or args.checkpoint:
                if args.checkpoint:
                    checkpoint_name = args.checkpoint
                else:
                    checkpoint_dir = os.path.join(get_run_root_dir(), full_experiment_name, "nn")
                    checkpoint_name = os.path.join(checkpoint_dir, f"{args.env}.pth")
                print(f"Resume from checkpoint: {redact_disturbance_scale_tags(checkpoint_name)}")
                train_config["checkpoint"] = checkpoint_name
            runner.run(train_config)

        elif args.train_or_test == "test":
            if args.checkpoint:
                checkpoint_name = args.checkpoint
            else:
                checkpoint_dir = os.path.join(get_run_root_dir(), full_experiment_name, "nn")
                checkpoint_name = os.path.join(checkpoint_dir, f"{args.env}.pth")
            print(f"Loading checkpoint: {redact_disturbance_scale_tags(checkpoint_name)}")
            runner.run({"train": False, "play": True, "checkpoint": checkpoint_name})
