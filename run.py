import os
import sys
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "rl_games"))
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from utils.rlgame_utils import RLGPUEnv, RLGPUAlgoObserver
from envs.fixed_circle_iwd import FixedCircleIWDEnv
from envs.eight_drift_iwd import EightDriftIWDEnv
from envs.state_tracker_iwd import StateTrackerIWDEnv
from envs.continuous_drift_iwd import ContinuousDriftIWDEnv
from envs.singlelane import SingleLaneChangeStabilityIWDEnv
from envs.lane import LaneChangeStabilityIWDEnv
from envs.moose import MooseIWDEnv
import yaml
import argparse
import glob
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from icecream import ic
import pprint
from datetime import datetime

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)
ic.configureOutput(argToStringFunction=lambda x: pprint.pformat(x, sort_dicts=False))

parser = argparse.ArgumentParser()
parser.add_argument("train_or_test", type=str, help="Train or test")
parser.add_argument("env", type=str)
parser.add_argument("--car-preset", type=str, default="racecar")
parser.add_argument("--device", type=str, default='cuda:0')
parser.add_argument("--dt", type=float, default=0.01)
parser.add_argument("--rnn", action='store_true')
parser.add_argument("--disturbed", action='store_true')
parser.add_argument("--randomize-tyre", nargs='?', const='big', default=None, type=str)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--exp-name", type=str, default="default")
parser.add_argument("--epochs", type=int, default=5000)
parser.add_argument("--num-parallel", type=int, default=40000)
parser.add_argument("--lr-schedule", type=str, default="adaptive")
parser.add_argument("--mini-epochs", type=int, default=5)
parser.add_argument("--mlp-size-last", type=int, default=64)
parser.add_argument("--gamma", type=float, default=0.99)
parser.add_argument("--horizon", type=int, default=200)
parser.add_argument("--score-to-win", type=int, default=20000)
parser.add_argument("--save-freq", type=int, default=50)
parser.add_argument("--epoch-index", type=int, default=-1, help="For test only, -1 for using latest")
parser.add_argument("--checkpoint", type=str, default=None, help="Direct path to checkpoint file (for test or resume training)")
parser.add_argument("--resume", action='store_true', help="Resume training from checkpoint")
parser.add_argument("--latent-size", type=int, default=8)
parser.add_argument("--quiet", action='store_true')
parser.add_argument("--aux-reward-decay-steps", type=int, default=0)
parser.add_argument("--aux-reward-coef", type=float, default=1.)
parser.add_argument("--env-variant", type=str, default="")
parser.add_argument("--ref-mode", type=str, default="hybrid")
args = parser.parse_args()

if args.train_or_test == "test":
    # Turn off disturbance and randomization when testing
    args.disturbed = False
    args.randomize_tyre = None

def get_num_parallel():
    if args.train_or_test == "train":
        return args.num_parallel
    elif args.train_or_test == "test":
        return 1                          

envs = {
    "fixed_circle_iwd": lambda **kwargs: FixedCircleIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "eight_drift_iwd": lambda **kwargs: EightDriftIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "state_tracker_iwd": lambda **kwargs: StateTrackerIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "continuous_drift_iwd": lambda **kwargs: ContinuousDriftIWDEnv(args.car_preset, get_num_parallel(), args.device, args.ref_mode, **kwargs),
    "singlelane": lambda **kwargs: SingleLaneChangeStabilityIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "lane": lambda **kwargs: LaneChangeStabilityIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
    "moose": lambda **kwargs: MooseIWDEnv(args.car_preset, get_num_parallel(), args.device, **kwargs),
}

# Common env config
default_env_config = {
    "dt": args.dt,
    "disturbance_param": (0.97, 4) if args.disturbed else None,
    "randomize_param": {
        None: {},
        "big": {
            "B": [8.3, 11.7],
            "C": [1.45, 2.38],
            "D": [0.75, 1.25],
            "m": [1612., 1930.],  # 车辆质量 (kg)，基准值1760，±30%范围
            "k_phi": [10350., 14650.],      # 横滚刚度降到极低，车辆应该严重侧倾
            "c_phi": [1040., 1460.],        # 横滚阻尼极低
            "k_theta": [8800., 11500.],    # 俯仰刚度极低
            "c_theta": [1610., 2090.],
        },
        "small": {
            "B": [9, 11.],
            "C": [1.6, 2.2],
            "D": [0.9, 1.1],
            "m": [1672., 1848.], 
            "k_phi": [11350., 14650.],      # 横滚刚度降到极低，车辆应该严重侧倾13000
            "c_phi": [1140., 1460.],        # 横滚阻尼极低1200
            "k_theta": [10000., 11500.],    # 俯仰刚度极低10000
            "c_theta": [1710., 1990.], #1800
        },
        "xuanjia": {
            "k_phi": [10350., 15650.],      # 横滚刚度降到极低，车辆应该严重侧倾
            "c_phi": [1040., 1560.],        # 横滚阻尼极低
            "k_theta": [8500., 12500.],    # 俯仰刚度极低
            "c_theta": [1710., 2190.],      # 俯仰阻尼极低
        },  
    }[args.randomize_tyre],
    "random_seed": args.seed,
    "quiet": args.quiet,
    "aux_reward_decay_steps": args.aux_reward_decay_steps,
    "aux_reward_coef": args.aux_reward_coef,
    "gamma": args.gamma,
    "train": (args.train_or_test == "train"),
    "min_rollout_steps": (args.horizon if args.train_or_test == "test" else 1),
}

# Environment-specific config
if args.env == "state_tracker_iwd":
    # Reference generation mode
    if args.train_or_test == "train":
        default_env_config["ref_mode"] = "hybrid"
    else:
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

blacklist_keys = lambda d, blacklist: {k: d[k] for k in d if not (k in blacklist)}
vecenv.register('RLGPU',
                lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
env_configurations.register('rlgpu', {
    'vecenv_type': 'RLGPU',
    'env_creator': lambda **env_config: envs[args.env](
        **blacklist_keys(default_env_config, env_config.keys()),
        **env_config,
    ),
})

runner = Runner(RLGPUAlgoObserver())
file_path = os.path.dirname(__file__)
with open(os.path.join(file_path, "runner_config.yaml")) as f:
    runner_config = yaml.safe_load(f)
full_experiment_name = args.env + "_" + args.exp_name
if args.train_or_test == "train":
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    full_experiment_name = f"{full_experiment_name}_{timestamp}"
# 获取算法名称并添加到文件名前面
algo_name = runner_config["params"]["algo"]["name"]
full_experiment_name = f"{algo_name}_{full_experiment_name}"
# 将实验名称传递给环境（用于按实验目录保存csv等数据）
default_env_config["full_experiment_name"] = full_experiment_name
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
runner_config["params"]["network"]["mlp"]["units"] = [args.mlp_size_last * i for i in (4, 2, 1)]
runner_config["params"]["config"]["save_frequency"] = args.save_freq
runner_config["params"]["config"]["device_name"] = args.device
runner_config["params"]["config"]["device"] = args.device
if not args.rnn:
    runner_config["params"]["network"].pop("rnn")

if args.quiet:
    with suppress_stdout_stderr():
        runner.load(runner_config)
else:
    runner.load(runner_config)

if __name__ == "__main__":
    if args.train_or_test == "train":
        train_config = {'train': True}
        
        # 如果指定了resume或checkpoint，从已有模型继续训练
        if args.resume or args.checkpoint:
            if args.checkpoint:
                checkpoint_name = args.checkpoint
            else:
                # 如果只指定了--resume，从最新的checkpoint继续
                checkpoint_dir = f"runs/{full_experiment_name}/nn"
                checkpoint_name = f"{checkpoint_dir}/{args.env}.pth"
            
            print(f"Resuming training from checkpoint: {checkpoint_name}")
            train_config['checkpoint'] = checkpoint_name
        
        runner.run(train_config)
    elif args.train_or_test == "test":
        # 如果直接指定了checkpoint路径，使用该路径
        if args.checkpoint:
            checkpoint_name = args.checkpoint
        else:
            # 否则根据实验名称和epoch索引查找
            checkpoint_dir = f"runs/{full_experiment_name}/nn"
            if args.epoch_index == -1:
                checkpoint_name = f"{checkpoint_dir}/{args.env}.pth"
            else:
                list_of_files = glob.glob(f"{checkpoint_dir}/last_{args.env}_ep_{args.epoch_index}_rew_*.pth")
                checkpoint_name = max(list_of_files, key=os.path.getctime)
        
        print(f"Loading checkpoint: {checkpoint_name}")
        runner.run({
            'train': False,
            'play': True,
            'checkpoint' : checkpoint_name,
        })
