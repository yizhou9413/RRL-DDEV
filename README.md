# RRL-DDEV

**Robust Reinforcement Learning for Distributed-Drive Electric Vehicles Under Extreme Maneuvers**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A PyTorch framework for training robust autonomous driving policies on distributed-drive EVs. Features a GPU-accelerated simulator running **30,000 parallel environments** and a **Robust Policy Iteration (RPI)** algorithm that requires no adversarial policy.

---

## Highlights

- **GPU-accelerated simulation** — ~6×10⁹ steps in ~3 hours on a single RTX 5070 Ti
- **Coupled spatial dynamics** — planar motion, heave/roll/pitch, unsprung mass, and active suspension
- **Robust Policy Iteration** — first-order worst-case value bound embedded into TD error and GAE
- **7 algorithms** under one framework — PPO, SAC, TD3, DR-PPO, RARL, K-RARL, and RPI

## Quick Start

### Install

```bash
conda env create -f environment.yml
git submodule init && git submodule update
```

### Train

```bash
# RPI training on moose test
python run5.py train moose \
    --car-preset tesla_model_3 \
    --device cuda:0 \
    --num-parallel 30000 \
    --seed 1 \
    --disturbed
```

### Evaluate

```bash
python run5.py test moose \
    --car-preset tesla_model_3 \
    --device cuda:0 \
    --num-parallel 10000 \
    --checkpoint runs/<path_to_model>
```

## Environments

| Scenario | Description | Code |
|----------|-------------|------|
| Moose Test | Double lane-change obstacle avoidance | [`moose.py`](envs/moose.py) |
| Emergency Lane Change | Single lane-change under extreme conditions | [`singlelane.py`](envs/singlelane.py) |
| Constant Radius Drifting | Sustained drifting with velocity regulation | [`fixed_circle_iwd.py`](envs/fixed_circle_iwd.py) |

## Code Structure

```
├── envs/                        # Task environments
├── experiments/                 # Training scripts & plots
├── rl_games/                    # RL algorithms (submodule)
├── utils/                       # Utilities
├── xcar-simulation/             # GPU simulator & vehicle dynamics
├── run5.py                      # Main entry point (PPO / RPI / DR-PPO)
├── run_rarl.py                  # RARL training
├── run_sac.py                   # SAC training
├── run_TD3.py                   # TD3 training
└── runner_config.yaml           # Default config
```

## Citation

```bibtex
@article{rrl_ddev2026,
  title={Robust Reinforcement Learning for Distributed Drive Electric Vehicles Under Extreme Conditions},
  author={Author},
  journal={IEEE Transactions},
  year={2026}
}
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
