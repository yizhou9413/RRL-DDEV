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

## Environments

| Scenario | Description |
|----------|-------------|
| Moose Test | Double lane-change obstacle avoidance |
| Emergency Lane Change | Single lane-change under extreme conditions |
| Constant Radius Drifting | Sustained drifting with velocity regulation |

## Repository Structure

```
├── experiments/                 # Experiment scripts per scenario
├── plots/                       # Plotting, rendering scripts and sample data
└── README.md
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
