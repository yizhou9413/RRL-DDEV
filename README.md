# \# RRL-DDEV: Robust Reinforcement Learning for Distributed Drive Electric Vehicles Under Extreme Conditions

# 

# A PyTorch-based framework for training robust autonomous driving policies for distributed drive electric vehicles (DDEVs) under extreme conditions. This work proposes a \*\*Robust Policy Iteration (RPI)\*\* algorithm supported by a massively parallelized DDEV simulator with coupled spatial vehicle dynamics and active suspension.

# 

# \[\*\*Installation\*\*](#installation) | \[\*\*Quick Start\*\*](#quick-start) | \[\*\*Environments\*\*](#environments) | \[\*\*RPI Algorithm\*\*](#rpi-algorithm) | \[\*\*Training\*\*](#training) | \[\*\*Experimental Results\*\*](#experimental-results) | \[\*\*Code Structure\*\*](#code-structure)

# 

# \## Key Features

# 

# \- \*\*Massively Parallel Simulation\*\* — GPU-accelerated simulator running 30,000 parallel environments, collecting \~6x10^9 interaction steps in \~3 hours on a single NVIDIA RTX 5070 Ti

# \- \*\*Coupled Spatial Vehicle Dynamics\*\* — Full-body vehicle model including planar motion, yaw, heave, roll, pitch, unsprung mass dynamics, and active suspension force regulation for each wheel corner

# \- \*\*Robust Policy Iteration (RPI)\*\* — Embeds a first-order worst-case value lower bound into TD error and GAE, providing conservative policy updates without explicit adversarial policy training

# \- \*\*Unified Robustness Evaluation\*\* — Systematic evaluation under external disturbances and parameter perturbations across multiple extreme scenarios with consistent protocol

# \- \*\*Multi-Algorithm Comparison\*\* — Includes implementations of PPO, SAC, TD3, DR-PPO, RARL, and K-RARL under a unified model, reward, and training budget

# 

# \## Installation

# 

# \### Requirements

# \- NVIDIA GPU with CUDA support (Tested on NVIDIA RTX 5070 Ti)

# \- CUDA 11.8 or later

# \- PyTorch 2.0.1 or later

# 

# Create a conda environment using the provided environment.yml file:

# 

# ```bash

# conda env create -f environment.yml

# ```

# 

# \### Submodules

# This project uses Git submodules. After cloning the repository, initialize and update the submodules:

# 

# ```bash

# git submodule init

# git submodule update

# ```

# 

# The following submodule is included:

# \- \*\*rl\_games\*\*: Reinforcement learning algorithm library (forked from \[Denys88/rl\_games])

# &#x20; - Path: `rl\_games`

# &#x20; - URL: https://github.com/yiwenlu66/rl\_games.git

# 

# \## Quick Start

# 

# To train an RPI policy for the moose test scenario:

# 

# ```bash

# python run5.py train moose \\

# &#x20;   --car-preset tesla\_model\_3 \\

# &#x20;   --device cuda:0 \\

# &#x20;   --num-parallel 30000 \\

# &#x20;   --seed 1 \\

# &#x20;   --disturbed

# ```

# 

# To train with domain randomization (DR-PPO):

# 

# ```bash

# python run5.py train singlelane \\

# &#x20;   --car-preset tesla\_model\_3 \\

# &#x20;   --device cuda:0 \\

# &#x20;   --num-parallel 30000 \\

# &#x20;   --seed 1 \\

# &#x20;   --disturbed \\

# &#x20;   --w-force 2500 --w-phi 25000 --w-theta 22000

# ```

# 

# See `DR-PPO\_sup.sh` for batch training scripts across all scenarios and algorithms.

# 

# \## Environments

# 

# Three extreme driving control scenarios are provided, each requiring coordinated wheel-torque and active-suspension control near the handling boundary:

# 

# | Environment | Description | Code |

# |------------|-------------|------|

# | Moose Test | Double lane-change obstacle avoidance at high speed | \[moose.py](envs/moose.py) |

# | Emergency Lane Change | Single lane-change maneuver under extreme conditions | \[singlelane.py](envs/singlelane.py) |

# | Constant Radius Drifting | Sustained drifting at large sideslip angle with velocity regulation | \[fixed\_circle\_iwd.py](envs/fixed\_circle\_iwd.py) |

# 

# Additional legacy environments are also available:

# \- \[eight\_drift\_iwd.py](envs/eight\_drift\_iwd.py) — Figure-eight drifting

# \- \[continuous\_drift\_iwd.py](envs/continuous\_drift\_iwd.py) — Continuous drift trajectories

# \- \[state\_tracker\_iwd.py](envs/state\_tracker\_iwd.py) — State tracking for drifting

# \- \[lane.py](envs/lane.py) — Lane change stability

# 

# \## RPI Algorithm

# 

# Robust Policy Iteration (RPI) addresses performance degradation under external disturbances and model mismatch by incorporating a first-order worst-case value lower bound into the policy optimization loop.

# 

# \### Core Idea

# 

# For a perturbed next state `x\_{t+1} = x^in\_{t+1} + Delta\_t` with `|Delta\_{t,i}| <= xi\_{t,i}`, RPI approximates the worst-case value as:

# 

# ```

# B\_t^R = V(x^in\_{t+1}) - eta |nabla\_x V(x^in\_{t+1})|^T xi\_t

# ```

# 

# where `eta > 0` is a penalty factor that scales the worst-case correction. This term penalizes states where the future value is sensitive to bounded nearby deviations.

# 

# \### Robust TD Error and GAE

# 

# The robust TD error replaces the standard bootstrapped value with the robust Bellman term:

# 

# ```

# delta\_t^R = r\_t + gamma(1-d\_t) B\_t^R - V(x\_t)

# ```

# 

# Robust GAE is computed via backward recursion with episode termination masking:

# 

# ```

# A\_t^R = delta\_t^R + gamma\*lambda\*(1-d\_t)\*A\_{t+1}^R

# ```

# 

# The actor is then updated using the PPO clipped surrogate with robust advantages `A\_t^R`, and the critic is trained to minimize the robust value loss. No adversarial disturbance policy is needed.

# 

# \### Comparison Methods

# 

# Seven algorithms are compared under a unified framework:

# 

# | Method | Type | Description |

# |--------|------|-------------|

# | \*\*RPI\*\* | Robust on-policy | Worst-case value lower bound in policy evaluation |

# | PPO | On-policy baseline | Standard proximal policy optimization |

# | SAC | Off-policy | Soft actor-critic |

# | TD3 | Off-policy | Twin delayed DDPG |

# | DR-PPO | Domain randomization | PPO with randomized dynamics during training |

# | RARL | Adversarial | Robust adversarial reinforcement learning |

# | K-RARL | Adversarial | Kernelized RARL |

# 

# \## Training

# 

# \### Training Command

# 

# ```bash

# python run5.py train <env\_name> \\

# &#x20;   --car-preset tesla\_model\_3 \\

# &#x20;   --device cuda:0 \\

# &#x20;   --num-parallel 30000 \\

# &#x20;   --seed <seed> \\

# &#x20;   --disturbed

# ```

# 

# Available environment names: `moose`, `singlelane`, `fixed\_circle\_iwd`, `eight\_drift\_iwd`, `continuous\_drift\_iwd`, `state\_tracker\_iwd`, `lane`

# 

# \### Key Training Flags

# 

# \- `--car-preset`: Vehicle configuration (tesla\_model\_3, xcar, racecar, sensorcar)

# \- `--device`: CUDA device

# \- `--num-parallel`: Number of parallel environments (default: 30000)

# \- `--seed`: Random seed

# \- `--disturbed`: Enable disturbance modeling for RPI

# \- `--w-force`: External force disturbance scale

# \- `--w-phi` / `--w-theta`: Roll/pitch moment disturbance scale

# \- `--w-m` / `--w-k21` / `--w-c21` / `--w-k23` / `--w-c23`: Parameter perturbation scales (mass, suspension stiffness/damping)

# 

# \### Hyperparameters

# 

# | Hyperparameter | Value |

# |---|---|

# | Actor learning rate | 3e-4 |

# | Critic learning rate | 3e-4 |

# | Discount factor gamma | 0.99 |

# | GAE coefficient lambda | 0.95 |

# | PPO clip ratio | 0.2 |

# | KL divergence target | 0.008 |

# | RPI penalty factor eta | 0.002 |

# | Polyak coefficient tau | 0.005 |

# | Activation function | tanh |

# | Actor/Critic MLP hidden layers | \[256, 128, 64] |

# | Number of parallel environments | 30000 |

# | Optimizer | Adam |

# 

# \## Experimental Results

# 

# \### Success Rates

# 

# Closed-loop success rates across three scenarios under external disturbances and parameter perturbations (3 seeds x 10000 trajectories each):

# 

# | Algorithm | Moose Disturb. | Moose Param. | Lane Change Disturb. | Lane Change Param. | Drifting Disturb. | Drifting Param. |

# |-----------|:-:|:-:|:-:|:-:|:-:|:-:|

# | \*\*RPI\*\* | \*\*99.3%\*\* | \*\*96.3%\*\* | \*\*97.1%\*\* | \*\*100.0%\*\* | \*\*84.3%\*\* | \*\*94.1%\*\* |

# | PPO | 96.8% | 77.6% | 86.1% | 61.7% | 31.4% | 57.1% |

# | DR-PPO | 99.2% | 92.3% | 92.2% | 94.0% | 43.0% | 65.4% |

# | RARL | 95.9% | 99.0% | 76.2% | 100.0% | 17.8% | 23.1% |

# | K-RARL | 98.4% | 98.9% | 92.6% | 98.5% | 14.2% | 31.9% |

# | SAC | 41.1% | 33.0% | 80.5% | 63.0% | 32.7% | 51.2% |

# | TD3 | 0.7% | 0.3% | 3.3% | 0.3% | 11.2% | 15.1% |

# 

# RPI achieves the highest success rate in 5 of 6 test conditions and remains competitive in the sixth, with success rates ranging from 84.3% to 100.0%.

# 

# \### Key Findings

# 

# \- RPI improves feasibility over PPO by 2.5–52.9 percentage points under external disturbances

# \- Under parameter perturbations, RPI gains 18.7–38.3 percentage points over PPO

# \- RPI shows the most consistent cross-scenario generalization among all evaluated methods

# \- Radar chart analysis confirms RPI maintains a balanced tradeoff among trajectory tracking, sideslip regulation, velocity control, and roll-pitch attitude stabilization

# 

# \### Disturbance Protocol

# 

# \*\*External disturbances\*\* include per-wheel force perturbations (Fx, Fy) and roll/pitch moment perturbations:

# 

# ```

# d\_ext = \[Fx\_fl, Fy\_fl, Fx\_fr, Fy\_fr, Fx\_rl, Fy\_rl, Fx\_rr, Fy\_rr, M\_phi, M\_theta]

# ```

# 

# \*\*Parameter perturbations\*\* randomize vehicle mass and suspension stiffness/damping within +/-25% of nominal values:

# 

# ```

# delta\_p = \[delta\_m, delta\_K21, delta\_C21, delta\_K23, delta\_C23]

# ```

# 

# \## Technical Details

# 

# \### Vehicle Dynamics

# 

# The simulator implements a coupled spatial vehicle dynamics model with the state vector:

# 

# ```

# x = \[X, Y, psi, vx, vy, r, Zs, Zs\_dot, phi, phi\_dot, theta, theta\_dot, z\_u, z\_u\_dot, omega]

# ```

# 

# and control input:

# 

# ```

# u = \[delta\_f, omega\_cmd, f\_active\_suspension]

# ```

# 

# The model captures planar motion, yaw dynamics, sprung mass heave/roll/pitch, unsprung mass displacements, and wheel angular speeds, with active suspension forces entering through the roll and pitch moment equations.

# 

# \### Parallel Simulation

# 

# The GPU-accelerated simulator runs `n` independent vehicle environments simultaneously using batched tensor operations. Each batch row stores one trajectory instance with its own state, task variables, parameters, disturbances, reward, and termination flag. Dynamics are advanced using fixed-step explicit Euler integration.

# 

# \### Architecture

# 

# \- \*\*Core Simulation\*\* (`xcar-simulation/`): `gpu\_vectorized\_car\_env.py` (parallelized Gym environment), `IWDCarDynamics.py` (vehicle dynamics module), `presets.yaml` (vehicle parameters)

# \- \*\*Environments\*\* (`envs/`): Task-specific environments inheriting from `GPUVectorizedCarEnv`, overriding observation and reward interfaces

# \- \*\*RL Algorithms\*\* (`rl\_games/`): Modified rl\_games library with RPI, RARL, SAC, and TD3 support

# \- \*\*Utilities\*\* (`utils/`): Disturbance utilities, RL game interface adapters

# 

# \## Code Structure

# 

# ```

# ├── envs/                        # Environment implementations

# │   ├── moose.py                 # Moose test scenario

# │   ├── singlelane.py            # Emergency lane change scenario

# │   ├── fixed\_circle\_iwd.py      # Constant radius drifting

# │   ├── lane.py                  # Lane change stability

# │   ├── eight\_drift\_iwd.py       # Figure-eight drifting

# │   ├── continuous\_drift\_iwd.py  # Continuous drift trajectories

# │   └── state\_tracker\_iwd.py     # State tracking for drifting

# ├── experiments/                 # Training scripts and visualizations

# │   ├── fixed\_circle\_iwd/

# │   ├── eight\_drift\_iwd/

# │   ├── continuous\_drift\_iwd/

# │   └── state\_tracker\_iwd/

# ├── rl\_games/                    # RL algorithm library (submodule)

# │   └── algos\_torch/

# │       ├── a2c\_continuous.py    # PPO / RPI implementation

# │       ├── rarl\_continuous.py   # RARL implementation

# │       ├── sac\_agent.py         # SAC implementation

# │       └── td3\_agent.py         # TD3 implementation

# ├── utils/                       # Utility functions

# │   ├── disturbance\_utils.py     # Disturbance and RPI scale utilities

# │   ├── rlgame\_utils.py          # RL game interface adapters

# │   └── generate\_segment\_racetrack.py

# ├── xcar-simulation/             # Core simulation components

# │   ├── gpu\_vectorized\_car\_env.py

# │   ├── IWDCarDynamics.py

# │   └── presets.yaml

# ├── run5.py                      # Main training/evaluation entry point

# ├── run\_rarl.py                  # RARL training script

# ├── run\_sac.py                   # SAC training script

# ├── run\_TD3.py                   # TD3 training script

# ├── DR-PPO\_sup.sh                # Batch training script

# └── runner\_config.yaml           # Default runner configuration

# ```

# 

# \## Citation

# 

# If you find this work useful, please cite:

# 

# ```bibtex

# @article{rrl\_ddev2026,

# &#x20; title={Robust Reinforcement Learning for Distributed Drive Electric Vehicles Under Extreme Conditions},

# &#x20; author={Author},

# &#x20; journal={IEEE Transactions},

# &#x20; year={2026}

# }

# ```

# 

# \## Contributing

# 

# Contributions are welcome! Please feel free to submit a Pull Request.

# 

# \## License

# 

# This project is licensed under the MIT License - see the LICENSE file for details.

# 

