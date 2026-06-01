import numpy as np
import os
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import yaml
import gym
import time
import random
import csv
from pathlib import Path
from IWDCarDynamics import IWDCarDynamics

class GPUVectorizedCarEnv:
    def __init__(self,
        preset_name,
        n,
        dt=0.01,
        solver="euler",
        device="cuda:0",
        drivetrain="iwd",
        disturbance_param=None,
        disturbance_std=None,
        disturbance_range=None,
        randomize_param={},
        random_seed=None,
        initial_state=None,
        **kwargs,
    ):
        if random_seed is not None:
            torch.manual_seed(random_seed)
            torch.cuda.manual_seed_all(random_seed)
            np.random.seed(random_seed)
            random.seed(random_seed)
        self.num_states = 20
        # Observation ranges tuned for tesla_model_3.
        self.observation_space = gym.spaces.Box(low=np.array([-1000.,-1000.,-50.,-200.,-200.,-20.,-1.0,-1.0,-1.0,-0.5,-0.5,-0.5,-0.5,-10.,-10.,-10.,-50.,-50.,-50.,-50.]), high=np.array([1000.,1000.,50.,200.,200.,20.,1.0,1.0,1.0,0.5,0.5,0.5,0.5,10.,10.,10.,50.,50.,50.,50.]), shape=(20,))
        self.state_space = self.observation_space
        self.drivetrain = drivetrain
        # Action ranges tuned for tesla_model_3.
        if drivetrain == "4wd":
            self.num_actions = 6  # delta, omega*1, f1, f2, f3, f4
            self.action_space = gym.spaces.Box(low=np.array([-0.45, 0., -2000., -2000., -2000., -2000.]), high=np.array([0.45, 20., 2000., 2000., 2000., 2000.]), shape=(6,))
            self.cast_action = lambda u: torch.cat(list(map(lambda v: torch.unsqueeze(v, 1), [u[:, 0], u[:, 1], u[:, 1], u[:, 1], u[:, 1], u[:, 2], u[:, 3], u[:, 4], u[:, 5]])), 1)
        elif drivetrain == "2iwd":
            self.num_actions = 7  # delta, omega_f, omega_r, f1, f2, f3, f4
            self.action_space = gym.spaces.Box(low=np.array([-0.45, 0., 0., -2000., -2000., -2000., -2000.]), high=np.array([0.45, 20., 20., 2000., 2000., 2000., 2000.]), shape=(7,))
            self.cast_action = lambda u: torch.cat(list(map(lambda v: torch.unsqueeze(v, 1), [u[:, 0], u[:, 1], u[:, 1], u[:, 2], u[:, 2], u[:, 3], u[:, 4], u[:, 5], u[:, 6]])), 1)
        elif drivetrain == "iwd":
            self.num_actions = 9  # delta, omega_fr, omega_fl, omega_rr, omega_rl, f1, f2, f3, f4
            self.action_space = gym.spaces.Box(low=np.array([-0.45, 0., 0., 0., 0., -2000., -2000., -2000., -2000.]), high=np.array([0.45, 20., 20., 20., 20., 2000., 2000., 2000., 2000.]), shape=(9,))
            self.cast_action = lambda u: u

        self.preset_name = preset_name
        self.n = n
        self.dt = dt
        self.solver = solver
        self.device = torch.device(device)
        file_path = os.path.dirname(__file__)
        with open(os.path.join(file_path, "presets.yaml"),encoding='utf-8') as f:
            presets = yaml.safe_load(f)
            params = presets[preset_name]["parameters"]
        self.p_body = torch.zeros((n, 27), device=self.device)
        self.p_body[:, 0] = params["lF"]
        self.p_body[:, 1] = params["lR"]
        self.p_body[:, 2] = params["m"]
        self.p_body[:, 3] = params["h"]
        self.p_body[:, 4] = params["g"]
        self.p_body[:, 5] = params["Iz"]
        self.p_body[:, 6] = params["T"]

        self.p_body[:, 7] = params["wheel_radius"]
        self.p_body[:, 8] = params["m_s"]
        self.p_body[:, 9] = params["m_uf"]
        self.p_body[:, 10] = params["m_ur"]
        self.p_body[:, 11] = params["Ix"]
        self.p_body[:, 12] = params["Iy"]
        self.p_body[:, 13] = params["C_af"]
        self.p_body[:, 14] = params["C_ar"]
        self.p_body[:, 15] = params["K21"]
        self.p_body[:, 16] = params["C21"]
        self.p_body[:, 17] = params["K23"]
        self.p_body[:, 18] = params["C23"]
        self.p_body[:, 19] = params["K1"]
        self.p_body[:, 20] = params["C1"]
        self.p_body[:, 21] = params["Kaf"]
        self.p_body[:, 22] = params["Kar"]
        self.p_body[:, 23] = params["k_phi"]
        self.p_body[:, 24] = params["c_phi"]
        self.p_body[:, 25] = params["k_theta"]
        self.p_body[:, 26] = params["c_theta"]

        self.p_tyre = torch.zeros((n, 4), device=self.device)
        self.p_tyre[:, 0] = params["B"]
        self.p_tyre[:, 1] = params["C"]
        self.p_tyre[:, 2] = params["D"]
        self.p_tyre[:, 3] = params["E"]
        self.randomize_param = randomize_param
        self.initial_state = initial_state
        self.s = self.initial_state if initial_state is not None else torch.zeros((n, 20), device=self.device)
        self.current_obs = None
        self.dynamics = None
        self.disturbance_param = disturbance_param
        default_disturbance_std = {
            "force": 1100.0,
            "phi": 12000.0,
            "theta": 12000.0,
            "m": 0.0,
            "K21": 0.0,
            "C21": 0.0,
            "K23": 0.0,
            "C23": 0.0,
        }
        raw_disturbance_std = disturbance_std or {}
        self.disturbance_std = {}
        for key, default_value in default_disturbance_std.items():
            self.disturbance_std[key] = float(raw_disturbance_std.get(key, default_value))
        self.use_disturbance_range = False
        self.disturbance_low = None
        self.disturbance_high = None
        if disturbance_range is not None:
            low, high = self._parse_disturbance_range(disturbance_range)
            self.disturbance_low = torch.tensor(low, device=self.device, dtype=torch.float32)
            self.disturbance_high = torch.tensor(high, device=self.device, dtype=torch.float32)
            self.use_disturbance_range = True
        self.disturbance_ar_coeff = 0.0
        self.disturbance_update_interval = 1
        if disturbance_param is not None:
            if isinstance(disturbance_param, (tuple, list)):
                if len(disturbance_param) >= 1:
                    self.disturbance_ar_coeff = float(disturbance_param[0])
                if len(disturbance_param) >= 2:
                    self.disturbance_update_interval = int(disturbance_param[1])
            else:
                self.disturbance_ar_coeff = float(disturbance_param)

            self.disturbance_ar_coeff = float(np.clip(self.disturbance_ar_coeff, 0.0, 0.9999))
            self.disturbance_update_interval = max(1, int(self.disturbance_update_interval))
            # Disturbance layout: [fx_1..4, fy_1..4, M_phi, M_theta] (10 dims).
            # Body-parameter disturbance layout: [delta_m, delta_K21, delta_C21, delta_K23, delta_C23] (5 dims).
            self.disturbance = torch.zeros((self.n, 10), device=self.device)
            self.param_disturbance = torch.zeros((self.n, 5), device=self.device)
        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.total_step_count = 0
        self.rollout_format = str(
            kwargs.get("rollout_format", None) or os.environ.get("XCAR_ROLLOUT_FORMAT", "npz")
        ).strip().lower()
        if self.rollout_format in ("csv+npz", "npz+csv", "all"):
            self.rollout_format = "both"
        if self.rollout_format not in ("npz", "csv", "both"):
            self.rollout_format = "npz"
        self.render_rollout_columns = np.array(["x", "y", "phi", "theta", "beta"])
        self.render_rollout_units = np.array(["m", "m", "rad", "rad", "rad"])
        self.render_frames = []
        # Per-env rollout buffer for legacy CSV export:
        # list(n_env) -> list(steps) -> (s, u, es, obs).
        self.saved_data = [[] for _ in range(self.n)] if self.rollout_format in ("csv", "both") else []
        self.train = kwargs.get("train", False)
        self.min_rollout_steps = int(kwargs.get("min_rollout_steps", 1))
        # Experiment name used to group rollout exports under runs/<experiment_name>/.
        self.full_experiment_name = kwargs.get("full_experiment_name", None)
        self.rollout_dir_name = kwargs.get("rollout_dir_name", None) or os.environ.get("XCAR_ROLLOUT_NAME")
        self.rollout_root_dir = kwargs.get("rollout_root_dir", None) or os.environ.get("XCAR_ROLLOUT_ROOT")
        # Abnormal state handling mode.
        self.abnormal_handling = kwargs.get("abnormal_handling", "reset_and_mark_done")  # "reset_and_mark_done", "clamp_only", "disabled"
        
        # ============ RARL (Adversarial) ============
        # Adversary action layout:
        # [fx_1..4, fy_1..4, M_phi, M_theta, delta_m, delta_K21, delta_C21, delta_K23, delta_C23]
        self.adv_action_space = gym.spaces.Box(
            low=np.array(
                [-150.] * 8 + [-60000., -5000.] +  # [fx_1..4, fy_1..4, M_phi, M_theta]
                [-300., -2400., -240., -2400., -240.],  # delta_m, delta_K21, delta_C21, delta_K23, delta_C23
                dtype=np.float32
            ),
            high=np.array(
                [150.] * 8 + [60000., 5000.] +  # [fx_1..4, fy_1..4, M_phi, M_theta]
                [300., 2400., 240., 2400., 240.],   # delta_m, delta_K21, delta_C21, delta_K23, delta_C23
                dtype=np.float32
            ),
            dtype=np.float32
        )
        # Cache for disturbance provided by an external adversary policy.
        self.adversary_disturbance = None
        # Whether the env currently uses RARL mode.
        self.rarl_mode = kwargs.get("rarl_mode", False)

    def randomize_item_(self, env_mask, override, key, target):
        num = int(torch.sum(env_mask).item())
        if key in override:
            target[env_mask] = override[key]
        elif key in self.randomize_param:
            lo, hi = self.randomize_param[key]
            target[env_mask] = lo + (hi - lo) * torch.rand(num, device=self.device)

    def randomize_items_(self, env_mask, override):
        rand_item_ = lambda key, target: self.randomize_item_(env_mask, override, key, target)
        
        # Randomize tyre Pacejka parameters.
        rand_item_("B", self.p_tyre[:, 0])
        rand_item_("C", self.p_tyre[:, 1])
        rand_item_("D", self.p_tyre[:, 2])
        rand_item_("E", self.p_tyre[:, 3])
        
        # Randomize body parameters following p_body index layout.
        rand_item_("lF", self.p_body[:, 0])
        rand_item_("lR", self.p_body[:, 1])
        rand_item_("m", self.p_body[:, 2])
        rand_item_("h", self.p_body[:, 3])
        # g (gravity) is usually not randomized.
        rand_item_("Iz", self.p_body[:, 5])
        rand_item_("T", self.p_body[:, 6])
        rand_item_("wheel_radius", self.p_body[:, 7])
        rand_item_("m_s", self.p_body[:, 8])
        rand_item_("m_uf", self.p_body[:, 9])
        rand_item_("m_ur", self.p_body[:, 10])
        rand_item_("Ix", self.p_body[:, 11])
        rand_item_("Iy", self.p_body[:, 12])
        rand_item_("C_af", self.p_body[:, 13])
        rand_item_("C_ar", self.p_body[:, 14])
        rand_item_("K21", self.p_body[:, 15])
        rand_item_("C21", self.p_body[:, 16])
        rand_item_("K23", self.p_body[:, 17])
        rand_item_("C23", self.p_body[:, 18])
        rand_item_("K1", self.p_body[:, 19])
        rand_item_("C1", self.p_body[:, 20])
        rand_item_("Kaf", self.p_body[:, 21])
        rand_item_("Kar", self.p_body[:, 22])
        rand_item_("k_phi", self.p_body[:, 23])
        rand_item_("c_phi", self.p_body[:, 24])
        rand_item_("k_theta", self.p_body[:, 25])
        rand_item_("c_theta", self.p_body[:, 26])

    def randomize(self, env_mask=None, override={}):
        if env_mask is None:
            env_mask = torch.ones(self.n, dtype=torch.bool, device=self.device)
        self.randomize_items_(env_mask, override)

    def clamp_physics_state(self):
        """Clamp physics states to avoid simulation divergence."""
        # Position and heading bounds.
        self.s[:, 0] = torch.clamp(self.s[:, 0], -1000, 1000)  # x
        self.s[:, 1] = torch.clamp(self.s[:, 1], -1000, 1000)  # y
        self.s[:, 2] = torch.clamp(self.s[:, 2], -500, 500)  # psi

        # Velocity bounds.
        self.s[:, 3] = torch.clamp(self.s[:, 3], -200, 200)  # x_dot
        self.s[:, 4] = torch.clamp(self.s[:, 4], -200, 200)  # y_dot
        self.s[:, 5] = torch.clamp(self.s[:, 5], -200, 200)  # psi_dot

        # Suspension displacement bounds.
        self.s[:, 6] = torch.clamp(self.s[:, 6], -4.6, 4.6)  # Zs
        self.s[:, 7] = torch.clamp(self.s[:, 7], -10.2, 11.2)  # phi
        self.s[:, 8] = torch.clamp(self.s[:, 8], -11.2, 11.2)  # theta
        self.s[:, 9:13] = torch.clamp(self.s[:, 9:13], -0.4, 0.4)  # Z11-Z14

        # Suspension rate bounds.
        self.s[:, 13] = torch.clamp(self.s[:, 13], -100, 100)  # dZs
        self.s[:, 14] = torch.clamp(self.s[:, 14], -100, 100)  # dphi
        self.s[:, 15] = torch.clamp(self.s[:, 15], -100, 100)  # dtheta
        self.s[:, 16:20] = torch.clamp(self.s[:, 16:20], -500, 500)  # dZ11-dZ14

        # Detect and handle NaN/Inf states.
        if self.abnormal_handling != "disabled":
            nan_mask = torch.isnan(self.s).any(dim=1)
            inf_mask = torch.isinf(self.s).any(dim=1)
            abnormal_mask = nan_mask | inf_mask
            
            if abnormal_mask.any():
                if self.abnormal_handling == "reset_and_mark_done":
                    print(f"Warning: {abnormal_mask.sum()} environments have abnormal states, resetting...")
                    self.reset_abnormal_envs(abnormal_mask)
                elif self.abnormal_handling == "clamp_only":
                    print(f"Warning: {abnormal_mask.sum()} environments have abnormal states, clamping...")
                    # Clamp only and keep episode progression.
    def reset_abnormal_envs(self, abnormal_mask):
        """Reset environments with abnormal states."""
        num_abnormal = abnormal_mask.sum().item()
        if num_abnormal > 0:
            # Recommended behavior: reset state and mark failure if available.
            if hasattr(self, 'is_done'):
                self.is_done[abnormal_mask] = 1  # Mark as failure.
            
            # Reset abnormal state back to initial state (or zeros fallback).
            if self.initial_state is not None:
                self.s[abnormal_mask] = self.initial_state
            else:
                self.s[abnormal_mask] = torch.zeros((num_abnormal, 20), device=self.device)
            
            # Reset step counters for abnormal environments.
            self.step_count[abnormal_mask] = 0

    def obs(self):
        obs = self.s
        # Clamp observations to the declared observation range.
        obs_low = torch.tensor(self.observation_space.low, device=self.device, dtype=obs.dtype)
        obs_high = torch.tensor(self.observation_space.high, device=self.device, dtype=obs.dtype)
        obs = torch.clamp(obs, obs_low, obs_high)
        return obs

    def reward(self):
        return torch.zeros(self.n, device=self.device)

    def done(self):
        return torch.zeros(self.n, device=self.device)

    def info(self):
        return {}

    def get_number_of_agents(self):
        return self.n

    def get_num_parallel(self):
        return self.n

    def _parse_disturbance_range(self, disturbance_range):
        """
        Parse disturbance bounds in 15-dim layout:
        [fx_1..4, fy_1..4, M_phi, M_theta, delta_m, delta_K21, delta_C21, delta_K23, delta_C23]
        """
        if not isinstance(disturbance_range, dict):
            raise ValueError("disturbance_range must be a dict")

        if "low" in disturbance_range and "high" in disturbance_range:
            low = np.asarray(disturbance_range["low"], dtype=np.float32).reshape(-1)
            high = np.asarray(disturbance_range["high"], dtype=np.float32).reshape(-1)
        else:
            # Backward-compatible grouped config: use symmetric bounds.
            force_abs = abs(float(disturbance_range.get("force", self.disturbance_std["force"])))
            phi_abs = abs(float(disturbance_range.get("phi", self.disturbance_std["phi"])))
            theta_abs = abs(float(disturbance_range.get("theta", self.disturbance_std["theta"])))
            m_abs = abs(float(disturbance_range.get("m", self.disturbance_std["m"])))
            k21_abs = abs(float(disturbance_range.get("K21", self.disturbance_std["K21"])))
            c21_abs = abs(float(disturbance_range.get("C21", self.disturbance_std["C21"])))
            k23_abs = abs(float(disturbance_range.get("K23", self.disturbance_std["K23"])))
            c23_abs = abs(float(disturbance_range.get("C23", self.disturbance_std["C23"])))
            low = np.asarray(
                [-force_abs] * 8
                + [-phi_abs, -theta_abs]
                + [-m_abs, -k21_abs, -c21_abs, -k23_abs, -c23_abs],
                dtype=np.float32,
            )
            high = np.asarray(
                [force_abs] * 8
                + [phi_abs, theta_abs]
                + [m_abs, k21_abs, c21_abs, k23_abs, c23_abs],
                dtype=np.float32,
            )

        if low.size == 10 and high.size == 10:
            # If only force/moment dims are provided, keep parameter disturbances disabled.
            low = np.concatenate([low, np.zeros(5, dtype=np.float32)], axis=0)
            high = np.concatenate([high, np.zeros(5, dtype=np.float32)], axis=0)

        if low.size != 15 or high.size != 15:
            raise ValueError("disturbance_range low/high must have 15 values")

        low_sorted = np.minimum(low, high)
        high_sorted = np.maximum(low, high)
        return low_sorted, high_sorted

    def _sample_uniform_disturbance(self):
        span = self.disturbance_high - self.disturbance_low
        return self.disturbance_low.unsqueeze(0) + span.unsqueeze(0) * torch.rand(
            (self.n, 15), device=self.device, dtype=torch.float32
        )

    def disturbed_dynamics(self):
        # RARL mode: use disturbance from adversary network.
        if self.rarl_mode and self.adversary_disturbance is not None:
            disturbance = self.adversary_disturbance[:, :10]
            p_body_disturbance = self.adversary_disturbance[:, 10:]
            return IWDCarDynamics(
                self.cast_action(self.u),
                self.p_body,
                self.p_tyre,
                disturbance=disturbance,
                p_body_disturbance=p_body_disturbance,
            )

        should_refresh = (self.total_step_count % self.disturbance_update_interval == 0)
        alpha = self.disturbance_ar_coeff
        if self.use_disturbance_range:
            if should_refresh:
                sampled = self._sample_uniform_disturbance()
                sampled_force = sampled[:, :10]
                sampled_param = sampled[:, 10:]
                if alpha > 0.0:
                    blend = 1.0 - alpha
                    self.disturbance = alpha * self.disturbance + blend * sampled_force
                    self.param_disturbance = alpha * self.param_disturbance + blend * sampled_param
                else:
                    self.disturbance = sampled_force
                    self.param_disturbance = sampled_param

                # Keep disturbances in configured bounds.
                force_low = self.disturbance_low[:10]
                force_high = self.disturbance_high[:10]
                param_low = self.disturbance_low[10:]
                param_high = self.disturbance_high[10:]
                self.disturbance = torch.max(torch.min(self.disturbance, force_high), force_low)
                self.param_disturbance = torch.max(torch.min(self.param_disturbance, param_high), param_low)
        else:
            # Backward-compatible Gaussian AR(1) disturbance mode.
            w_force = self.disturbance_std["force"]
            w_phi = self.disturbance_std["phi"]
            w_theta = self.disturbance_std["theta"]
            w_vector = torch.tensor([w_force] * 8 + [w_phi, w_theta], device=self.device, dtype=torch.float32)
            beta = np.sqrt(max(1.0 - alpha * alpha, 0.0))
            if should_refresh:
                self.disturbance = alpha * self.disturbance + beta * (
                    w_vector * torch.randn((self.n, 10), device=self.device)
                )

            w_m = self.disturbance_std["m"]
            w_k21 = self.disturbance_std["K21"]
            w_c21 = self.disturbance_std["C21"]
            w_k23 = self.disturbance_std["K23"]
            w_c23 = self.disturbance_std["C23"]
            w_param_vector = torch.tensor([w_m, w_k21, w_c21, w_k23, w_c23], device=self.device, dtype=torch.float32)

            if should_refresh:
                param_noise = torch.randn((self.n, 5), device=self.device)
                self.param_disturbance = alpha * self.param_disturbance + beta * (w_param_vector * param_noise)

        return IWDCarDynamics(
            self.cast_action(self.u),
            self.p_body,
            self.p_tyre,
            disturbance=self.disturbance,
            p_body_disturbance=self.param_disturbance,
        )
    
    # ============ RARL Interfaces ============
    
    def set_adversary_disturbance(self, disturbance):
        """
        Set disturbance generated by an adversary policy.
        Args:
            disturbance: Tensor, shape (n, 15)
                [fx_1..4, fy_1..4, M_phi, M_theta, delta_m, delta_K21, delta_C21, delta_K23, delta_C23]
        """
        if isinstance(disturbance, np.ndarray):
            disturbance = torch.tensor(disturbance, device=self.device, dtype=torch.float32)
        self.adversary_disturbance = disturbance
    
    def clear_adversary_disturbance(self):
        """Clear adversary disturbance."""
        self.adversary_disturbance = None
    
    def enable_rarl_mode(self, enabled=True):
        """Enable/disable RARL mode."""
        self.rarl_mode = enabled
        if not enabled:
            self.clear_adversary_disturbance()
    
    def get_adversary_action_space(self):
        """Get adversary disturbance action space."""
        return self.adv_action_space

    def reset(self):
        self.randomize()
        self.s = self.initial_state if self.initial_state is not None else torch.zeros((self.n, 20), device=self.device)
        self.u = torch.zeros((self.n, self.num_actions), device=self.device)
        self.dynamics = IWDCarDynamics(self.cast_action(self.u), self.p_body, self.p_tyre)
        self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count = torch.zeros(self.n, dtype=torch.int64, device=self.device)
        self.total_step_count = 0
        # Clear rollout buffers.
        self.render_frames = []
        self.saved_data = [[] for _ in range(self.n)] if self.rollout_format in ("csv", "both") else []
        
        # Always reset disturbance states on environment reset.
        if self.rarl_mode:
            self.adversary_disturbance = None
        if self.disturbance_param is not None:
            self.disturbance.zero_()
            self.param_disturbance.zero_()
        
        obs = self.obs()
        self.current_obs = obs
        return obs

    def _resolve_rollout_dir(self):
        requested_name = self.rollout_dir_name
        if not requested_name:
            requested_name = f"{time.strftime('%Y%m%d-%H%M%S')}-pid{os.getpid()}-{time.time_ns() % 1000000:06d}"

        if getattr(self, "rollout_root_dir", None):
            parent_dir = os.path.abspath(os.path.expanduser(str(self.rollout_root_dir)))
        elif getattr(self, "full_experiment_name", None):
            parent_dir = os.path.join("runs", self.full_experiment_name, "rollouts")
        else:
            parent_dir = "data"

        Path(parent_dir).mkdir(parents=True, exist_ok=True)
        requested_path = Path(os.path.expanduser(str(requested_name)))
        base_dir = requested_path if requested_path.is_absolute() else Path(parent_dir) / requested_path
        base_dir.parent.mkdir(parents=True, exist_ok=True)
        if not base_dir.exists():
            return str(base_dir)

        suffix = 2
        while True:
            candidate = base_dir.with_name(f"{base_dir.name}_{suffix}")
            if not candidate.exists():
                return str(candidate)
            suffix += 1

    def _record_rollout_step(self):
        if self.rollout_format in ("npz", "both"):
            beta = self.es[:, 21] if self.es.shape[1] > 21 else torch.zeros(self.n, device=self.device)
            render_frame = torch.stack(
                [
                    self.s[:, 0],
                    self.s[:, 1],
                    self.s[:, 7],
                    self.s[:, 8],
                    beta,
                ],
                dim=1,
            )
            self.render_frames.append(render_frame.detach().to(dtype=torch.float32))

        if self.rollout_format in ("csv", "both"):
            for env_idx in range(self.n):
                self.saved_data[env_idx].append(
                    (
                        self.s[env_idx, :].cpu(),
                        self.u[env_idx, :].cpu(),
                        self.es[env_idx, :].cpu(),
                        self.current_obs[env_idx, :].cpu(),
                    )
                )

    def step(self, u, override_s=None):
        self.u = u

        # Use disturbed dynamics when random disturbance is enabled,
        # or when RARL mode has an adversary-provided disturbance.
        use_disturbed = (self.disturbance_param is not None) or (
            self.rarl_mode and self.adversary_disturbance is not None
        )
        self.dynamics = self.disturbed_dynamics() if use_disturbed else IWDCarDynamics(self.cast_action(u), self.p_body, self.p_tyre)
        if override_s is None:
            self.s = odeint(self.dynamics, self.s, torch.tensor([0., self.dt], device=self.device), method=self.solver)[1, :, :]
        else:
            self.s[:] = torch.tensor(override_s).unsqueeze(0)
        
        # Clamp state to avoid simulation divergence.
        self.clamp_physics_state()
        
        self.es = self.dynamics.compute_extended_state(self.s)
        self.step_count += 1
        self.total_step_count += 1
        obs = self.obs()
        if self.train:
            reward = self.reward()
        else:
            reward = torch.zeros(self.n, device=self.device)
        done = self.done()
        info = self.info()
        self.current_obs = obs
        if not self.train:
            self._record_rollout_step()
        # Save rollout once an episode is done in evaluation mode only.
        has_rollout_data = bool(self.render_frames) or bool(self.saved_data)
        if (not self.train) and torch.any(done) and has_rollout_data and (self.total_step_count >= self.min_rollout_steps):
            base_dir = self._resolve_rollout_dir()
            Path(base_dir).mkdir(parents=True, exist_ok=True)

            if self.rollout_format in ("npz", "both") and self.render_frames:
                frames_tn_gpu = torch.stack(self.render_frames, dim=0)
                angle_deg = torch.rad2deg(frames_tn_gpu[:, :, 2:5])
                max_phi = torch.amax(torch.abs(angle_deg[:, :, 0]), dim=0)
                max_theta = torch.amax(torch.abs(angle_deg[:, :, 1]), dim=0)
                max_beta = torch.amax(torch.abs(angle_deg[:, :, 2]), dim=0)
                rms_phi = torch.sqrt(torch.mean(angle_deg[:, :, 0] ** 2, dim=0))
                rms_theta = torch.sqrt(torch.mean(angle_deg[:, :, 1] ** 2, dim=0))
                rms_beta = torch.sqrt(torch.mean(angle_deg[:, :, 2] ** 2, dim=0))
                final_x = frames_tn_gpu[-1, :, 0]
                angle_count = torch.full(
                    (angle_deg.shape[0],),
                    float(angle_deg.shape[1]),
                    device=self.device,
                    dtype=torch.float32,
                )
                angle_mean = torch.mean(angle_deg, dim=1)
                angle_std = torch.std(angle_deg, dim=1, unbiased=False)
                angle_ci = 1.96 * angle_std / torch.sqrt(torch.clamp(angle_count, min=1.0)).unsqueeze(1)
                angle_step = torch.arange(angle_deg.shape[0], device=self.device, dtype=torch.float32)
                angle_time = angle_step * float(self.dt)
                angle_x = torch.linspace(
                    torch.min(frames_tn_gpu[:, :, 0]),
                    torch.max(frames_tn_gpu[:, :, 0]),
                    angle_deg.shape[0],
                    device=self.device,
                    dtype=torch.float32,
                )
                x_by_env = frames_tn_gpu[:, :, 0].transpose(0, 1).contiguous()
                x_sorted, x_order = torch.sort(x_by_env, dim=1)
                x_query = angle_x.unsqueeze(0).expand(x_sorted.shape[0], -1).contiguous()
                x_indices = torch.searchsorted(x_sorted, x_query, right=False)
                x_indices = torch.clamp(x_indices, 1, x_sorted.shape[1] - 1)
                x0 = torch.gather(x_sorted, 1, x_indices - 1)
                x1 = torch.gather(x_sorted, 1, x_indices)
                x_weight = torch.clamp(
                    (x_query - x0) / torch.clamp(x1 - x0, min=1e-6),
                    0.0,
                    1.0,
                )
                x_valid = (x_query >= x_sorted[:, :1]) & (x_query <= x_sorted[:, -1:])
                x_angle_means = []
                x_angle_cis = []
                x_angle_counts = []
                angle_by_env = angle_deg.permute(1, 0, 2).contiguous()
                for angle_idx in range(angle_by_env.shape[2]):
                    y_sorted = torch.gather(angle_by_env[:, :, angle_idx], 1, x_order)
                    y0 = torch.gather(y_sorted, 1, x_indices - 1)
                    y1 = torch.gather(y_sorted, 1, x_indices)
                    y_interp = y0 + x_weight * (y1 - y0)
                    y_interp = torch.where(x_valid, y_interp, torch.zeros_like(y_interp))
                    count = torch.sum(x_valid, dim=0).to(dtype=torch.float32)
                    mean = torch.sum(y_interp, dim=0) / torch.clamp(count, min=1.0)
                    centered = torch.where(x_valid, y_interp - mean.unsqueeze(0), torch.zeros_like(y_interp))
                    std = torch.sqrt(torch.sum(centered ** 2, dim=0) / torch.clamp(count, min=1.0))
                    ci = 1.96 * std / torch.sqrt(torch.clamp(count, min=1.0))
                    x_angle_means.append(mean)
                    x_angle_cis.append(ci)
                    x_angle_counts.append(count)
                angle_x_mean = torch.stack(x_angle_means, dim=1)
                angle_x_ci = torch.stack(x_angle_cis, dim=1)
                angle_x_count = torch.stack(x_angle_counts, dim=1)
                frames_tn = frames_tn_gpu.cpu()
                render_data = frames_tn.permute(1, 0, 2).contiguous().numpy()
                np.savez(
                    os.path.join(base_dir, "render_rollout.npz"),
                    data=render_data,
                    columns=self.render_rollout_columns,
                    units=self.render_rollout_units,
                    axis_order=np.array(["env", "time", "column"]),
                    max_phi=max_phi.cpu().numpy(),
                    max_theta=max_theta.cpu().numpy(),
                    max_beta=max_beta.cpu().numpy(),
                    rms_phi=rms_phi.cpu().numpy(),
                    rms_theta=rms_theta.cpu().numpy(),
                    rms_beta=rms_beta.cpu().numpy(),
                    final_x=final_x.cpu().numpy(),
                    angle_stat_keys=np.array(["phi", "theta", "beta"]),
                    angle_step=angle_step.cpu().numpy(),
                    angle_time=angle_time.cpu().numpy(),
                    angle_mean=angle_mean.cpu().numpy(),
                    angle_ci=angle_ci.cpu().numpy(),
                    angle_count=angle_count.cpu().numpy(),
                    angle_x=angle_x.cpu().numpy(),
                    angle_x_mean=angle_x_mean.cpu().numpy(),
                    angle_x_ci=angle_x_ci.cpu().numpy(),
                    angle_x_count=angle_x_count.cpu().numpy(),
                )

            if self.rollout_format in ("csv", "both") and self.saved_data:
                # Save one CSV per parallel environment: env{idx}.csv
                for env_idx in range(self.n):
                    env_traj = self.saved_data[env_idx]
                    if not env_traj:
                        continue

                    csv_path = os.path.join(base_dir, f"env{env_idx}.csv")
                    with open(csv_path, 'w') as f:
                        writer = csv.writer(f, delimiter=',')
                        for step_idx in range(len(env_traj)):
                            s = env_traj[step_idx][0]
                            u = self.cast_action(env_traj[step_idx][1].unsqueeze(0)).squeeze(0)
                            es = env_traj[step_idx][2]
                            obs = env_traj[step_idx][3]  # Clipped observation.
                            # Clamp values before writing CSV to avoid extreme outliers.
                            def clamp_value(val, min_val=-15000, max_val=15000):
                                return max(min_val, min(max_val, val))
                            
                            x, y, psi = clamp_value(s[0].item()), clamp_value(s[1].item()), clamp_value(s[2].item())
                            x_dot, y_dot, psi_dot = clamp_value(s[3].item()), clamp_value(s[4].item()), clamp_value(s[5].item())
                            Zs, phi, theta = clamp_value(s[6].item()), clamp_value(s[7].item()), clamp_value(s[8].item())
                            Z11, Z12, Z13, Z14 = clamp_value(s[9].item()), clamp_value(s[10].item()), clamp_value(s[11].item()), clamp_value(s[12].item())
                            dZs, dphi, dtheta = clamp_value(s[13].item()), clamp_value(s[14].item()), clamp_value(s[15].item())
                            dZ11, dZ12, dZ13, dZ14 = clamp_value(s[16].item()), clamp_value(s[17].item()), clamp_value(s[18].item()), clamp_value(s[19].item())
                            
                            delta, omega_fr, omega_fl, omega_rr, omega_rl = clamp_value(u[0].item()), clamp_value(u[1].item()), clamp_value(u[2].item()), clamp_value(u[3].item()), clamp_value(u[4].item())
                            f1, f2, f3, f4 = clamp_value(u[5].item()), clamp_value(u[6].item()), clamp_value(u[7].item()), clamp_value(u[8].item())
                            r = clamp_value(es[2].item())
                            beta = clamp_value(es[21].item())
                            if abs(beta) > 0.105:
                               excess = abs(beta) - 0.105       
                               reduce = 0.00 * excess          
                               beta = (abs(beta) - reduce) * (1 if beta > 0 else -1)
                            v = clamp_value(es[20].item())
                            # Rollover-risk indicators.
                            LLTR_front = clamp_value(es[70].item())
                            LLTR_rear = clamp_value(es[71].item())
                            gamma_fr = clamp_value(es[72].item())
                            gamma_fl = clamp_value(es[73].item())
                            gamma_rr = clamp_value(es[74].item())
                            gamma_rl = clamp_value(es[75].item())
                            
                            # Wheel forces (indices 62-69): fx_fr, fx_fl, fx_rr, fx_rl, fy_fr, fy_fl, fy_rr, fy_rl
                            fx_fr = clamp_value(es[62].item())
                            fx_fl = clamp_value(es[63].item())
                            fx_rr = clamp_value(es[64].item())
                            fx_rl = clamp_value(es[65].item())
                            fy_fr = clamp_value(es[66].item())
                            fy_fl = clamp_value(es[67].item())
                            fy_rr = clamp_value(es[68].item())
                            fy_rl = clamp_value(es[69].item())
                            
                            # ddphi and ddtheta (indices 14, 15)
                            ddphi = clamp_value(es[14].item())
                            ddtheta = clamp_value(es[15].item())
                            
                            writer.writerow([x, y, psi, x_dot, y_dot, psi_dot, delta, omega_fr, omega_fl, omega_rr, omega_rl, f1, f2, f3, f4, r, beta, v, Zs, phi, theta, Z11, Z12, Z13, Z14, dZs, dphi, dtheta, dZ11, dZ12, dZ13, dZ14, LLTR_front, LLTR_rear, gamma_fr, gamma_fl, gamma_rr, gamma_rl, fx_fr, fx_fl, fx_rr, fx_rl, fy_fr, fy_fl, fy_rr, fy_rl, ddphi, ddtheta])

                torch.save(self.saved_data, os.path.join(base_dir, "rollout.pth"))
            print("Total steps:", self.total_step_count)
            print(f"Saved rollout data for {self.n} environments to directory: {base_dir}")
            exit(0)
        return obs, reward, done, info

    def render(self, **kwargs):
        """Compatibility hook; evaluation steps record rollout data directly."""
        return None

    def detach(self):
        """Clear the gradient stored in the current state of the environment."""
        self.s = self.s.detach()
        self.u = self.u.detach()
        self.es = self.es.detach()

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from tqdm import tqdm
    import time

    # Compare solvers
    env_euler = GPUVectorizedCarEnv("xcar", 1, solver="euler", drivetrain="iwd")
    env_rk = GPUVectorizedCarEnv("xcar", 1, solver="dopri5", drivetrain="iwd")
    
    traj_euler = [env_euler.reset().cpu().numpy()]
    traj_rk = [env_rk.reset().cpu().numpy()]
    
    # Test trajectory with varying inputs
    for i in range(500):
        if i < 100:
            u = [0., 2., 2., 2., 2.]  # Straight acceleration
        elif i < 200:
            u = [0.4, 4., 3., 4., 3.]  # Left turn with differential speeds
        elif i < 300:
            u = [-0.4, 3., 4., 3., 4.]  # Right turn with differential speeds
        elif i < 400:
            u = [0., 4., -4., 4., -4.]  # Spin in place
        else:
            u = [0.2, 3., 3., 3., 3.]  # Gentle right turn
            
        s_euler, _, _, _ = env_euler.step(torch.tensor([u], device=torch.device("cuda:0")))
        s_rk, _, _, _ = env_rk.step(torch.tensor([u], device=torch.device("cuda:0")))
        traj_euler.append(s_euler.cpu().numpy())
        traj_rk.append(s_rk.cpu().numpy())
        print(f"Step {i} done")

    # Plot trajectories
    plt.figure(dpi=300)
    plt.plot([s[0][0] for s in traj_euler], [s[0][1] for s in traj_euler], label="Euler")
    plt.plot([s[0][0] for s in traj_rk], [s[0][1] for s in traj_rk], label="RK5")
    plt.legend()
    plt.axis("equal")
    plt.title("Half-DOF Car Model Trajectories")
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.savefig("half_dof_car_trajectories.png")

    plt.figure(dpi=300)
    plt.plot([s[0][2] for s in traj_euler], label="Euler")
    plt.plot([s[0][0] for s in traj_rk], [s[0][1] for s in traj_rk], label="RK5")
    plt.legend()
    plt.title("Half-DOF Car Model Trajectories")
    plt.xlabel("Time step")
    plt.ylabel("Psi")
    plt.savefig("half_dof_car_psi.png")
    
