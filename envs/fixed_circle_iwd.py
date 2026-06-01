import os
import sys
import gym
import torch
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../xcar-simulation"))
from gpu_vectorized_car_env import GPUVectorizedCarEnv

class FixedCircleIWDEnv(GPUVectorizedCarEnv):
    """Example 2IWD task of CCW circling around (0, 1) with radius 1 and sideslip angle -1."""
    def __init__(self, preset_name, n, device, **kwargs):
        super().__init__(preset_name, n, device=device, drivetrain="iwd", **kwargs)
        # 基于tesla_model_3调整最大步数
        self.max_steps = 1500  # 适应更大的车辆和更大的圆形

        self.num_states = 20
        self.num_actions = 9  # delta, omega_fr, omega_fl, omega_rr, omega_rl, f1, f2, f3, f4
        # 基于tesla_model_3调整观测空间范围center_dist, dir_diff, r, beta, v, vfx, last_delta, last_omega_fr, last_omega_fl, last_omega_rr, last_omega_rl, is_failed
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_states,))
        # 基于tesla_model_3调整动作空间范围
        # 4驱学习优化：缩小轮速范围，降低学习难度
        self.action_space = gym.spaces.Box(
            low=np.array([-0.70, 0.0, 0.0, 0.0, 0.0, -12000., -12000., -12000., -12000.]), 
            high=np.array([0.70, 70., 70., 70., 70., 12000., 12000., 12000., 12000.]), 
            shape=(9,)
        )
        self.state_space = self.observation_space

        self.need_reset = False
        self.recent_obs = [None, None]  # most recent first
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)   # 0 = not done, 1 = failed, 2 = succeeded

        self.train = kwargs.get("train", False)
        # 基于tesla_model_3调整车辆尺寸，圆形半径相应调整
        self.radius = 12.  # 从1m调整刐8m，适应Tesla Model 3的尺寸

        # 初始化奖励组件字典
        self.reward_components = {}


    def update_recent_obs(self, obs):
        self.recent_obs[1] = self.recent_obs[0]
        self.recent_obs[0] = obs

    def obs(self):
        x = self.s[:, 0]
        y = self.s[:, 1]
        psi = self.s[:, 2]
        xd = self.s[:, 3]
        yd = self.s[:, 4]
        psid = self.s[:, 5]
        center_dist = torch.hypot(x, y - self.radius)  # 车辆到圆心(0,8)的距离 (m)
        phase = torch.atan2(y - self.radius, x)  # 车辆与圆心连线的角度 (rad)
        tangent_dir = phase + torch.pi / 2  # 圆上最近点的切线方向 (rad)
        tangent_dir = torch.atan2(torch.sin(tangent_dir), torch.cos(tangent_dir))  # 将角度归一化到[-π,π]区间
        veldir = torch.atan2(yd, xd)  # 车辆速度向量的方向 (rad)
        dir_diff = tangent_dir - veldir  # 切线方向与速度方向之间的差异 (rad)
        dir_diff = torch.atan2(torch.sin(dir_diff), torch.cos(dir_diff))  # 将角度差异归一化到[-π,π]区间

        psid = self.es[:, 2]
        beta = self.es[:, 21]
        v = self.es[:, 20]
        vfx = (self.es[:, 22] + self.es[:, 23]) / 2

        # 调整失败条件 - 参考成功配置的严格程度，但适应大车特性
        dir_diff_fail = torch.abs(dir_diff) > 1.5
        center_dist_fail = torch.abs(center_dist - self.radius) > 4
        failed = ((torch.abs(dir_diff) > 1.5) & (v > 2)) | (torch.abs(center_dist - self.radius) > 4)

        # 在测试模式下打印失败原因
        if not self.train and failed.any():
            failed_indices = torch.where(failed)[0]
            for idx in failed_indices:
                idx_val = idx.item()
                reasons = []
                if dir_diff_fail[idx_val]:
                    reasons.append(f"dir_diff过大: {dir_diff[idx_val].item():.4f} (阈值: 1.5)")
                if center_dist_fail[idx_val]:
                    dist_error = torch.abs(center_dist[idx_val] - self.radius).item()
                    reasons.append(f"center_dist偏差过大: {dist_error:.4f} (阈值: 4), center_dist: {center_dist[idx_val].item():.4f}, radius: {self.radius:.4f}")
                print(f"[Test Failed] Env {idx_val} - 失败原因: {', '.join(reasons)}")
                print(f"  当前状态: x={x[idx_val].item():.4f}, y={y[idx_val].item():.4f}, psi={psi[idx_val].item():.4f}, v={v[idx_val].item():.4f}")

        # Success when reaching max_steps
        succeeded = (self.step_count >= self.max_steps)

        self.is_done[failed] = 1
        self.is_done[succeeded] = 2

        last_delta = self.u[:, 0]
        last_omega_fr = self.u[:, 1]
        last_omega_fl = self.u[:, 2]
        last_omega_rr = self.u[:, 3]
        last_omega_rl = self.u[:, 4]

        # 添加稳定性观测量
        Zs = self.s[:, 6]      # 垂直位移
        phi = self.s[:, 7]     # 横滚角
        theta = self.s[:, 8]   # 俯仰角
        dZs = self.s[:, 13]    # 垂直速度
        dphi = self.s[:, 14]   # 横滚角速度
        dtheta = self.s[:, 15] # 俯仰角速度
        
        # 轮胎载荷
        Z11 = self.s[:, 9]   # 左前轮载荷
        Z12 = self.s[:, 10]  # 右前轮载荷
        Z13 = self.s[:, 11]  # 左后轮载荷
        Z14 = self.s[:, 12]  # 右后轮载荷
        
        # 载荷平衡指标
        load_balance_lr = (Z11 + Z13) - (Z12 + Z14)  # 左右载荷差
        load_balance_fr = (Z11 + Z12) - (Z13 + Z14)  # 前后载荷差

        center_dist, dir_diff, psid, beta, v, vfx, last_delta, last_omega_fr, last_omega_fl, last_omega_rr, last_omega_rl, Zs, phi, theta, dZs, dphi, dtheta, load_balance_lr, load_balance_fr, is_failed = map(lambda t: torch.unsqueeze(t, 1), [center_dist, dir_diff, psid, beta, v, vfx, last_delta, last_omega_fr, last_omega_fl, last_omega_rr, last_omega_rl, Zs, phi, theta, dZs, dphi, dtheta, load_balance_lr, load_balance_fr, failed.to(dtype=torch.float32)])
        obs = torch.cat([center_dist, dir_diff, psid, beta, v, vfx, last_delta, last_omega_fr, last_omega_fl, last_omega_rr, last_omega_rl, Zs, phi, theta, dZs, dphi, dtheta, load_balance_lr, load_balance_fr, is_failed], 1)



        self.update_recent_obs(obs)

        return obs

    def reward(self):
        obs = self.recent_obs[0]
        last_obs = self.recent_obs[1] if self.recent_obs[1] is not None else torch.zeros_like(obs)

        # Extracting values from obs
        center_dist = obs[:, 0]
        dir_diff = obs[:, 1]
        psid = obs[:, 2]
        beta = obs[:, 3]
        v = obs[:, 4]
        vfx = obs[:, 5]
        delta = obs[:, 6]
        omega_fr = obs[:, 7]
        omega_fl = obs[:, 8]
        omega_rr = obs[:, 9]
        omega_rl = obs[:, 10]
        Zs = obs[:, 11]
        phi = obs[:, 12]
        theta = obs[:, 13]
        dZs = obs[:, 14]
        is_failed = obs[:, 19]

        # Last timestep values
        last_delta = last_obs[:, 6]
        last_omega_fr = last_obs[:, 7]
        last_omega_fl = last_obs[:, 8]
        last_omega_rr = last_obs[:, 9]
        last_omega_rl = last_obs[:, 10]

        # Reward components - 基于Tesla Model 3调整
        rew_beta = torch.where(
          (beta >= -1.0) & (beta <= -0.4),
          torch.zeros_like(beta),
          torch.where(
            beta > -0.4,
            -(beta + 0.4) ** 2,
            -(beta + 1.0) ** 2
          )
        )


        
        # 调整中心距离奖励的权重，因为圆形半径从1m增加到10m
        rew_center_dist = -(center_dist - self.radius) ** 2 if self.recent_obs[1] is not None else torch.zeros_like(rew_beta)
        rew_dir_diff = -dir_diff ** 2 if self.recent_obs[1] is not None else torch.zeros_like(rew_beta)
        
        # 调整平滑性奖励，适应更大的车辆和更高的速度
        omega_avg = (omega_fr + omega_fl + omega_rr + omega_rl) / 4
        last_omega_avg = (last_omega_fr + last_omega_fl + last_omega_rr + last_omega_rl) / 4
        # 4驱学习优化：增强平滑性约束
        rew_smooth = -((delta - last_delta) ** 2 + 0.005 * ((omega_fr - last_omega_fr) ** 2 + (omega_fl - last_omega_fl) ** 2 + (omega_rr - last_omega_rr) ** 2 + (omega_rl - last_omega_rl) ** 2)) if self.recent_obs[1] is not None else torch.zeros_like(rew_beta)
        rew_sf = -(vfx - omega_avg) ** 2 if self.recent_obs[1] is not None else torch.zeros_like(rew_beta)
        
        # 漂移专用：鼓励合理的轮速差异和打滑
        if self.recent_obs[1] is not None:
            # 1. 鼓励后轮输出功率（漂移需要后轮驱动）
            rear_power_bonus = 0.05 * (omega_rr + omega_rl)  # 鼓励后轮输出
            # 2. 防止极端轮速差异（保持基本控制）
            extreme_diff_penalty = -0.02 * (torch.abs(omega_fr - omega_fl) > 8).float() * (torch.abs(omega_fr - omega_fl) - 8) ** 2
            extreme_diff_penalty += -0.02 * (torch.abs(omega_rr - omega_rl) > 8).float() * (torch.abs(omega_rr - omega_rl) - 8) ** 2
            # 3. 防止单轮输出过度（避免损坏）
            single_wheel_limit = -0.01 * torch.sum(
                (torch.abs(torch.stack([omega_fr, omega_fl, omega_rr, omega_rl], dim=1)) > 18).float() * 
                (torch.abs(torch.stack([omega_fr, omega_fl, omega_rr, omega_rl], dim=1)) - 18) ** 2, dim=1)
            rew_drift_control = rear_power_bonus + extreme_diff_penalty + single_wheel_limit
        else:
            rew_drift_control = torch.zeros_like(rew_beta)

        # 速度奖励：鼓励适合漂移的速度范围（4-8 m/s）
        target_speed = 12  # 目标速度 6 m/s
        rew_speed = -((v - target_speed) ** 2) / 6  # 速度偏离目标的惩罚
        rew_final = -is_failed  # 失败惩罚
        
        # 收敛奖励：Zs收敛到-0.19，dZs收敛到0，phi收敛到-0.01，theta收敛到0
        rew_Zs_conv = -((Zs - (-0.23)) ** 2)  # Zs收敛到-0.19（悬挂平衡位置）
        rew_dZs_conv = -(dZs ** 2)  # dZs收敛到0
        rew_phi_conv = -(phi ** 2)  # phi收敛到-0.01
        rew_theta_conv = -(theta ** 2)  # theta收敛到0
        
        # 增加theta角变化率惩罚，促进俯仰角稳定
        if self.recent_obs[1] is not None:
            last_theta = last_obs[:, 13]
            theta_change_rate = torch.abs(theta - last_theta)
            rew_theta_stability = -5.0 * (theta_change_rate ** 2)  # 惩罚theta角快速变化
        else:
            rew_theta_stability = torch.zeros_like(rew_theta_conv)

        
        # Tesla真实车辆权重：重点调整相对重要性
        survival_bonus = 2.5   # 大幅降低存活奖励，避免过度保守策略
        
        # 漂移专用奖励结构：鼓励打滑而非约束打滑
        rew = ( survival_bonus + 
               2. * rew_beta +      # 提高侧滑角奖励（漂移关键）
               0.1 * rew_smooth +   # 降低平滑性约束
               0.3 * rew_speed +    # 提高速度奖励
               0.0002 * rew_sf +      # 大幅降低滑移约束（允许打滑）
               2.5 * rew_center_dist +
               1.7 * rew_dir_diff +
               0.1 * rew_drift_control +  # 漂移专用控制
               95. * rew_final +
               1.6 * rew_Zs_conv +   # Zs悬挂位置奖励
               1.2 * rew_dZs_conv +  # dZs垂直速度奖励
               10.6 * rew_phi_conv +
               60.0 * rew_theta_conv +  # 大幅增强theta角稳定奖励
               rew_theta_stability)    # 新增theta角变化率惩罚

        # 保存各个奖励组件用于TensorBoard记录（漂移专用权重）
        self.reward_components = {
            'survival_bonus': survival_bonus,
            'rew_beta': 2. * rew_beta,         # 侧滑角权重
            'rew_smooth': 0.1 * rew_smooth,    # 降低平滑性权重
            'rew_speed': 0.3 * rew_speed,
            'rew_sf': 0.0002 * rew_sf,           # 降低滑移约束
            'rew_center_dist': 2.5 * rew_center_dist,
            'rew_dir_diff': 1.7 * rew_dir_diff,
            'rew_drift_control': 0.1 * rew_drift_control,  # 漂移控制
            'rew_final': 95. * rew_final,
            'rew_Zs_conv': 1.6 * rew_Zs_conv,   # Zs悬挂位置奖励
            'rew_dZs_conv': 1.2 * rew_dZs_conv,
            'rew_phi_conv': 10.6 * rew_phi_conv,
            'rew_theta_conv': 60.0 * rew_theta_conv,  # 大幅增强theta角稳定奖励
            'rew_theta_stability': rew_theta_stability  # 新增theta角变化率惩罚
        }

        return rew

    def done(self):
        return self.is_done

    def reset(self):
        super().reset()
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.s[:, :] = self._gen_random_state(self.n)
        self.s[:, 6] = -0.23
        # 初始化悬架力历史
        self.last_forces = torch.zeros((self.n, 4), device=self.device)
        return self.obs()

    def info(self):
        info_dict = {
            "time_outs": (self.is_done == 2)
        }
        
        # 添加奖励组件信息
        if hasattr(self, 'reward_components') and self.reward_components:
            for key, value in self.reward_components.items():
                info_dict[f'reward_{key}'] = value
                
        return info_dict

    def reset_done_envs(self):
        """Only reset envs that are already done."""
        is_done = self.is_done.bool()
        size = torch.sum(is_done)
        self.step_count[is_done] = 0
        self.s[is_done, :] = 0
        self.u[is_done, :] = 0
        self.s[is_done, :] = self._gen_random_state(int(size.item()))
        self.is_done[:] = 0

    def _gen_random_state(self, size: int) -> torch.Tensor:
        x = torch.zeros((size, 1), device=self.device)
        y = 1.0 * (torch.rand((size, 1), device=self.device) - 0.5)
        psi = 0.2 * torch.randn((size, 1), device=self.device)

        random_indices = torch.bernoulli(1.0 * torch.ones((size, 1), device=self.device))
        random_r = (0.5 * torch.rand((size, 1), device=self.device) + 0.2) * random_indices
        random_beta = (-0.1 * torch.rand((size, 1), device=self.device) - 0.1) * random_indices
        random_V = (0.5 * torch.rand((size, 1), device=self.device) + 0.5) * random_indices

        xd = random_V * torch.cos(random_beta + psi)
        yd = random_V * torch.sin(random_beta + psi)
        psid = random_r

        Zs = -0.2 * (2 * torch.rand((size, 1), device=self.device) - 1)
        phi = 0.01 * (2 * torch.rand((size, 1), device=self.device) - 1)
        theta = 0.01 * (2 * torch.rand((size, 1), device=self.device) - 1)
        Z11 = 0.005 * (2 * torch.rand((size, 1), device=self.device) - 1)
        Z12 = 0.005 * (2 * torch.rand((size, 1), device=self.device) - 1)
        Z13 = 0.005 * (2 * torch.rand((size, 1), device=self.device) - 1)
        Z14 = 0.005 * (2 * torch.rand((size, 1), device=self.device) - 1)
        dZs = 0.05 * (2 * torch.rand((size, 1), device=self.device) - 1)
        dphi = 0.05 * (2 * torch.rand((size, 1), device=self.device) - 1)
        dtheta = 0.05 * (2 * torch.rand((size, 1), device=self.device) - 1)
        dZ11 = 0.05 * (2 * torch.rand((size, 1), device=self.device) - 1)
        dZ12 = 0.05 * (2 * torch.rand((size, 1), device=self.device) - 1)
        dZ13 = 0.05 * (2 * torch.rand((size, 1), device=self.device) - 1)
        dZ14 = 0.05 * (2 * torch.rand((size, 1), device=self.device) - 1)

        return torch.cat([x, y, psi, xd, yd, psid, Zs, phi, theta, Z11, Z12, Z13, Z14, dZs, dphi, dtheta, dZ11, dZ12, dZ13, dZ14], 1)

    def step(self, action, **kwargs):
        self.reset_done_envs()
        obs, reward, done, info = super().step(action, **kwargs)
        return obs, reward, done, info
