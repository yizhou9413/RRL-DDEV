import os
import sys
import gym
import torch
import numpy as np
file_path = os.path.dirname(__file__)
sys.path.append(os.path.join(file_path, "../xcar-simulation"))
from gpu_vectorized_car_env import GPUVectorizedCarEnv
from IWDCarDynamics import IWDCarDynamics

class MooseIWDEnv(GPUVectorizedCarEnv):
    """沿轨迹行驶"""
    def __init__(self, preset_name, n, device, **kwargs):
        # 强制使用euler求解器（最快，对GPU友好）
       
        super().__init__(preset_name, n, device=device, drivetrain="iwd", **kwargs)
        self.max_steps = 500  
        
        # 观测空间
        # 新观测空间：轨迹跟踪相关信息 + 侧翻风险指标
        # 0: 横向误差 (y_current - y_ref)
        # 1: 航向角偏差 (psi_current - psi_ref)
        # 2-3: 前方1米参考点 (y_ref, psi_ref)
        # 4-5: 前方3米参考点 (y_ref, psi_ref)
        # 6-7: 前方5米参考点 (y_ref, psi_ref)
        # 8: 纵向速度 xd
        # 9: 横向速度 yd
        # 10: 横摆角速度 psid
        # 11: 速度 v
        # 12: 侧滑角 beta
        # 13: 方向盘角度 delta
        # 14-17: 车轮角速度 (omega_fr, omega_fl, omega_rr, omega_rl)
        # 18-21: 悬架力 (f1, f2, f3, f4)
        # 22: 悬架垂直位移 Zs
        # 23: 横滚角 phi
        # 24: 俯仰角 theta
        # 25: 垂直速度 dZs
        # 26: 横滚角速度 dphi
        # 27: 俯仰角速度 dtheta
        # 28: 前轴载荷转移率 LLTR_front
        # 29: 后轴载荷转移率 LLTR_rear
        # 30-33: 轮胎摩擦饱和率 (gamma_fr, gamma_fl, gamma_rr, gamma_rl)
        self.num_states = 32
        self.num_actions = 9
        
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.num_states,)
        )
        
        # 动作空间：直接使用物理范围（不归一化）
        self.action_space = gym.spaces.Box(
            low=np.array([-0.70, 0.0, 0.0, 0.0, 0.0, -12000., -12000., -12000., -12000.]), 
            high=np.array([0.70, 100., 100., 100., 100., 12000., 12000., 12000., 12000.]), 
            shape=(9,)
        )
        self.state_space = self.observation_space

        self.need_reset = False
        self.recent_obs = None
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        self.train = kwargs.get("train", False)
        self.obs_noise = kwargs.get("obs_noise", False)
        
        # 时间参数
        self.dt = kwargs.get('dt', 0.01)
        
        # 奖励记录
        self.reward_components = {}
        
        # 生成完美轨迹
        self.perfect_trajectory = self._generate_perfect_trajectory()

    def _generate_perfect_trajectory(self):
        """
        生成一条完美的轨迹：三通道轨迹（A道-弯道-B道-弯道-C道）
        
        轨迹设计：
        1. A道：12m直线，y=0
        2. A到B弯道：13.5m，y从0变到4
        3. B道：11m直线，y=4
        4. B到C弯道：12.5m，y从4变回0
        5. C道：12m直线，y=0
        
        轨迹包含：
        - 位置和姿态：x, y, psi (世界坐标系)
        - 速度：xd, yd, psid
        - 悬架状态：Zs, phi, theta, dZs, dphi, dtheta
        - 动作：delta, omega_fr, omega_fl, omega_rr, omega_rl, f1, f2, f3, f4
        
        坐标系说明：
        - 世界坐标系：x 向前（车辆初始方向），y 向左
        - psi：航向角，从 +x 轴逆时针旋转的角度
        - psi = 0 表示车辆朝向 +x 方向
        
        Returns:
            trajectory: dict with keys 'states' and 'actions'
                states: tensor of shape (num_steps, 16)  # 状态维度
                actions: tensor of shape (num_steps, 9)  # 动作维度
        """
        num_steps = self.max_steps
        trajectory_states = torch.zeros(num_steps, 16, device=self.device)
        trajectory_actions = torch.zeros(num_steps, 9, device=self.device)
        
        # 目标速度
        target_speed = 20.0  # m/s
        
        # 轨迹参数（五段）
        lane_a_length = 22.0      # A道长度 (m)
        curve_ab_length = 18    # A到B弯道长度 (m)
        lane_b_length = 11.0      # B道长度 (m)
        curve_bc_length = 16   # B到C弯道长度 (m)
        lane_c_length = 12.0      # C道长度 (m)
        
        lane_b_offset = 3.2      # B道的y偏移量 (m)
        
        # 计算各段的结束位置
        x1_end = lane_a_length
        x2_end = x1_end + curve_ab_length
        x3_end = x2_end + lane_b_length
        x4_end = x3_end + curve_bc_length
        x5_end = x4_end + lane_c_length
        total_length = x5_end
        
        pi = 3.14159265359
        
        for t in range(num_steps):
            time_sec = t * self.dt
            
            # 当前纵向位置（假设匀速运动）
            x_current = target_speed * time_sec
            
            # 根据当前位置判断处于哪个阶段
            if x_current <= x1_end:
                # === 第一段：A道直线 ===
                x = x_current
                y = 0.0
                psi = 0.0
                xd = target_speed
                yd = 0.0
                psid = 0.0
                target_delta = 0.0
                
            elif x_current <= x2_end:
                # === 第二段：A到B弯道 ===
                s = x_current - x1_end
                progress = s / curve_ab_length  # 0 到 1
                
                # 使用余弦插值实现平滑过渡（y从0到lane_b_offset）
                # y = lane_b_offset * (1 - cos(π * progress)) / 2
                y_offset = lane_b_offset * (1 - torch.cos(torch.tensor(pi * progress, device=self.device))) / 2
                dy_ds = lane_b_offset * pi / curve_ab_length * torch.sin(torch.tensor(pi * progress, device=self.device))
                
                x = x_current
                y = y_offset.item()
                
                # 计算切向角
                dx_ds = 1.0
                psi = torch.atan2(dy_ds, torch.tensor(dx_ds, device=self.device)).item()
                
                # 速度分解
                xd = dx_ds * target_speed
                yd = dy_ds.item() * target_speed
                
                # 横摆角速度
                d2y_ds2 = lane_b_offset * (pi / curve_ab_length)**2 * torch.cos(torch.tensor(pi * progress, device=self.device))
                dpsi_ds = d2y_ds2 / (1 + dy_ds**2)
                psid = (dpsi_ds * target_speed).item()
                
                # 方向盘角度
                wheelbase = 3.0
                target_delta = (dpsi_ds * wheelbase).item()
                target_delta = torch.clamp(torch.tensor(target_delta, device=self.device), -0.5, 0.5).item()
                
            elif x_current <= x3_end:
                # === 第三段：B道直线 ===
                x = x_current
                y = lane_b_offset
                psi = 0.0
                xd = target_speed
                yd = 0.0
                psid = 0.0
                target_delta = 0.0
                
            elif x_current <= x4_end:
                # === 第四段：B到C弯道 ===
                s = x_current - x3_end
                progress = s / curve_bc_length  # 0 到 1
                
                # 使用余弦插值实现平滑过渡（y从lane_b_offset回到0）
                # y = lane_b_offset * (1 + cos(π * progress)) / 2
                y_offset = lane_b_offset * (1 + torch.cos(torch.tensor(pi * progress, device=self.device))) / 2
                dy_ds = -lane_b_offset * pi / curve_bc_length * torch.sin(torch.tensor(pi * progress, device=self.device))
                
                x = x_current
                y = y_offset.item()
                
                # 计算切向角
                dx_ds = 1.0
                psi = torch.atan2(dy_ds, torch.tensor(dx_ds, device=self.device)).item()
                
                # 速度分解
                xd = dx_ds * target_speed
                yd = dy_ds.item() * target_speed
                
                # 横摆角速度
                d2y_ds2 = -lane_b_offset * (pi / curve_bc_length)**2 * torch.cos(torch.tensor(pi * progress, device=self.device))
                dpsi_ds = d2y_ds2 / (1 + dy_ds**2)
                psid = (dpsi_ds * target_speed).item()
                
                # 方向盘角度
                wheelbase = 3.0
                target_delta = (dpsi_ds * wheelbase).item()
                target_delta = torch.clamp(torch.tensor(target_delta, device=self.device), -0.5, 0.5).item()
                
            else:
                # === 第五段：C道直线 ===
                x = x_current
                y = 0.0
                psi = 0.0
                xd = target_speed
                yd = 0.0
                psid = 0.0
                target_delta = 0.0
            
            # === 填充状态 ===
            trajectory_states[t, 0] = x
            trajectory_states[t, 1] = y
            trajectory_states[t, 2] = psi  # 航向角（世界坐标系）
            trajectory_states[t, 3] = xd   # x方向速度（世界坐标系）
            trajectory_states[t, 4] = yd   # y方向速度（世界坐标系）
            trajectory_states[t, 5] = psid # 横摆角速度
            trajectory_states[t, 6] = -0.23  # Zs：标准悬架压缩量
            trajectory_states[t, 7] = 0.0  # phi：横滚角
            trajectory_states[t, 8] = 0.0  # theta：俯仰角
            trajectory_states[t, 13] = 0.0  # dZs
            trajectory_states[t, 14] = 0.0  # dphi
            trajectory_states[t, 15] = 0.0  # dtheta
            
            # === 填充动作 ===
            trajectory_actions[t, 0] = target_delta  # delta：方向盘角度
            
            # 车轮角速度（根据速度计算）
            # 假设轮胎半径约0.3m，omega = v / r
            wheel_radius = 0.33
            wheel_omega = target_speed / wheel_radius
            trajectory_actions[t, 1] = wheel_omega  # omega_fr
            trajectory_actions[t, 2] = wheel_omega  # omega_fl
            trajectory_actions[t, 3] = wheel_omega  # omega_rr
            trajectory_actions[t, 4] = wheel_omega  # omega_rl
            
            # 悬架力：保持平衡（接近0）
            trajectory_actions[t, 5] = 0.0  # f1
            trajectory_actions[t, 6] = 0.0  # f2
            trajectory_actions[t, 7] = 0.0  # f3
            trajectory_actions[t, 8] = 0.0  # f4
        
        return {
            'states': trajectory_states,
            'actions': trajectory_actions,
            'num_steps': num_steps
        }

    def get_trajectory_reference(self, x_positions):
        """
        根据x位置查找轨迹参考点（向量化版本，GPU并行）
        
        Args:
            x_positions: tensor of shape (n,) - 要查询的x坐标
        
        Returns:
            dict with keys:
                'y': tensor of shape (n,) - 参考y坐标
                'psi': tensor of shape (n,) - 参考航向角
                'xd': tensor of shape (n,) - 参考x方向速度
                'yd': tensor of shape (n,) - 参考y方向速度
        """
        traj_states = self.perfect_trajectory['states']
        traj_x = traj_states[:, 0].contiguous()
        
        # 限制在轨迹范围内
        xq = torch.clamp(x_positions, traj_x[0], traj_x[-1])
        
        # 使用 searchsorted 找到右侧索引，再回退一个得到左侧索引
        idx2 = torch.searchsorted(traj_x, xq, right=True)
        idx2 = torch.clamp(idx2, 1, traj_x.shape[0] - 1)
        idx1 = idx2 - 1
        
        x1 = traj_x[idx1]
        x2 = traj_x[idx2]
        denom = torch.clamp(x2 - x1, min=1e-6)
        alpha = ((xq - x1) / denom).unsqueeze(1)  # (n,1)
        
        s1 = traj_states[idx1]
        s2 = traj_states[idx2]
        interp = (1 - alpha) * s1 + alpha * s2
        
        return {
            'y': interp[:, 1],
            'psi': interp[:, 2],
            'xd': interp[:, 3],
            'yd': interp[:, 4]
        }

    def get_trajectory_reference_multi(self, x_positions, offsets):
        """
        一次性查询多个前视距离的轨迹参考。
        Args:
            x_positions: (n,)
            offsets: 序列或 tensor，长度为k，例如 [0,1,3,5]
        Returns:
            dict: 每个键对应形状 (n,k) 的张量
        """
        traj_states = self.perfect_trajectory['states']
        traj_x = traj_states[:, 0].contiguous()
        if not torch.is_tensor(offsets):
            offsets = torch.tensor(offsets, device=x_positions.device, dtype=x_positions.dtype)
        offsets = offsets.view(1, -1)
        xq = x_positions.view(-1, 1) + offsets  # (n,k)
        xq = torch.clamp(xq, traj_x[0], traj_x[-1])
        
        # 展平做一次 searchsorted
        xq_flat = xq.reshape(-1)
        idx2 = torch.searchsorted(traj_x, xq_flat, right=True)
        idx2 = torch.clamp(idx2, 1, traj_x.shape[0] - 1)
        idx1 = idx2 - 1
        
        x1 = traj_x[idx1]
        x2 = traj_x[idx2]
        denom = torch.clamp(x2 - x1, min=1e-6)
        alpha = ((xq_flat - x1) / denom).unsqueeze(1)  # (n*k,1)
        
        s1 = traj_states[idx1]
        s2 = traj_states[idx2]
        interp = ((1 - alpha) * s1 + alpha * s2).reshape(xq.shape[0], xq.shape[1], -1)  # (n,k,16)
        
        return {
            'y': interp[:, :, 1],
            'psi': interp[:, :, 2],
            'xd': interp[:, :, 3],
            'yd': interp[:, :, 4]
        }


    def obs(self):
        """观测空间：轨迹跟踪相关信息"""
        # 基础状态
        x = self.s[:, 0]
        y = self.s[:, 1]
        psi = self.s[:, 2]
        xd = self.s[:, 3]
        yd = self.s[:, 4]
        psid = self.s[:, 5]
        Zs = self.s[:, 6]  # 垂直位移
        phi = self.s[:, 7]  # 横滚角
        theta = self.s[:, 8]  # 俯仰角
        dZs = self.s[:, 13]  # 垂直速度
        dphi = self.s[:, 14]  # 横滚角速度
        dtheta = self.s[:, 15]  # 俯仰角速度
        
        # 扩展状态
        v = self.es[:, 20]  # 速度
        beta = self.es[:, 21]  # 侧滑角
        
        # 侧翻风险指标
        LLTR_front = self.es[:, 70]  # 前轴载荷转移率
        LLTR_rear = self.es[:, 71]   # 后轴载荷转移率
        gamma_fr = self.es[:, 72]    # 右前轮摩擦饱和率
        gamma_fl = self.es[:, 73]    # 左前轮摩擦饱和率
        gamma_rr = self.es[:, 74]    # 右后轮摩擦饱和率
        gamma_rl = self.es[:, 75]    # 左后轮摩擦饱和率
        
        # 车轮角速度
        omega_fr = self.es[:, 6]  # 右前轮
        omega_fl = self.es[:, 7]  # 左前轮
        omega_rr = self.es[:, 8]  # 右后轮
        omega_rl = self.es[:, 9]  # 左后轮
        
        # 上一时刻动作
        delta = self.u[:, 0]  # 方向盘角度
        
        # 悬架力
        f1 = self.u[:, 5]  # 悬架力1
        f2 = self.u[:, 6]  # 悬架力2
        f3 = self.u[:, 7]  # 悬架力3
        f4 = self.u[:, 8]  # 悬架力4
        
        # === 获取轨迹参考点（合并查询，避免重复扫描） ===
        with torch.no_grad():
            refs = self.get_trajectory_reference_multi(x, [0.0, 1.0, 3.0, 5.0])
        y_ref = refs['y'][:, 0]
        psi_ref = refs['psi'][:, 0]
        y_ref_1m = refs['y'][:, 1]
        psi_ref_1m = refs['psi'][:, 1]
        y_ref_3m = refs['y'][:, 2]
        psi_ref_3m = refs['psi'][:, 2]
        y_ref_5m = refs['y'][:, 3]
        psi_ref_5m = refs['psi'][:, 3]
        
        # 计算误差
        lateral_error = y - y_ref
        psi_v = torch.atan2(yd, xd + 1e-8)
        heading_error = psi_v - psi_ref
        # 将航向角误差归一化到 [-π, π]
        heading_error = torch.atan2(torch.sin(heading_error), torch.cos(heading_error))
        
        # 失败条件：偏离轨迹太远或姿态不稳定或侧翻风险过高
        failed = (
            (torch.abs(lateral_error) > 1.9) |  # 横向偏离超过2米
            (torch.abs(heading_error) > 0.80) |  # 航向角偏差超过36度
            (torch.abs(phi) > 0.8) |  # 横滚角超过46度
            (v < 10.0) |  # 速度过低
            (torch.abs(theta) > 0.8)   # 俯仰角过大
    )
        
        # 成功条件：到达轨迹终点附近
        traj_length = self.perfect_trajectory['states'][-1, 0]
        succeeded = (x >= traj_length - 3.0) & (torch.abs(lateral_error) < 0.5) & (torch.abs(heading_error) < 0.2)
        
        self.is_done[failed] = 1
        self.is_done[succeeded] = 2
        
        # 组装观测（34维）- 不进行归一化
        obs = torch.stack([
            lateral_error,  # 0. 横向误差 (m)
            heading_error,  # 1. 航向角偏差 (rad)
            y_ref_1m,  # 2. 前方1米参考y (m)
            psi_ref_1m,  # 3. 前方1米参考psi (rad)
            y_ref_3m,  # 4. 前方3米参考y (m)
            psi_ref_3m,  # 5. 前方3米参考psi (rad)
            y_ref_5m,  # 6. 前方5米参考y (m)
            psi_ref_5m,  # 7. 前方5米参考psi (rad)
            xd,  # 8. 纵向速度 (m/s)
            yd,  # 9. 横向速度 (m/s)
            psid,  # 10. 横摆角速度 (rad/s)
            v,  # 11. 速度 (m/s)
            beta,  # 12. 侧滑角 (rad)
            delta,  # 13. 方向盘角度 (rad)
            omega_fr,  # 14. 右前轮角速度 (rad/s)
            omega_fl,  # 15. 左前轮角速度 (rad/s)
            omega_rr,  # 16. 右后轮角速度 (rad/s)
            omega_rl,  # 17. 左后轮角速度 (rad/s)
            f1,  # 18. 悬架力1 (N)
            f2,  # 19. 悬架力2 (N)
            f3,  # 20. 悬架力3 (N)
            f4,  # 21. 悬架力4 (N)
            Zs,  # 22. 垂直位移 (m)
            phi,  # 23. 横滚角 (rad)
            theta,  # 24. 俯仰角 (rad)
            dZs,  # 25. 垂直速度 (m/s)
            dphi,  # 26. 横滚角速度 (rad/s)
            dtheta,  # 27. 俯仰角速度 (rad/s)
            LLTR_front,  # 28. 前轴载荷转移率 [-1, 1]
            LLTR_rear,   # 29. 后轴载荷转移率 [-1, 1]
            y, 
            x,  # 33. 左后轮摩擦饱和率 [0, ~2]
        ], dim=1)
        
        self.recent_obs = obs
        # 每一维观测对应一个噪声比例（百分比）
        obs_noise_scale = torch.tensor([
            0.05,  # 0  lateral_error
            0.13,  # 1  heading_error
            0.02,  # 2  y_ref_1m
            0.02,  # 3  psi_ref_1m
            0.02,  # 4  y_ref_3m
            0.02,  # 5  psi_ref_3m
            0.02,  # 6  y_ref_5m
            0.02,  # 7  psi_ref_5m
            0.15,  # 8  xd
            0.15,  # 9  yd
            0.05,  # 10 psid
            0.01,  # 11 v
            0.33,  # 12 beta
            0.02,  # 13 delta
            0.10,  # 14 omega_fr
            0.10,  # 15 omega_fl
            0.10,  # 16 omega_rr
            0.10,  # 17 omega_rl
            0.65,  # 18 f1
            0.75,  # 19 f2
            0.85,  # 20 f3
            0.95,  # 21 f4
            0.05,  # 22 Zs
            5.85,  # 23 phi
            5.85,  # 24 theta
            0.05,  # 25 dZs
            0.93,  # 26 dphi
            0.93,  # 27 dtheta
            0.01,  # 28 LLTR_front
            0.01,  # 29 LLTR_rear
            0.00,  # 30 y（不加噪）
            0.00,  # 31 x（不加噪）
        ], device=obs.device)

        # 如果开启观测噪声，添加5%的动态变化（仅影响返回值，不影响内部状态）
        if self.obs_noise:
            rand = torch.rand_like(obs) * 2.0 - 1.0
            noise = rand * obs_noise_scale
            obs_with_noise = obs * (1.0 + noise)
            return obs_with_noise

        return obs


    def reward(self):
        """奖励函数：轨迹跟踪"""
        obs = self.recent_obs
        
        # 从观测中提取信息（不需要反归一化，直接使用）
        lateral_error = obs[:, 0]  # 横向误差 (m)
        heading_error = obs[:, 1]  # 航向角偏差 (rad)
        xd = obs[:, 8]  # 纵向速度 (m/s)
        yd = obs[:, 9]  # 横向速度 (m/s)
        psid = obs[:, 10]  # 横摆角速度 (rad/s)
        v = obs[:, 11]  # 速度 (m/s) - 整体速度 sqrt(xd² + yd²)
        beta = obs[:, 12]  # 侧滑角 (rad)
        phi = obs[:, 23]  # 横滚角 (rad)
        theta = obs[:, 24]  # 俯仰角 (rad)
        dZs = obs[:, 25]  # 垂直速度 (m/s)
        dphi = obs[:, 26]  # 横滚角速度 (rad/s)
        dtheta = obs[:, 27]  # 俯仰角速度 (rad/s)
        
        # 提取悬架力
        f1 = obs[:, 18]  # 悬架力1 (N)
        f2 = obs[:, 19]  # 悬架力2 (N)
        f3 = obs[:, 20]  # 悬架力3 (N)
        f4 = obs[:, 21]  # 悬架力4 (N)
        
        # 提取侧翻风险指标
        LLTR_front = self.es[:, 70]  # 前轴载荷转移率
        LLTR_rear = self.es[:, 71]   # 后轴载荷转移率
        gamma_fr = self.es[:, 72]    # 右前轮摩擦饱和率
        gamma_fl = self.es[:, 73]    # 左前轮摩擦饱和率
        gamma_rr = self.es[:, 74]    # 右后轮摩擦饱和率
        gamma_rl = self.es[:, 75]    # 左后轮摩擦饱和率
        
        # 检查是否失败或成功
        is_failed = (self.is_done == 1).float()
        is_succeeded = (self.is_done == 2).float()
        
        # 1. 存活奖励（根据当前进度给予0-10的奖励）
        x = self.s[:, 0]  # 当前纵向位置
        traj_length = self.perfect_trajectory['states'][-1, 0]  # 轨迹总长度
        progress = torch.clamp(x / traj_length, 0.0, 1.0)  # 进度范围：0到1
        rew_survival = 23.0 * progress  # 根据进度给予0-10的奖励
        
        # 2. 横向跟踪误差惩罚（核心奖励）
        lateral_error_abs = torch.abs(lateral_error)
        rew_lateral = -5.6 * lateral_error_abs - 1.8 * lateral_error_abs ** 2
        
        # 3. 航向角跟踪误差惩罚（核心奖励）
        heading_error_abs = torch.abs(heading_error)
        rew_heading = -15.2 * heading_error_abs - 2.0 * heading_error_abs ** 2
        
        # 4. 速度保持奖励（保持20 m/s左右）
        speed_error = torch.abs(v - 20.0)
        rew_speed = -0.06 * speed_error - 0.1 * speed_error ** 2
        
        # 5. 姿态稳定性奖励
        phi_abs = torch.abs(phi)
        theta_abs = torch.abs(theta)
        rew_attitude = -0.8 * phi_abs ** 2 - 0.8 * theta_abs ** 2
        
        # 6. 侧滑角惩罚（稳定性）- 参考 fixed_circle
        # 轨迹跟踪时希望侧滑角接近0（不同于漂移任务）
        beta_abs = torch.abs(beta)
        rew_beta = -0.09 * torch.where(beta_abs > 0.1, beta_abs ** 2, torch.zeros_like(beta_abs))
        
        # 7. 垂直速度收敛奖励 - 参考 fixed_circle
        # dZs 收敛到 0，保持悬架稳定
        rew_dZs_conv = -1.15 * (dZs ** 2)
        
        # 8. 悬架力平滑性奖励 - 参考 fixed_circle
        # 初始化上一步悬架力历史
        if not hasattr(self, '_last_f_values'):
            self._last_f_values = torch.zeros((self.n, 4), device=self.device)
        
        # 计算悬架力变化
        f_values = torch.stack([f1, f2, f3, f4], dim=1)
        f_changes = torch.abs(f_values - self._last_f_values)
        
        # 智能平滑性约束：惩罚过大变化
        # 参考 fixed_circle：ideal_change_max=1.5N, excessive_change=9N (速度12m/s)
        # lane_change 速度更高(20m/s)，适度放宽约束
        ideal_change_min = 0.0    # 最小理想变化 (N)
        ideal_change_max = 4.5    # 最大理想变化 (N) - 比 fixed_circle 稍大
        excessive_change = 26.0   # 过度变化阈值 (N) - 比 fixed_circle 稍大
        
        f_change_rewards = torch.zeros_like(f_changes)
        
        
        
        # 惩罚过大变化 (参考 fixed_circle: -0.01)
        too_large_mask = f_changes > excessive_change
        f_change_rewards[too_large_mask] = -0.004 * ((f_changes[too_large_mask] - excessive_change) ** 2)
        
        rew_f_smooth = torch.sum(f_change_rewards, dim=1)
        
        # 更新悬架力历史
        self._last_f_values = f_values.detach().clone()
        
        # 9. 悬架力极限惩罚 - 参考 fixed_circle
        # 防止 f 值达到 ±7000N 边界
        # fixed_circle: threshold=4000N, 权重=0.000001
        # lane_change 速度更高，适度提高阈值但保持相似的惩罚强度
        f_limit_threshold = 11500.0  # 在4500N时开始惩罚（略高于 fixed_circle）
        f_abs = torch.abs(f_values)
        
        # 对接近极限的f值进行指数惩罚
        f_over_threshold = torch.clamp(f_abs - f_limit_threshold, min=0)
        rew_f_limit = -0.000002 * torch.sum(f_over_threshold ** 2, dim=1)  # 权重与 fixed_circle 接近
        
        # 10. 侧翻风险惩罚 - LLTR (Lateral Load Transfer Ratio)
        # LLTR 范围 [-1, 1]，接近 ±1 时表示即将侧翻
        # 使用平滑的二次惩罚，当 |LLTR| > 0.6 时开始惩罚
        LLTR_threshold = 0.65
        LLTR_front_abs = torch.abs(LLTR_front)
        LLTR_rear_abs = torch.abs(LLTR_rear)
        
        # 计算超过阈值的部分
        LLTR_front_over = torch.clamp(LLTR_front_abs - LLTR_threshold, min=0)
        LLTR_rear_over = torch.clamp(LLTR_rear_abs - LLTR_threshold, min=0)
        
        # 指数惩罚：接近侧翻时惩罚急剧增加
        rew_LLTR = -0.01* (LLTR_front_over  + LLTR_rear_over )
        
        # 11. 轮胎摩擦饱和惩罚 - gamma (friction saturation)
        # gamma > 1 表示轮胎打滑，应避免
        # 对所有四个轮子的 gamma 进行惩罚
        gamma_threshold = 0.85  # 在达到极限前就开始惩罚
        
        gamma_fr_over = torch.clamp(gamma_fr - gamma_threshold, min=0)
        gamma_fl_over = torch.clamp(gamma_fl - gamma_threshold, min=0)
        gamma_rr_over = torch.clamp(gamma_rr - gamma_threshold, min=0)
        gamma_rl_over = torch.clamp(gamma_rl - gamma_threshold, min=0)
        
        # 平滑惩罚
        rew_gamma = -0.001 * (gamma_fr_over  + gamma_fl_over  + 
                             gamma_rr_over  + gamma_rl_over )
        
        # 12. 成功奖励
        rew_success = 400.0 * is_succeeded
        
        # 13. 失败惩罚
        rew_failure = -130.0 * is_failed
        
        # 总奖励
        rew = (
            rew_survival +
            3 * rew_lateral +
            2 * rew_heading +
            rew_speed +
            100 * rew_attitude +
            100 * rew_beta +
            rew_dZs_conv +
            0.0000005 * rew_f_smooth +
            0.1 * rew_f_limit +
            rew_LLTR +
            rew_success +
            rew_failure
        )
        
        # 保存奖励组件
        self.reward_components = {
            'rew_survival': rew_survival,
            'rew_lateral': 3 * rew_lateral,
            'rew_heading': 2 * rew_heading,
            'rew_speed': rew_speed,
            'rew_attitude': 100 * rew_attitude,
            'rew_beta': 100 * rew_beta,
            'rew_dZs_conv': rew_dZs_conv,
            'rew_f_smooth': 0.0000005 * rew_f_smooth,
            'rew_f_limit': 0.1 * rew_f_limit,
            'rew_LLTR': rew_LLTR,
            'rew_success': rew_success,
            'rew_failure': rew_failure
        }
        
        return rew

    def done(self):
        return self.is_done

    def reset(self):
        super().reset()
        self.is_done = torch.zeros(self.n, dtype=torch.uint8, device=self.device)
        
        # 初始化悬架力历史（用于平滑性奖励）
        self._last_f_values = torch.zeros((self.n, 4), device=self.device)
        
        # 初始化车辆状态：所有车辆从轨迹起点（原点）开始（向量化）
        traj_states = self.perfect_trajectory['states']
        traj_state = traj_states[0]  # 使用轨迹第一个点作为起点
        
        # 从原点开始，添加小的随机扰动增加训练鲁棒性（向量化）
        self.s[:, 0] = traj_state[0] + torch.randn(self.n, device=self.device) * 0.0003  # x: 轻微扰动
        self.s[:, 1] = traj_state[1] + torch.randn(self.n, device=self.device) * 0.0002  # y: 轻微扰动
        self.s[:, 2] = traj_state[2] + torch.randn(self.n, device=self.device) * 0.00003  # psi: 轻微扰动
        self.s[:, 3] = traj_state[3] + torch.randn(self.n, device=self.device) * 0.00005  # xd: 速度轻微扰动
        self.s[:, 4] = traj_state[4] + torch.randn(self.n, device=self.device) * 0.00001  # yd
        self.s[:, 5] = traj_state[5] + torch.randn(self.n, device=self.device) * 0.00002  # psid
        self.s[:, 6] = -0.23  # Zs: 标准悬架高度
        self.s[:, 7] = 0.000003 * (torch.rand(self.n, device=self.device) - 0.5)  # phi: 小扰动
        self.s[:, 8] = 0.00003 * (torch.rand(self.n, device=self.device) - 0.5)  # theta: 小扰动
        
        self.u[:, 5:9] = 0.0  # 悬架力初始化为0
        
        # 使初始轮速与车速匹配，避免起步瞬间高滑移导致 gamma 偏高
        wheel_radius = self.p_body[:, 7]
        vx_init = torch.clamp(self.s[:, 3], min=0.0)
        omega_init = torch.where(wheel_radius > 1e-6, vx_init / wheel_radius, torch.zeros_like(vx_init))
        self.u[:, 1] = omega_init  # omega_fr
        self.u[:, 2] = omega_init  # omega_fl
        self.u[:, 3] = omega_init  # omega_rr
        self.u[:, 4] = omega_init  # omega_rl
        
        # 重建dynamics
        self.dynamics = IWDCarDynamics(self.cast_action(self.u), self.p_body, self.p_tyre)
        self.es = self.dynamics.compute_extended_state(self.s)
        
        return self.obs()

    def info(self):
        info_dict = {
            "time_outs": (self.is_done == 2)
        }
        
        # 添加奖励组件（使用 reward_ 前缀以便训练代码识别）
        if hasattr(self, 'reward_components') and self.reward_components:
            for key, value in self.reward_components.items():
                info_dict[f'reward_{key}'] = value
        
        # 添加轨迹跟踪状态信息
        obs = self.recent_obs
        if obs is not None:
            info_dict['lateral_error'] = torch.abs(obs[:, 0])  # 横向误差 (m)
            info_dict['heading_error'] = torch.abs(obs[:, 1])  # 航向角偏差 (rad)
            info_dict['speed'] = obs[:, 11]  # 速度 (m/s)
            
            # 计算跟踪性能指标
            x = self.s[:, 0]
            traj_length = self.perfect_trajectory['states'][-1, 0]
            info_dict['progress'] = torch.clamp(x / traj_length, 0.0, 1.0)  # 轨迹完成进度
                
        return info_dict

    def reset_done_envs(self):
        """重置完成的环境：从完美轨迹随机抽样（向量化版本）"""
        is_done = self.is_done.bool()
        size = torch.sum(is_done).item()
        
        if size == 0:
            return
        
        # 向量化：为所有需要重置的环境生成随机扰动
        num_reset = size
        
        # 从完美轨迹中随机抽样起始点（为每个环境独立采样）
        num_traj_steps = self.perfect_trajectory['num_steps']
        random_indices = torch.randint(0, num_traj_steps, (num_reset,), device=self.device)
        
        # 为每个需要重置的环境设置对应的step_count（从随机索引开始）
        # 创建一个临时数组来存储新的step_count值
        done_indices = torch.where(is_done)[0]
        # 向量化赋值，避免 Python 循环
        self.step_count[done_indices] = random_indices
        
        # 从轨迹中获取随机抽样的状态和动作
        traj_states = self.perfect_trajectory['states'][random_indices]  # shape: (num_reset, 16)
        traj_actions = self.perfect_trajectory['actions'][random_indices]  # shape: (num_reset, 9)
        
        # 从随机轨迹点开始，添加小的随机扰动（向量化）
        self.s[is_done, 0] = traj_states[:, 0] + torch.randn(num_reset, device=self.device) * 0.3  # x
        self.s[is_done, 1] = traj_states[:, 1] + torch.randn(num_reset, device=self.device) * 0.2  # y
        self.s[is_done, 2] = traj_states[:, 2] + torch.randn(num_reset, device=self.device) * 0.03  # psi
        self.s[is_done, 3] = traj_states[:, 3] + torch.randn(num_reset, device=self.device) * 0.5  # xd
        self.s[is_done, 4] = traj_states[:, 4] + torch.randn(num_reset, device=self.device) * 0.1  # yd
        self.s[is_done, 5] = traj_states[:, 5] + torch.randn(num_reset, device=self.device) * 0.02  # psid
        self.s[is_done, 6] = -0.23  # Zs
        self.s[is_done, 7] = 0.003 * (torch.rand(num_reset, device=self.device) - 0.5)  # phi
        self.s[is_done, 8] = 0.003 * (torch.rand(num_reset, device=self.device) - 0.5)  # theta
        self.s[is_done, 13] = 0.0  # dZs
        self.s[is_done, 14] = 0.0  # dphi
        self.s[is_done, 15] = 0.0  # dtheta
        
        # 设置初始动作（接近轨迹随机点的动作，向量化）
        self.u[is_done, 0] = traj_actions[:, 0] + torch.randn(num_reset, device=self.device) * 0.01  # delta
        self.u[is_done, 1] = traj_actions[:, 1] + torch.randn(num_reset, device=self.device) * 1.0  # omega_fr
        self.u[is_done, 2] = traj_actions[:, 2] + torch.randn(num_reset, device=self.device) * 1.0  # omega_fl
        self.u[is_done, 3] = traj_actions[:, 3] + torch.randn(num_reset, device=self.device) * 1.0  # omega_rr
        self.u[is_done, 4] = traj_actions[:, 4] + torch.randn(num_reset, device=self.device) * 1.0  # omega_rl
        self.u[is_done, 5] = 0.0  # f1
        self.u[is_done, 6] = 0.0  # f2
        self.u[is_done, 7] = 0.0  # f3
        self.u[is_done, 8] = 0.0  # f4
        
        # 重置完成标志
        self.is_done[:] = 0

    def step(self, action, **kwargs):
        """
        执行一步仿真
        
        Args:
            action: 物理动作，shape (n, 9)
        """
        self.reset_done_envs()
        # 直接使用物理动作，不需要反归一化
        obs, reward, done, info = super().step(action, **kwargs)
        
        # 测试模式下的调试打印（环境结束时打印）
        if not self.train:
            self._print_debug_info()
        
        return obs, reward, done, info
    
    def _print_debug_info(self):
        """打印测试模式下的调试信息：环境结束时的状态"""
        # 检查哪些环境刚刚结束
        done_mask = self.is_done != 0
        done_indices = torch.where(done_mask)[0]
        
        if len(done_indices) == 0:
            return
        
        for idx in done_indices:
            idx = idx.item()
            
            # 存活步数
            steps = self.step_count[idx].item()
            
            # 计算轨迹进度
            x = self.s[idx, 0].item()
            traj_length = self.perfect_trajectory['states'][-1, 0].item()
            progress = (x / traj_length) * 100
            
            # 成功或失败
            if self.is_done[idx] == 2:
                status = "✓ 成功"
                print(f"环境 {idx:3d}: 存活 {steps:4d} 步 | 进度 {progress:5.1f}% | {status}")
            else:
                status = "✗ 失败"
                # 计算失败原因
                obs = self.recent_obs[idx]
                lateral_error = obs[0].item()
                heading_error = obs[1].item()
                phi = obs[23].item()
                theta = obs[24].item()
                v = obs[11].item()
                
                reasons = []
                if abs(lateral_error) > 1.9:
                    reasons.append(f"横向偏离过大({lateral_error:+.2f}m)")
                if abs(heading_error) > 0.80:
                    reasons.append(f"航向角偏差过大({np.rad2deg(heading_error):+.1f}°)")
                if abs(phi) > 0.8:
                    reasons.append(f"横滚角过大({np.rad2deg(phi):+.1f}°)")
                if v < 10.0:
                    reasons.append(f"速度过低({v:.1f}m/s)")
                if abs(theta) > 0.8:
                    reasons.append(f"俯仰角过大({np.rad2deg(theta):+.1f}°)")
                
                reason = ", ".join(reasons) if reasons else "未知原因"
                print(f"环境 {idx:3d}: 存活 {steps:4d} 步 | 进度 {progress:5.1f}% | {status} | 原因: {reason}")
      
