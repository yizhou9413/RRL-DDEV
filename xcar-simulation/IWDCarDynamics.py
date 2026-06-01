import torch
import torch.nn as nn

class IWDCarDynamics(nn.Module):
    def __init__(self, u, p_body, p_tyre, differentiable=False, disturbance=None, p_body_disturbance=None, drivetrain="iwd"):
        """
        u: [batch_size, 9] - (delta, omega_fr, omega_fl, omega_rr, omega_rl, f1, f2, f3, f4)
        p_body: [batch_size, 8] - (lf, lr, m, h, g, Iz, T, wheel_radius)
        p_tyre: [batch_size, 4] - (B, C, D, E) for Pacejka formula
        disturbance: [batch_size, 10] - (fx_1..4, fy_1..4, M_phi, M_theta) 力和力矩扰动
        p_body_disturbance: [batch_size, 5] - (delta_m, delta_K21, delta_C21, delta_K23, delta_C23) 参数扰动
        """
        super().__init__()
        to_param = nn.Parameter if differentiable else lambda x: x
        self.u = to_param(u)
        self.p_body = to_param(p_body)
        self.p_tyre = to_param(p_tyre)
        self.disturbance = disturbance
        self.p_body_disturbance = p_body_disturbance
        self.drivetrain = drivetrain

    def compute_extended_state(self, s):
        """
        s: [batch_size, 6] - (x, y, psi, vx, vy, psid)
        Returns: Extended state including velocities, forces, and accelerations
        """
        # Decoding state and parameters
        x, y, psi = s[:, 0], s[:, 1], s[:, 2]
        vx, vy, psid = s[:, 3], s[:, 4], s[:, 5]
        Zs, phi, theta, Z11, Z12, Z13, Z14, dZs, dphi, dtheta, dZ11, dZ12, dZ13, dZ14 = s[:, 6:].T
        delta = self.u[:, 0]
        omega_fr = self.u[:, 1]
        omega_fl = self.u[:, 2] 
        omega_rr = self.u[:, 3]
        omega_rl = self.u[:, 4]
        f1 = self.u[:, 5]
        f2 = self.u[:, 6]
        f3 = self.u[:, 7]
        f4 = self.u[:, 8]
        
        lf = self.p_body[:, 0]
        lr = self.p_body[:, 1]
        m = self.p_body[:, 2]
        h = self.p_body[:, 3]
        g = self.p_body[:, 4]
        Iz = self.p_body[:, 5]
        T = self.p_body[:, 6]

        wheel_radius = self.p_body[:, 7]
        m_s = self.p_body[:, 8]
        m_uf = self.p_body[:, 9]
        m_ur = self.p_body[:, 10]
        Ix = self.p_body[:, 11]
        Iy = self.p_body[:, 12]
        C_af = self.p_body[:, 13]
        C_ar = self.p_body[:, 14]
        K21 = self.p_body[:, 15]
        C21 = self.p_body[:, 16]
        K23 = self.p_body[:, 17]
        C23 = self.p_body[:, 18]
        K1 = self.p_body[:, 19]
        C1 = self.p_body[:, 20]
        Kaf = self.p_body[:, 21]
        Kar = self.p_body[:, 22]
        k_phi = self.p_body[:, 23]
        c_phi = self.p_body[:, 24]
        k_theta = self.p_body[:, 25]
        c_theta = self.p_body[:, 26]

        B = self.p_tyre[:, 0]
        C = self.p_tyre[:, 1]
        D = self.p_tyre[:, 2]
        E = self.p_tyre[:, 3]

        # 应用参数扰动（如果存在）
        # p_body_disturbance: [batch_size, 5] = [delta_m, delta_K21, delta_C21, delta_K23, delta_C23]
        if self.p_body_disturbance is not None:
            m = m + self.p_body_disturbance[:, 0]
            K21 = K21 + self.p_body_disturbance[:, 1]
            C21 = C21 + self.p_body_disturbance[:, 2]
            K23 = K23 + self.p_body_disturbance[:, 3]
            C23 = C23 + self.p_body_disturbance[:, 4]

        eps = 1e-8
        tiny = 1e-6

        # Keep critical physical parameters away from singular values.
        # Relative lower bounds keep behavior consistent across different presets.
        m = torch.clamp(m, min=torch.clamp(self.p_body[:, 2] * 0.1, min=tiny))
        m_s = torch.clamp(m_s, min=torch.clamp(self.p_body[:, 8] * 0.1, min=tiny))
        m_uf = torch.clamp(m_uf, min=torch.clamp(self.p_body[:, 9] * 0.1, min=tiny))
        m_ur = torch.clamp(m_ur, min=torch.clamp(self.p_body[:, 10] * 0.1, min=tiny))
        Iz = torch.clamp(Iz, min=torch.clamp(self.p_body[:, 5] * 0.1, min=tiny))
        Ix = torch.clamp(Ix, min=torch.clamp(self.p_body[:, 11] * 0.1, min=tiny))
        Iy = torch.clamp(Iy, min=torch.clamp(self.p_body[:, 12] * 0.1, min=tiny))
        wheel_radius = torch.clamp(wheel_radius, min=torch.clamp(self.p_body[:, 7] * 0.2, min=tiny))
        K21 = torch.clamp(K21, min=torch.clamp(self.p_body[:, 15] * 0.05, min=tiny))
        K23 = torch.clamp(K23, min=torch.clamp(self.p_body[:, 17] * 0.05, min=tiny))
        C21 = torch.clamp(C21, min=torch.clamp(self.p_body[:, 16] * 0.05, min=tiny))
        C23 = torch.clamp(C23, min=torch.clamp(self.p_body[:, 18] * 0.05, min=tiny))

        B = torch.clamp(B, min=tiny)
        C = torch.clamp(C, min=tiny)
        D = torch.clamp(D, min=tiny)
        E = torch.clamp(torch.abs(E), min=tiny)

        # f1, f2, f3, f4 are now controlled via action inputs
        # Transform velocities from world to body frame [batch_size]
        c_psi = torch.cos(psi)
        s_psi = torch.sin(psi)
        vx_body = c_psi * vx + s_psi * vy
        vy_body = -s_psi * vx + c_psi * vy

        # Calculate vehicle states
        v = torch.hypot(vx_body, vy_body)
        beta = torch.atan2(vy, vx + eps) - psi
        beta = torch.atan2(torch.sin(beta), torch.cos(beta))

        # Wheel positions relative to CoG [batch_size, 4]
        corner_x = torch.stack([lf, lf, -lr, -lr], dim=1)  # fr, fl, rr, rl
        corner_y = torch.stack([-T/2, T/2, -T/2, T/2], dim=1)  # fr, fl, rr, rl

        # Wheel velocities [batch_size, 2, 4]
        v_base = torch.stack([
            vx_body.unsqueeze(1).expand(-1, 4),
            vy_body.unsqueeze(1).expand(-1, 4)
        ], dim=1)

        delta_v = torch.stack([-corner_y, corner_x], dim=1) * psid.unsqueeze(1).unsqueeze(2)
        v_wheel = v_base + delta_v

        # Apply steering to front wheels [batch_size, 2, 2]
        c_delta = torch.cos(delta)
        s_delta = torch.sin(delta)
        R_steer = torch.stack([
            torch.stack([c_delta, s_delta], dim=1),
            torch.stack([-s_delta, c_delta], dim=1)
        ], dim=1)
        
        # Transform front wheel velocities [batch_size, 2, 4]
        v_wheel_local = v_wheel.clone()
        v_wheel_local[:, :, :2] = torch.bmm(R_steer, v_wheel[:, :, :2])

        # Reset front wheel velocities when drivetrain is "rwd"
        if self.drivetrain == "rwd":
            omega_fr = v_wheel_local[:, 0, 0]
            omega_fl = v_wheel_local[:, 0, 1]
        
        # Wheel angular velocities [batch_size, 4]
        omega = torch.stack([omega_fr, omega_fl, omega_rr, omega_rl], dim=1)
        
        # Calculate slip ratios (physically consistent) [batch_size, 4]
        vx_local = v_wheel_local[:, 0, :]  # [batch_size, 4]
        vy_local = v_wheel_local[:, 1, :]  # [batch_size, 4]
        
        # Longitudinal slip: kappa = (R*omega - Vx) / max(|Vx|, v_min)
        r_expanded = wheel_radius.unsqueeze(1)  # [batch_size, 1]
        vx_ref = torch.clamp(torch.abs(vx_local), min=0.5)  # stabilizer for low speeds
        sx = (r_expanded * omega - vx_local) / vx_ref
        
        # Lateral slip surrogate using slip angle small-angle approx: tan(alpha) ~ Vy / Vx
        sy = vy_local / vx_ref
        
        # Combined slip
        slip = torch.hypot(sx, sy)
        
        # Pacejka magic formula [batch_size, 4]
        mu = D.unsqueeze(1) * torch.sin(
            C.unsqueeze(1) * torch.atan(
                B.unsqueeze(1) * slip - E.unsqueeze(1) * (B.unsqueeze(1) * slip - torch.atan(B.unsqueeze(1) * slip))
            )
        )

        # Slip angles [batch_size, 4]
        alpha = torch.atan2(sy, sx + eps)

        # Friction components [batch_size, 4]
        mux = -torch.cos(alpha) * mu
        muy = -torch.sin(alpha) * mu
        
        # Load transfer
        G = m * g
        l = lf + lr
        
        # Longitudinal load transfer [batch_size]
        load_transfer_denom = l + h * (
            mux[:, :2].mean(dim=1) * c_delta - muy[:, :2].mean(dim=1) * s_delta - mux[:, 2:].mean(dim=1)
        )
        load_transfer_denom = torch.where(
            load_transfer_denom >= 0.0,
            torch.clamp(load_transfer_denom, min=tiny),
            torch.clamp(load_transfer_denom, max=-tiny)
        )
        fz_front = (lr * G - h * G * mux[:, 2:].mean(dim=1)) / load_transfer_denom
        fz_rear = G - fz_front

        # Individual wheel loads [batch_size, 4]
        fz = torch.cat([
            fz_front.unsqueeze(1).expand(-1, 2) / 2,
            fz_rear.unsqueeze(1).expand(-1, 2) / 2
        ], dim=1)
        
        # Tire forces in wheel frame [batch_size, 4]
        fx_local = mux * fz
        fy_local = muy * fz
        
        # Transform forces to body frame
        fx = fx_local.clone()
        fy = fy_local.clone()
        fx[:, :2] = fx_local[:, :2] * c_delta.unsqueeze(1) - fy_local[:, :2] * s_delta.unsqueeze(1)
        fy[:, :2] = fx_local[:, :2] * s_delta.unsqueeze(1) + fy_local[:, :2] * c_delta.unsqueeze(1)
        
        if self.disturbance is not None:
            fx = fx + self.disturbance[:, :4]
            fy = fy + self.disturbance[:, 4:8]
        
        # Body accelerations [batch_size]
        ax = torch.sum(fx, dim=1) / m  # Longitudinal acceleration from forces
        ay = torch.sum(fy, dim=1) / m  # Update lateral acceleration with force contribution
        
        # Yaw acceleration [batch_size]
        yaw_moment = (fy[:, :2].sum(dim=1) * lf) - (fy[:, 2:].sum(dim=1) * lr) + \
                    (fx[:, [0,2]].sum(dim=1) * (T/2)) - (fx[:, [1,3]].sum(dim=1) * (T/2))
        psid_dot = yaw_moment / Iz
       
        # Transform accelerations back to world frame [batch_size]
        ax_world = c_psi * ax - s_psi * ay
        ay_world = s_psi * ax + c_psi * ay

        # tire_force
        F11 = K1 * (Z11 - 0) + C1 * (dZ11 - 0)
        F12 = K1 * (Z12 - 0) + C1 * (dZ12 - 0)
        F13 = K1 * (Z13 - 0) + C1 * (dZ13 - 0)
        F14 = K1 * (Z14 - 0) + C1 * (dZ14 - 0)
        # 悬架预压缩量计算（平衡重力）
        preload_f = m_s * g / (4 * K21)  # 前悬架预压缩
        preload_r = m_s * g / (4 * K23)  # 后悬架预压缩
        #f1=0
        #f2=0
        #f3=0
        #f4=0
        # 悬架力（包含预压缩）
        F21 = K21 * (Z11 - (Zs - lf * theta - T / 2 * phi) + preload_f) + C21 * (
                dZ11 - (dZs - lf * dtheta - T / 2 * dphi)) - Kaf / T * (phi - (Z12 - Z11) / T)  + f1
        F22 = K21 * (Z12 - (Zs - lf * theta + T / 2 * phi) + preload_f) + C21 * (
                dZ12 - (dZs - lf * dtheta + T / 2 * dphi)) + Kaf / T * (phi - (Z12 - Z11) / T)  + f2
        F23 = K23 * (Z13 - (Zs + lr * theta - T / 2 * phi) + preload_r) + C23 * (
                dZ13 - (dZs + lr * dtheta - T / 2 * dphi)) - Kar / T * (phi - (Z13 - Z14) / T)  + f3
        F24 = K23 * (Z14 - (Zs + lr * theta + T / 2 * phi) + preload_r) + C23 * (
                dZ14 - (dZs + lr * dtheta + T / 2 * dphi)) + Kar / T * (phi - (Z13 - Z14) / T)  + f4

        # 车身方程

        # z a
        ddZs = (F21 + F22 + F23 + F24) / m_s - g
        # 侧倾
        ddphi = (m_s * v * (beta + psid) * h - k_phi * phi - c_phi * dphi + (F22 + F24 - F21 - F23) * T / 2) / Ix
        # 俯仰（添加纵向加速度产生的惯性力矩）
        # 物理一致：纵向惯性俯仰力矩 M = m_s * h * ax
        pitch_inertia_moment = m_s * ax * h
        
        # 增强俯仰阻尼，提高俯仰稳定性
        
        ddtheta = (pitch_inertia_moment - k_theta * theta - c_theta * dtheta + lf * (F23 + F24) - lr * (F21 + F22)) / Iy
        
        # 添加力矩扰动（如果启用）
        # disturbance[:, 8] = 侧倾力矩扰动 M_phi (N·m)
        # disturbance[:, 9] = 俯仰力矩扰动 M_theta (N·m)
        if self.disturbance is not None and self.disturbance.shape[1] >= 10:
            ddphi = ddphi + self.disturbance[:, 8] / Ix
            ddtheta = ddtheta + self.disturbance[:, 9] / Iy
        # 注意：移除了角加速度裁剪，让悬架参数随机化能真正影响车辆行为
        # 如果仿真发散，可以恢复以下裁剪：
        # ddtheta = torch.clamp(ddtheta, -5.0, 5.0)
        # ddphi = torch.clamp(ddphi, -2.6, 2.6)
        # 调试输出：验证横滚角和俯仰角的物理合理性
        # if torch.rand(1).item() < 0.002:  # 偶尔打印一次，避免输出过多
        #     idx = 0  # 查看第一个环境
        #     v_val = v[idx].item()
        #     phi_val = phi[idx].item()
        #     theta_val = theta[idx].item()
        #     ax_val = ax[idx].item()
        #     dtheta_val = dtheta[idx].item()
        #     phi_deg = phi_val * 180 / 3.14159
        #     theta_deg = theta_val * 180 / 3.14159
        #
        #     # 计算各个力矩分量
        #     M_inertia = pitch_inertia_moment[idx].item()
        #     M_suspension = (lf[idx] * (F23[idx] + F24[idx]) - lr[idx] * (F21[idx] + F22[idx])).item()
        #     M_stiffness = -k_theta[idx].item() * theta_val
        #     M_damping = -c_theta[idx].item() * dtheta_val
        #     M_total = M_inertia + M_suspension + M_stiffness + M_damping
        #
        #     print("="*70)
        #     print(f"俯仰力矩分析 (v={v_val:.2f} m/s, ax={ax_val:.3f} m/s²):")
        #     print(f"  θ = {theta_val:.4f} rad = {theta_deg:.1f}°, dθ = {dtheta_val:.4f}")
        #     print(f"力矩分量:")
        #     print(f"  惯性力矩    = {M_inertia:+6.0f} N·m (ax效应)")
        #     print(f"  悬架力矩    = {M_suspension:+6.0f} N·m (载荷转移)")
        #     print(f"  恢复力矩    = {M_stiffness:+6.0f} N·m (弹性)")
        #     print(f"  阻尼力矩    = {M_damping:+6.0f} N·m (阻尼)")
        #     print(f"  总力矩      = {M_total:+6.0f} N·m")
        #     if v_val < 8 and theta_val > 0:
        #         print("  ⚠️  注意：v在下降但θ为正的情况！")
        #     print("="*70)
        
        # 轮端方程（暂时移除载荷转移项以修复悬架力矩异常）
        ddZ11 = (-F11 - F21) / m_uf  # 前左轮
        ddZ12 = (-F12 - F22) / m_uf  # 前右轮
        ddZ13 = (-F13 - F23) / m_ur  # 后左轮
        ddZ14 = (-F14 - F24) / m_ur  # 后右轮
        # === 计算侧翻风险指标 ===
        # 使用轮胎竖向力 F11..F14（接地法向反力）作为每个角点的接触载荷
        # 映射关系：F11=前左(FL), F12=前右(FR), F13=后左(RL), F14=后右(RR)
        fz_fl_contact = torch.clamp(torch.abs(F11), min=eps)
        fz_fr_contact = torch.clamp(torch.abs(F12), min=eps)
        fz_rl_contact = torch.clamp(torch.abs(F13), min=eps)
        fz_rr_contact = torch.clamp(torch.abs(F14), min=eps)

        # 1) Lateral Load Transfer Ratio (LLTR) - 左右轮载荷转移率（用接地点载荷）
        # 注意顺序: [fr, fl, rr, rl]
        LLTR_front = (fz_fr_contact - fz_fl_contact) / (fz_fr_contact + fz_fl_contact + eps)
        LLTR_rear  = (fz_rr_contact - fz_rl_contact) / (fz_rr_contact + fz_rl_contact + eps)
        

        # 2) 轮胎摩擦饱和利用率 γ（4驱独立）
        # 物理意义：实际滑移率 vs 峰值滑移率的比值
        # γ = slip_magnitude / slip_at_peak
        
        # 计算滑移率幅值
        slip_magnitude = torch.sqrt(sx**2 + sy**2)  # [batch, 4]
        
        # 估算峰值滑移率：Pacejka 公式在 slip ≈ 1/(B*E) 时达到峰值
        # 对于 tesla_model_3: B=10, E=0.97, 峰值滑移率 ≈ 0.103
        slip_peak = 1.0 / (B.unsqueeze(1) * E.unsqueeze(1) + eps)  # [batch, 1]
        
        # gamma = 当前滑移率 / 峰值滑移率
        gamma_all = slip_magnitude / (slip_peak + eps)
        # 数值稳定：限制到合理范围
        gamma_all = torch.clamp(gamma_all, 0.0, 3.0)
        gamma_fr, gamma_fl, gamma_rr, gamma_rl = gamma_all[:, 0], gamma_all[:, 1], gamma_all[:, 2], gamma_all[:, 3]
        
        # Combine all extended states
        extended_states = [
            vx, vy, psid,  # Original velocities [3]
            ax_world, ay_world, psid_dot,  # Accelerations [3]
            dZs, dphi, dtheta, dZ11, dZ12, dZ13, dZ14, ddZs, ddphi, ddtheta, ddZ11, ddZ12, ddZ13, ddZ14,
            v, beta,  # Vehicle states [2]
            v_wheel_local[:, 0, :], v_wheel_local[:, 1, :],  # Wheel velocities [8]
            sx, sy, slip,  # Slip ratios [12]
            mu, alpha,  # Friction states [8]
            mux, muy,  # Friction components [8]
            fz, fx, fy,  # Forces [12]
            LLTR_front, LLTR_rear,  # Lateral Load Transfer Ratio [2]
            gamma_fr, gamma_fl, gamma_rr, gamma_rl,  # Friction saturation (fr, fl, rr, rl) [4]
        ]

        extended = torch.cat([t.unsqueeze(1) if t.dim() == 1 else t for t in extended_states], 1)
        # Last-line guard to avoid NaN/Inf propagation into the ODE integrator.
        return torch.nan_to_num(extended, nan=0.0, posinf=1e6, neginf=-1e6)
    
    def forward(self, t, s):
        es = self.compute_extended_state(s)
        return es[:, :20]  # 返回状态导数 (ds/dt)，必须与s的维度(20)匹配
