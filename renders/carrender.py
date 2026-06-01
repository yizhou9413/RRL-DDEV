import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle, Polygon, FancyArrowPatch
import glob

# 解决OpenMP警告（如果存在）
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# === 设置论文级别的绘图样式 ===
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
plt.rcParams['font.size'] = 11
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['legend.framealpha'] = 0.95
plt.rcParams['legend.edgecolor'] = 'gray'
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['lines.linewidth'] = 2.0

# === 1. Read Data ===
# Method 1: Specify CSV file path directly in code (Recommended)
data_file = "runs/a2c_continuous_moose_default/rollouts/20260108-204430/env0.csv"  # Modify to your CSV file path

# Method 2: Specify via command line argument
if len(sys.argv) > 1:
    data_file = sys.argv[1]
    print(f"Using file from command line: {data_file}")
else:
    # If specified file doesn't exist, auto-find latest CSV file
    if not os.path.exists(data_file):
        csv_files = glob.glob("data/*.csv")
        if csv_files:
            data_file = max(csv_files, key=os.path.getctime)
            print(f"Specified file not found, using latest data file: {data_file}")
        else:
            raise FileNotFoundError("No CSV file found in data directory")

try:
    # Read CSV file without header (first row is data)
    df = pd.read_csv(data_file, header=None)
    print(f"Successfully loaded data file: {data_file}")
    print(f"Number of rows: {len(df)}")
    print(f"Number of columns: {len(df.columns)}")
except Exception as e:
    print(f"Error reading data file: {e}")
    raise

# Set column names (according to actual column order in CSV file)
num_cols = len(df.columns)
col_names_32 = ["x", "y", "psi", "x_dot", "y_dot", "psi_dot", "delta",
                "omega_fr", "omega_fl", "omega_rr", "omega_rl", "f1", "f2", "f3", "f4", 
                "r", "beta", "v", "Zs", "phi", "theta", 
                "Z11", "Z12", "Z13", "Z14", 
                "dZs", "dphi", "dtheta", "dZ11", "dZ12", "dZ13", "dZ14"]
col_names_extra = ["LLTR_front", "LLTR_rear", "gamma_fr", "gamma_fl", "gamma_rr", "gamma_rl"]
col_names_38 = col_names_32 + col_names_extra

if num_cols >= len(col_names_38):
    df.columns = col_names_38[:num_cols] + [f"col_{i}" for i in range(len(col_names_38), num_cols)]
    print(f"Using extended column names (>= {len(col_names_38)})")
elif num_cols == 32:
    df.columns = col_names_32
    print("Using standard 32-column names")
elif num_cols >= 19:
    # At least need basic columns
    df.columns = col_names_32[:num_cols] + [f"col_{i}" for i in range(len(col_names_32), num_cols)]
    print(f"Using first {min(num_cols, len(col_names_32))} standard column names")
else:
    raise ValueError(f"Data file has insufficient columns: {num_cols} columns, at least 19 required")

# Extract data
x_data = df["x"].values
y_data = df["y"].values
psi_data = df["psi"].values
v_data = df["v"].values
Zs_data = df["Zs"].values
phi_data = df["phi"].values
theta_data = df["theta"].values
f1_data = df["f1"].values
f2_data = df["f2"].values
f3_data = df["f3"].values
f4_data = df["f4"].values
# 可能存在的额外列（若 CSV 提供）
LLTR_front_data = df["LLTR_front"].values if "LLTR_front" in df.columns else None
LLTR_rear_data = df["LLTR_rear"].values if "LLTR_rear" in df.columns else None

# 将姿态角转换为角度用于绘图显示
phi_deg_data = np.degrees(phi_data)
theta_deg_data = np.degrees(theta_data)
# 转向角（弧度→度）用于显示
delta_data = df["delta"].values
delta_deg_data = np.degrees(delta_data)
# 侧偏角 beta（弧度→度）
beta_data = df["beta"].values if "beta" in df.columns else np.zeros_like(x_data)
beta_deg_data = np.degrees(beta_data)

# Validate data
if len(x_data) == 0:
    raise ValueError("Data file is empty")
print(f"Data validation passed: {len(x_data)} data points")
print(f"X range: [{x_data.min():.2f}, {x_data.max():.2f}] m")
print(f"Y range: [{y_data.min():.2f}, {y_data.max():.2f}] m")

# === 2. 道路参数（A-B-C车道结构）===
lane_a_length = 22.0      # A道长度 (m)
curve_ab_length = 18      # A到B弯道长度 (m)
lane_b_length = 11.0      # B道长度 (m)
curve_bc_length = 16      # B到C弯道长度 (m)
lane_c_length = 12.0      # C道长度 (m)

lane_width = 3.0          # 车道宽度 (m)
lane_a_y = 0.0            # A道的y坐标
lane_b_y = 3.2            # B道的y坐标（偏移量）
lane_c_y = 0.0            # C道的y坐标

# 计算各段的结束位置
x1_end = lane_a_length                           # 12m：A道结束
x2_end = x1_end + curve_ab_length                # 30m：弯道AB结束
x3_end = x2_end + lane_b_length                  # 41m：B道结束
x4_end = x3_end + curve_bc_length                # 57m：弯道BC结束
x5_end = x4_end + lane_c_length                  # 69m：C道结束

# === 3. 生成参考轨迹（目标路径中心线）===
def get_target_y(x):
    """
    根据x坐标计算目标y位置（道路中心线）
    - 阶段1 (0-12m): A道直线 y=0
    - 阶段2 (12-30m): A到B弯道（从y=0到y=3.2，使用余弦插值）
    - 阶段3 (30-41m): B道直线 y=3.2
    - 阶段4 (41-57m): B到C弯道（从y=3.2回到y=0，使用余弦插值）
    - 阶段5 (≥57m): C道直线 y=0
    """
    x_array = np.asarray(x)
    is_scalar = x_array.ndim == 0
    if is_scalar:
        x_array = x_array[np.newaxis]
    
    y = np.zeros_like(x_array)
    
    # 第一段：A道直线 y=0 (x <= 12m)
    mask1 = x_array <= x1_end
    y[mask1] = lane_a_y
    
    # 第二段：A到B弯道（从y=0到y=3.2, 12m < x <= 30m）
    mask2 = (x_array > x1_end) & (x_array <= x2_end)
    if np.any(mask2):
        s = x_array[mask2] - x1_end  # 曲线段的偏移量
        progress = s / curve_ab_length  # 0 到 1
        progress = np.clip(progress, 0.0, 1.0)
        
        # 使用余弦插值：y(s) = lane_b_y * (1 - cos(π * progress)) / 2
        y_curve = lane_b_y * (1.0 - np.cos(np.pi * progress)) / 2.0
        y[mask2] = y_curve
    
    # 第三段：B道直线 y=3.2 (30m < x <= 41m)
    mask3 = (x_array > x2_end) & (x_array <= x3_end)
    y[mask3] = lane_b_y
    
    # 第四段：B到C弯道（从y=3.2回到y=0, 41m < x <= 57m）
    mask4 = (x_array > x3_end) & (x_array <= x4_end)
    if np.any(mask4):
        s = x_array[mask4] - x3_end  # 曲线段的偏移量
        progress = s / curve_bc_length  # 0 到 1
        progress = np.clip(progress, 0.0, 1.0)
        
        # 使用余弦插值：y(s) = lane_b_y * (1 + cos(π * progress)) / 2
        y_curve = lane_b_y * (1.0 + np.cos(np.pi * progress)) / 2.0
        y[mask4] = y_curve
    
    # 第五段：C道直线 y=0 (x > 57m)
    mask5 = x_array > x4_end
    y[mask5] = lane_c_y
    
    return y[0] if is_scalar else y

# 生成参考轨迹点
x_max = max(float(np.max(x_data)), float(x5_end)) + 5
x_ref = np.linspace(0, x_max, 1000)
y_ref = get_target_y(x_ref)
#
# 预计算完整参考y用于偏移量绘制
y_target_full = get_target_y(x_data)

# === 4. Setup Canvas ===
# 仅保留主图，注释掉数据子图
fig = plt.figure(figsize=(26, 30), dpi=120)
# gs = fig.add_gridspec(5, 2, height_ratios=[3.5, 1, 1, 1, 1], width_ratios=[1, 1], 
#                       hspace=0.35, wspace=0.25)

# 主动画区域（道路和车辆）- 使用 add_axes 精确控制位置和大小
# [left, bottom, width, height] 以图形尺寸的比例表示
ax_main = fig.add_axes([0.05, 0.08, 0.92, 0.87])  # 左边距5%，底部8%，宽92%，高87%
ax_main.set_aspect('equal')
ax_main.set_facecolor('#f8f9fa')

# 设置坐标轴范围
x_min = min(min(x_data), 0) - 2
x_max_plot = max(max(x_data), x5_end) + 5
y_min = -2
y_max = lane_b_y + lane_width + 2

ax_main.set_xlim(x_min, x_max_plot)
ax_main.set_ylim(y_min, y_max)
ax_main.set_xlabel("Longitudinal Position, X (m)", fontsize=13, fontweight='bold')
ax_main.set_ylabel("Lateral Position, Y (m)", fontsize=13, fontweight='bold')
ax_main.set_title("Vehicle Trajectory on A-B-C Lane Change Maneuver", 
                  fontsize=15, fontweight='bold', pad=15)
ax_main.grid(True, alpha=0.35, linewidth=0.8)

# === 5. Draw Road Boundaries ===
# 定义专业配色
color_boundary = '#2c3e50'  # 深灰蓝色边界
color_centerline = '#f39c12'  # 橙色中心线

# Lane A boundaries (y=0, width 3m)
lane_a_left = lane_a_y - lane_width / 2
lane_a_right = lane_a_y + lane_width / 2

# Lane A left boundary line
ax_main.plot([0, x1_end], [lane_a_left, lane_a_left], 
             color=color_boundary, linewidth=2.5, alpha=0.9, label='Lane Boundary', solid_capstyle='round')
# Lane A right boundary line
ax_main.plot([0, x1_end], [lane_a_right, lane_a_right], 
             color=color_boundary, linewidth=2.5, alpha=0.9, solid_capstyle='round')
# Lane A centerline
ax_main.plot([0, x1_end], [lane_a_y, lane_a_y], 
             color=color_centerline, linestyle='--', linewidth=2, alpha=0.7, 
             label='Lane Centerline', dashes=(8, 4))

# Lane B boundaries (y=3.2, width 3m)
lane_b_left = lane_b_y - lane_width / 2
lane_b_right = lane_b_y + lane_width / 2

# Lane B left boundary line
ax_main.plot([x2_end, x3_end], [lane_b_left, lane_b_left], 
             color=color_boundary, linewidth=2.5, alpha=0.9, solid_capstyle='round')
# Lane B right boundary line
ax_main.plot([x2_end, x3_end], [lane_b_right, lane_b_right], 
             color=color_boundary, linewidth=2.5, alpha=0.9, solid_capstyle='round')
# Lane B centerline
ax_main.plot([x2_end, x3_end], [lane_b_y, lane_b_y], 
             color=color_centerline, linestyle='--', linewidth=2, alpha=0.7, dashes=(8, 4))

# Lane C boundaries (y=0, width 3m)
lane_c_left = lane_c_y - lane_width / 2
lane_c_right = lane_c_y + lane_width / 2

# Lane C left boundary line
ax_main.plot([x4_end, x5_end], [lane_c_left, lane_c_left], 
             color=color_boundary, linewidth=2.5, alpha=0.9, solid_capstyle='round')
# Lane C right boundary line
ax_main.plot([x4_end, x5_end], [lane_c_right, lane_c_right], 
             color=color_boundary, linewidth=2.5, alpha=0.9, solid_capstyle='round')
# Lane C centerline
ax_main.plot([x4_end, x5_end], [lane_c_y, lane_c_y], 
             color=color_centerline, linestyle='--', linewidth=2, alpha=0.7, dashes=(8, 4))

# Note: Curve sections do not draw road boundaries (transition zones)

# === 5.5. Draw Distance Marker at x=35 ===
# x=35 位于 B 道直线段内，标注 A 道和 B 道之间的 y 距离
marker_x = 35.0
lane_a_y_at_marker = lane_a_y  # A 道中心线 y 坐标
lane_b_y_at_marker = lane_b_y  # B 道中心线 y 坐标
lane_distance = abs(lane_b_y_at_marker - lane_a_y_at_marker)  # A-B 道之间的距离

# 绘制垂直参考线（虚线）
ax_main.plot([marker_x, marker_x], [y_min, y_max], 
             color='#9b59b6', linestyle='--', linewidth=1.8, alpha=0.6, 
             label='Reference Line (x=35m)', dashes=(10, 5))

# 绘制 A 道中心线标记点
ax_main.plot(marker_x, lane_a_y_at_marker, 'o', color='#e74c3c', 
             markersize=9, markeredgewidth=1.5, markeredgecolor='white', 
             zorder=12, label='Lane A Center')
ax_main.text(marker_x + 0.4, lane_a_y_at_marker, 'A', 
             fontsize=9, verticalalignment='center', color='#e74c3c', 
             fontweight='bold', style='italic')

# 绘制 B 道中心线标记点
ax_main.plot(marker_x, lane_b_y_at_marker, 'o', color='#3498db', 
             markersize=9, markeredgewidth=1.5, markeredgecolor='white',
             zorder=12, label='Lane B Center')
ax_main.text(marker_x + 0.4, lane_b_y_at_marker, 'B', 
             fontsize=9, verticalalignment='center', color='#3498db', 
             fontweight='bold', style='italic')

# 绘制距离标注（双箭头）
# 计算标注线的 x 偏移位置
annotation_x_offset = 2.0
annotation_x = marker_x + annotation_x_offset
# 绘制双箭头标注
distance_arrow = FancyArrowPatch((annotation_x, lane_a_y_at_marker), 
                                 (annotation_x, lane_b_y_at_marker),
                                 arrowstyle='<->', mutation_scale=22, 
                                 linewidth=2.5, color='#27ae60', zorder=13)
ax_main.add_patch(distance_arrow)

# 添加距离文字标注
mid_y = (lane_a_y_at_marker + lane_b_y_at_marker) / 2
ax_main.text(annotation_x + 0.5, mid_y, f'Δy = {lane_distance:.2f} m', 
             fontsize=9.5, verticalalignment='center', horizontalalignment='left',
             bbox=dict(boxstyle='round,pad=0.35', facecolor='#ffffcc', 
                      alpha=0.92, edgecolor='#27ae60', linewidth=1.5),
             fontweight='bold', color='#2c3e50')

# === 6. Draw Reference Trajectory (Target Path) ===
ref_trajectory_line, = ax_main.plot([], [], color='#16a085', linestyle='--', 
                                    linewidth=2.5, alpha=0.75, label='Reference Trajectory',
                                    dashes=(10, 5))

# === 7. Data Plot Areas ===
# 注释掉所有数据子图
# # 为所有子图设置背景色
# subplot_bg = '#fafafa'

# # Velocity plot
# ax_v = fig.add_subplot(gs[1, 0])
# ax_v.set_facecolor(subplot_bg)
# ax_v.set_xlabel("Time Step", fontsize=11, fontweight='bold')
# ax_v.set_ylabel("Velocity, v (m/s)", fontsize=11, fontweight='bold')
# ax_v.set_title("(a) Longitudinal Velocity", fontsize=12, fontweight='bold', pad=10)
# ax_v.grid(True, alpha=0.35, linewidth=0.8, which='major')
# ax_v.minorticks_on()
# ax_v.grid(True, alpha=0.15, linewidth=0.5, which='minor')

# # Suspension vertical displacement plot
# ax_Zs = fig.add_subplot(gs[1, 1])
# ax_Zs.set_facecolor(subplot_bg)
# ax_Zs.set_xlabel("Time Step", fontsize=11, fontweight='bold')
# ax_Zs.set_ylabel("Vertical Displacement, $Z_s$ (m)", fontsize=11, fontweight='bold')
# ax_Zs.set_title("(b) Body Vertical Displacement", fontsize=12, fontweight='bold', pad=10)
# ax_Zs.grid(True, alpha=0.35, linewidth=0.8, which='major')
# ax_Zs.minorticks_on()
# ax_Zs.grid(True, alpha=0.15, linewidth=0.5, which='minor')

# # Roll angle plot
# ax_phi = fig.add_subplot(gs[2, 0])
# ax_phi.set_facecolor(subplot_bg)
# ax_phi.set_xlabel("Time Step", fontsize=11, fontweight='bold')
# ax_phi.set_ylabel("Roll Angle, $\\phi$ (deg)", fontsize=11, fontweight='bold')
# ax_phi.set_title("(c) Vehicle Roll Angle", fontsize=12, fontweight='bold', pad=10)
# ax_phi.grid(True, alpha=0.35, linewidth=0.8, which='major')
# ax_phi.minorticks_on()
# ax_phi.grid(True, alpha=0.15, linewidth=0.5, which='minor')

# # Pitch angle plot
# ax_theta = fig.add_subplot(gs[2, 1])
# ax_theta.set_facecolor(subplot_bg)
# ax_theta.set_xlabel("Time Step", fontsize=11, fontweight='bold')
# ax_theta.set_ylabel("Pitch Angle, $\\theta$ (deg)", fontsize=11, fontweight='bold')
# ax_theta.set_title("(d) Vehicle Pitch Angle", fontsize=12, fontweight='bold', pad=10)
# ax_theta.grid(True, alpha=0.35, linewidth=0.8, which='major')
# ax_theta.minorticks_on()
# ax_theta.grid(True, alpha=0.15, linewidth=0.5, which='minor')

# # LLTR plots (front and rear) - 若数据存在则渲染
# ax_lltr_f = fig.add_subplot(gs[3, 0])
# ax_lltr_f.set_facecolor(subplot_bg)
# ax_lltr_f.set_xlabel("Time Step", fontsize=11, fontweight='bold')
# ax_lltr_f.set_ylabel("Lateral Deviation, $e_y$ (m)", fontsize=11, fontweight='bold')
# ax_lltr_f.set_title("(e) Lateral Deviation", fontsize=12, fontweight='bold', pad=10)
# ax_lltr_f.grid(True, alpha=0.35, linewidth=0.8, which='major')
# ax_lltr_f.minorticks_on()
# ax_lltr_f.grid(True, alpha=0.15, linewidth=0.5, which='minor')

# ax_lltr_r = fig.add_subplot(gs[3, 1])
# ax_lltr_r = None
# ax_beta = fig.add_subplot(gs[3, 1])
# ax_beta.set_facecolor(subplot_bg)
# ax_beta.set_xlabel("Time Step", fontsize=11, fontweight='bold')
# ax_beta.set_ylabel("Sideslip Angle, $\\beta$ (deg)", fontsize=11, fontweight='bold')
# ax_beta.set_title("(f) Sideslip Angle", fontsize=12, fontweight='bold', pad=10)
# ax_beta.grid(True, alpha=0.35, linewidth=0.8, which='major')
# ax_beta.minorticks_on()
# ax_beta.grid(True, alpha=0.15, linewidth=0.5, which='minor')

# # Steering angle plot (delta)
# ax_delta = fig.add_subplot(gs[4, 0])
# ax_delta.set_facecolor(subplot_bg)
# ax_delta.set_xlabel("Time Step", fontsize=11, fontweight='bold')
# ax_delta.set_ylabel("Steering Angle, $\\delta$ (deg)", fontsize=11, fontweight='bold')
# ax_delta.set_title("(g) Steering Angle", fontsize=12, fontweight='bold', pad=10)
# ax_delta.grid(True, alpha=0.35, linewidth=0.8, which='major')
# ax_delta.minorticks_on()
# ax_delta.grid(True, alpha=0.15, linewidth=0.5, which='minor')

# # Suspension forces plot (f1, f2, f3, f4)
# ax_forces = None

# === 8. Initialize Plot Elements ===
# 定义专业配色
color_trajectory = '#2980b9'  # 蓝色轨迹
color_vehicle = '#34495e'     # 深灰色车辆
color_arrow = '#f1c40f'       # 金黄色箭头

# Vehicle trajectory line
trajectory_line, = ax_main.plot([], [], color=color_trajectory, linestyle='-', 
                                linewidth=2.5, alpha=0.85, label='Actual Trajectory',
                                solid_capstyle='round')

# Vehicle polygon (using polygon to support rotation) - 更真实的车辆形状
car_length = 3.6
car_width = 1.8
# 创建更像车辆的轮廓：前部略圆，后部略方
# 车辆轮廓点（相对于中心，前部在+x方向）
car_body_points = np.array([
    [-car_length/2, -car_width/2],          # 后左下
    [car_length/2 - 0.3, -car_width/2],     # 前左下（前部略收窄）
    [car_length/2, -car_width/3],           # 前下（圆角）
    [car_length/2, car_width/3],            # 前上（圆角）
    [car_length/2 - 0.3, car_width/2],      # 前右上（前部略收窄）
    [-car_length/2, car_width/2],           # 后右上
])

car_polygon = Polygon(car_body_points, 
                      facecolor=color_vehicle, edgecolor='white', 
                      linewidth=2.5, zorder=10, alpha=0.95, label='Vehicle Body')
ax_main.add_patch(car_polygon)

# 添加前挡风玻璃（梯形）
windshield_points = np.array([
    [car_length/2 - 0.5, -car_width/4],
    [car_length/2 - 0.2, -car_width/6],
    [car_length/2 - 0.2, car_width/6],
    [car_length/2 - 0.5, car_width/4],
])
windshield = Polygon(windshield_points, 
                     facecolor='#87ceeb', edgecolor='#4682b4', 
                     linewidth=1.5, zorder=12, alpha=0.6)
ax_main.add_patch(windshield)

# Vehicle direction arrow
arrow = ax_main.annotate('', xy=(0, 0), xytext=(0, 0),
                         arrowprops=dict(arrowstyle='->', lw=3.5, color=color_arrow,
                                       mutation_scale=25),
                         zorder=11)

# Information text - 放在左下角，避免遮挡道路和车辆轨迹
info_text = ax_main.text(0.02, 0.02, '', transform=ax_main.transAxes,
                         fontsize=10, verticalalignment='bottom', horizontalalignment='left',
                         bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                                 alpha=0.85, edgecolor='gray', linewidth=1.5),
                         fontfamily='monospace')

# Vehicle speed text (跟随车辆移动) - 降低透明度
speed_text = ax_main.text(0, 0, '', fontsize=12, fontweight='bold',
                          verticalalignment='bottom', horizontalalignment='center',
                          bbox=dict(boxstyle='round,pad=0.5', facecolor='#e8f4f8', 
                                   alpha=0.6, edgecolor='#2980b9', linewidth=2),
                          color='#2c3e50', zorder=15)

# Steering angle text (跟随车辆移动，显示转向角δ)
steer_text = ax_main.text(0, 0, '', fontsize=11, fontweight='bold',
                          verticalalignment='bottom', horizontalalignment='center',
                          bbox=dict(boxstyle='round,pad=0.45', facecolor='#fff5e6',
                                    alpha=0.95, edgecolor='#e67e22', linewidth=2),
                          color='#7f8c8d', zorder=15)

# Data plot lines - 注释掉所有数据图线
# v_line, = ax_v.plot([], [], linewidth=2.5, color='#2c3e50', label='Velocity', 
#                     solid_capstyle='round')
# Zs_line, = ax_Zs.plot([], [], linewidth=2.5, color='#27ae60', label='$Z_s$',
#                       solid_capstyle='round')
# phi_line, = ax_phi.plot([], [], linewidth=2.5, color='#e74c3c', label='$\\phi$',
#                         solid_capstyle='round')
# theta_line, = ax_theta.plot([], [], linewidth=2.5, color='#8e44ad', label='$\\theta$',
#                             solid_capstyle='round')

# # LLTR plot lines
# lltrf_line, = ax_lltr_f.plot([], [], linewidth=2.5, color='#1abc9c', label='Lateral Deviation',
#                              solid_capstyle='round')
# beta_line, = ax_beta.plot([], [], linewidth=2.5, color='#2ecc71', label='$\\beta$ (deg)',
#                           solid_capstyle='round')

# # Steering angle plot line
# delta_line, = ax_delta.plot([], [], linewidth=2.5, color='#e67e22', label='$\\delta$',
#                              solid_capstyle='round')

# beta_legend_params = {'loc': 'upper right', 'fontsize': 10, 'framealpha': 0.95,
#                       'edgecolor': 'gray', 'fancybox': True, 'shadow': True}
# ax_beta.legend(**beta_legend_params)

# # 添加图例 - 统一样式
# legend_params = {'loc': 'upper right', 'fontsize': 10, 'framealpha': 0.95, 
#                 'edgecolor': 'gray', 'fancybox': True, 'shadow': True}
# ax_v.legend(**legend_params)
# ax_Zs.legend(**legend_params)
# ax_phi.legend(**legend_params)
# ax_theta.legend(**legend_params)
# ax_lltr_f.legend(**legend_params)
# ax_delta.legend(**legend_params)

# 主图图例放在底部中央，避免遮挡道路
ax_main.legend(loc='upper center', fontsize=9.5, framealpha=0.95, 
              edgecolor='gray', fancybox=True, shadow=True, ncol=4,
              bbox_to_anchor=(0.5, -0.02))

# Set data plot ranges - 注释掉所有数据图范围设置
# frames = np.arange(len(x_data))
# ax_v.set_xlim(0, len(x_data))
# ax_Zs.set_xlim(0, len(x_data))
# ax_phi.set_xlim(0, len(x_data))
# ax_theta.set_xlim(0, len(x_data))
# ax_lltr_f.set_xlim(0, len(x_data))
# ax_delta.set_xlim(0, len(x_data))
# ax_beta.set_xlim(0, len(x_data))

# # Set y-axis ranges (with some margin)
# # 速度图y轴固定为0-30 m/s
# ax_v.set_ylim(0, 30)

# ax_Zs.set_ylim(-0.5, 0.2)

# ax_phi.set_ylim(-10, 10)

# ax_theta.set_ylim(-10, 10)

# # Steering angle y-axis range: -45 to 45 degrees
# ax_delta.set_ylim(-45, 45)

# # LLTR y-axis ranges
# ax_lltr_f.set_ylim(-0.4, 0.4)
# ax_beta.set_ylim(-10, 10)

# # Current value indicator lines - 使用统一的高可见度样式
# indicator_color = '#c0392b'
# indicator_style = {'color': indicator_color, 'linestyle': '--', 'alpha': 0.65, 
#                   'linewidth': 2.2, 'dashes': (5, 3)}

# v_vline, = ax_v.plot([0, 0], ax_v.get_ylim(), **indicator_style, label='Current Position')
# Zs_vline, = ax_Zs.plot([0, 0], ax_Zs.get_ylim(), **indicator_style)
# phi_vline, = ax_phi.plot([0, 0], ax_phi.get_ylim(), **indicator_style)
# theta_vline, = ax_theta.plot([0, 0], ax_theta.get_ylim(), **indicator_style)
# lltrf_vline, = ax_lltr_f.plot([0, 0], ax_lltr_f.get_ylim(), **indicator_style)
# delta_vline, = ax_delta.plot([0, 0], ax_delta.get_ylim(), **indicator_style)
# beta_vline, = ax_beta.plot([0, 0], ax_beta.get_ylim(), **indicator_style)

# === 9. 初始化和更新函数 ===
def init():
    """初始化动画"""
    trajectory_line.set_data([], [])
    ref_trajectory_line.set_data([], [])
    car_polygon.set_xy(car_body_points)
    # 初始化挡风玻璃
    windshield.set_xy(windshield_points)
    # v_line.set_data([], [])
    # Zs_line.set_data([], [])
    # phi_line.set_data([], [])
    # theta_line.set_data([], [])
    # delta_line.set_data([], [])
    # beta_line.set_data([], [])
    info_text.set_text('')
    speed_text.set_text('')
    steer_text.set_text('')
    return (trajectory_line, ref_trajectory_line, car_polygon, arrow, info_text, speed_text, steer_text,
            windshield)
            # v_line, Zs_line, phi_line, theta_line,
            # lltrf_line, beta_line, delta_line,
            # v_vline, Zs_vline, phi_vline, theta_vline, lltrf_vline, beta_vline, delta_vline)

def update(frame):
    # 获取当前帧数据
    x = x_data[frame]
    y = y_data[frame]
    psi = psi_data[frame]  # 航向角（弧度）
    v = v_data[frame]
    Zs = Zs_data[frame]
    phi = phi_data[frame]
    theta = theta_data[frame]
    delta = delta_data[frame]
    
    # 更新实际轨迹线
    trajectory_line.set_data(x_data[:frame+1], y_data[:frame+1])
    
    # 更新参考轨迹线
    ref_trajectory_line.set_data(x_ref, y_ref)
    
    # 更新车辆位置和朝向
    dx = np.cos(psi)
    dy = np.sin(psi)
    
    # 旋转矩阵
    cos_psi = np.cos(psi)
    sin_psi = np.sin(psi)
    
    # 旋转并平移车辆主体
    rotated_body = np.zeros_like(car_body_points)
    for i, point in enumerate(car_body_points):
        rotated_body[i, 0] = x + point[0] * cos_psi - point[1] * sin_psi
        rotated_body[i, 1] = y + point[0] * sin_psi + point[1] * cos_psi
    
    # 更新车辆主体多边形顶点
    car_polygon.set_xy(rotated_body)
    
    # 更新挡风玻璃位置
    rotated_windshield = np.zeros_like(windshield_points)
    for i, point in enumerate(windshield_points):
        rotated_windshield[i, 0] = x + point[0] * cos_psi - point[1] * sin_psi
        rotated_windshield[i, 1] = y + point[0] * sin_psi + point[1] * cos_psi
    windshield.set_xy(rotated_windshield)
    
    # 更新方向箭头
    arrow_length = car_length * 0.6
    arrow_x = x + arrow_length * dx
    arrow_y = y + arrow_length * dy
    arrow.set_position((arrow_x, arrow_y))
    arrow.xy = (arrow_x, arrow_y)
    arrow.xytext = (x, y)
    
    # 更新速度文本（车辆上方）
    speed_offset = car_width / 2 + 0.8  # 车辆上方偏移量
    speed_text_x = x
    speed_text_y = y + speed_offset  # 在世界坐标系中，车辆上方即y轴正方向
    speed_text.set_position((speed_text_x, speed_text_y))
    speed_kmh = v * 3.6  # 转换为km/h
    speed_text.set_text(f'{speed_kmh:.1f} km/h')
    '''
    # 更新转向角文本（车辆上方，略高于速度文本避免重叠）
    steer_offset = car_width / 2 + 2.6
    steer_text_x = x
    steer_text_y = y + steer_offset
    steer_text.set_position((steer_text_x, steer_text_y))
    delta_deg = np.degrees(delta)
    steer_text.set_text(f'δ = {delta_deg:.1f}°')
    '''
    # 计算目标y位置和横向误差
    target_y = get_target_y(x)
    lateral_error = y - target_y
    
    # Determine current stage
    if x <= x1_end:
        stage = "Lane A Straight"
    elif x <= x2_end:
        stage = "Curve A-B"
    elif x <= x3_end:
        stage = "Lane B Straight"
    elif x <= x4_end:
        stage = "Curve B-C"
    else:
        stage = "Lane C Straight"
    
    # Update information text - 格式化为专业论文风格（紧凑版）
    info_text.set_text(
        f'═══════════════════════\n'
        f'Step: {frame:4d}/{len(x_data)-1}\n'
        f'───────────────────────\n'
        f'Position:\n'
        f'  X = {x:6.2f} m\n'
        f'  Y = {y:6.2f} m\n'
        f'───────────────────────\n'
        f'Stage: {stage}\n'
        f'───────────────────────\n'
        f'Target Y:  {target_y:6.2f} m\n'
        f'Lat. Err:  {lateral_error:+6.3f} m\n'
        f'───────────────────────\n'
        f'Vel: {v:5.2f} m/s\n'
        f'     ({v*3.6:5.1f} km/h)\n'
        f'═══════════════════════'
    )
    
    # Update data plots - 注释掉所有数据图更新
    # v_line.set_data(frames[:frame+1], v_data[:frame+1])
    # Zs_line.set_data(frames[:frame+1], Zs_data[:frame+1])
    # phi_line.set_data(frames[:frame+1], phi_deg_data[:frame+1])
    # theta_line.set_data(frames[:frame+1], theta_deg_data[:frame+1])
    # delta_line.set_data(frames[:frame+1], delta_deg_data[:frame+1])
    # # Update lateral deviation plots (replace LLTR)
    # lat_err_series = y_data[:frame+1] - y_target_full[:frame+1]
    # lltrf_line.set_data(frames[:frame+1], lat_err_series)
    # # Update beta plot
    # beta_line.set_data(frames[:frame+1], beta_deg_data[:frame+1])
    
    # # Update vertical indicator lines
    # v_vline.set_data([frame, frame], ax_v.get_ylim())
    # Zs_vline.set_data([frame, frame], ax_Zs.get_ylim())
    # phi_vline.set_data([frame, frame], ax_phi.get_ylim())
    # theta_vline.set_data([frame, frame], ax_theta.get_ylim())
    # delta_vline.set_data([frame, frame], ax_delta.get_ylim())
    # lltrf_vline.set_data([frame, frame], ax_lltr_f.get_ylim())
    # beta_vline.set_data([frame, frame], ax_beta.get_ylim())
    
    return (trajectory_line, ref_trajectory_line, car_polygon, arrow, info_text, speed_text, steer_text,
            windshield)
            # v_line, Zs_line, phi_line, theta_line,
            # lltrf_line, beta_line, delta_line,
            # v_vline, Zs_vline, phi_vline, theta_vline, lltrf_vline, beta_vline, delta_vline)

# === 10. Create Animation ===
print("="*60)
print("Starting animation creation...")
print(f"Total frames: {len(x_data)}")
print(f"Figure size: {fig.get_size_inches()[0]:.1f}\" x {fig.get_size_inches()[1]:.1f}\"")
print(f"DPI: {fig.dpi}")
print("="*60)

# 使用更小的interval使动画更流畅（单位：毫秒）
# interval=20表示每帧之间间隔20ms，约50fps
# blit=True 启用快速渲染，仅更新变化的部分
ani = FuncAnimation(fig, update, init_func=init, frames=len(x_data), 
                   interval=1.5,blit=True, repeat=True, cache_frame_data=False)

# 调整布局以防止重叠
# plt.tight_layout(pad=1.5)  # 注释掉，因为使用了 add_axes 手动定位

# 添加整体标题（可选，用于论文）
fig.suptitle('Vehicle Lane Change Maneuver - Dynamic Simulation Results', 
             fontsize=16, fontweight='bold', y=0.98)  # 调整标题位置更靠近图形

plt.show()

print("="*60)
print("Visualization complete!")
print("="*60)

