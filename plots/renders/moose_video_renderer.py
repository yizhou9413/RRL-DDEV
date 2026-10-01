import argparse
from dataclasses import dataclass
from pathlib import Path
from shutil import which
from typing import List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib import animation
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

# ==========================================
# 核心数据列定义
# ==========================================
COL_NAMES_32 = [
    "x", "y", "psi", "x_dot", "y_dot", "psi_dot", "delta",
    "omega_fr", "omega_fl", "omega_rr", "omega_rl",
    "f1", "f2", "f3", "f4", "r", "beta", "v",
    "Zs", "phi", "theta", "Z11", "Z12", "Z13", "Z14",
    "dZs", "dphi", "dtheta", "dZ11", "dZ12", "dZ13", "dZ14",
]
COL_NAMES_38 = COL_NAMES_32 + [
    "LLTR_front", "LLTR_rear", "gamma_fr", "gamma_fl", "gamma_rr", "gamma_rl",
]
COL_NAMES_48 = COL_NAMES_38 + [
    "fx_fr", "fx_fl", "fx_rr", "fx_rl", "fy_fr", "fy_fl", "fy_rr", "fy_rl",
    "ddphi", "ddtheta",
]

# ==========================================
# 学术风格配色方案
# ==========================================
BG_COLOR = "#ffffff"          # 纯白背景，更适合学术报告
CARD_COLOR = "#f8f9fa"        # 浅灰面板
ROAD_FILL = "#f1f3f5"         # 车道填充色
ROAD_EDGE = "#868e96"         # 车道边缘线
CENTERLINE_COLOR = "#adb5bd"  # 中心参考线
REFERENCE_COLOR = "#ced4da"   # 规划轨迹对比线
TRAJECTORY_COLOR = "#4A7AFF"  # 主轨迹颜色 (明亮蓝)
RECENT_COLOR = "#FF8C00"      # 最近轨迹/高亮颜色 (深橙色)
CAR_BODY_COLOR = "#495057"    # 车身主体颜色
WINDSHIELD_COLOR = "#e9ecef"  # 挡风玻璃颜色
WHEEL_COLOR = "#212529"       # 车轮颜色
TEXT_COLOR = "#343a40"
MUTED_TEXT = "#6c757d"
GRID_COLOR = "#e9ecef"

# 曲线面板颜色
SPEED_COLOR = "#4A7AFF"
ERROR_COLOR = "#FF8C00"
BETA_COLOR = "#17a2b8"
ROLL_COLOR = "#d62828"
PITCH_COLOR = "#4f772d"
THRESHOLD_COLOR = "#dc3545"

@dataclass
class MooseTrackConfig:
    lane_a_length: float = 22.0
    curve_ab_length: float = 18.0
    lane_b_length: float = 11.0
    curve_bc_length: float = 16.0
    lane_c_length: float = 12.0
    lane_b_offset: float = 3.2
    lane_width: float = 3.0

    @property
    def x1_end(self) -> float: return self.lane_a_length
    @property
    def x2_end(self) -> float: return self.x1_end + self.curve_ab_length
    @property
    def x3_end(self) -> float: return self.x2_end + self.lane_b_length
    @property
    def x4_end(self) -> float: return self.x3_end + self.curve_bc_length
    @property
    def total_length(self) -> float: return self.x4_end + self.lane_c_length

@dataclass
class TrackSamples:
    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray

@dataclass
class TrajectoryData:
    source: Path
    frame_index: np.ndarray
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    speed_mps: np.ndarray
    speed_kph: np.ndarray
    delta_deg: np.ndarray
    phi_deg: np.ndarray
    theta_deg: np.ndarray
    beta_deg: np.ndarray
    lateral_error: np.ndarray

    @property
    def num_steps(self) -> int: return int(self.x.shape[0])

def configure_matplotlib() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 10,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#ced4da",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": "white",
        "grid.color": GRID_COLOR,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "savefig.facecolor": BG_COLOR,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a presentation-grade moose maneuver animation.")
    parser.add_argument("input_path", type=str, help="CSV file path or folder.")
    parser.add_argument("--output", type=str, default=None, help="Output path (.gif, .mp4).")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for batch-rendered videos.")
    parser.add_argument("--all-csv", action="store_true", help="Render all CSV files in the input folder.")
    parser.add_argument("--recursive", action="store_true", help="Search CSV files recursively when used with --all-csv.")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation step size.")
    parser.add_argument("--fps", type=int, default=30, help="Output animation frame rate.")
    parser.add_argument("--frame-stride", type=int, default=1, help="Render every Nth step.")
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI.")
    parser.add_argument("--trail-length", type=int, default=60, help="Recent steps highlighted.")
    parser.add_argument("--zoom-width", type=float, default=20.0, help="Zoom-panel width.")
    parser.add_argument("--zoom-height", type=float, default=12.0, help="Zoom-panel height.")
    parser.add_argument("--angle-threshold", type=float, default=10.0, help="Threshold line for angles.")
    parser.add_argument("--title", type=str, default=None, help="Optional figure title.")
    return parser

def resolve_input_csv(input_path: str) -> Path:
    path = Path(input_path).expanduser().resolve()
    if path.is_file() and path.suffix.lower() == ".csv": return path
    if not path.is_dir(): raise FileNotFoundError(f"Path not found: {path}")
    csv_files = sorted(path.glob("*.csv"), key=lambda p: p.stat().st_mtime)
    if not csv_files: raise FileNotFoundError(f"No CSVs in {path}")
    return csv_files[-1]

def collect_input_csvs(input_path: str, recursive: bool = False) -> Tuple[Path, List[Path]]:
    path = Path(input_path).expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() != ".csv":
            raise FileNotFoundError(f"CSV file not found: {path}")
        return path.parent, [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")

    globber = path.rglob if recursive else path.glob
    csv_files = sorted((p for p in globber("*.csv") if p.is_file()), key=lambda p: str(p.relative_to(path)))
    if not csv_files:
        scope = "recursively" if recursive else "in"
        raise FileNotFoundError(f"No CSVs found {scope} {path}")
    return path, csv_files

def build_batch_output_path(csv_path: Path, input_root: Path, output_dir: Path) -> Path:
    relative_stem = csv_path.relative_to(input_root).with_suffix("")
    safe_name = "__".join(relative_stem.parts)
    return output_dir / f"{safe_name}_render.mp4"

def render_csv_to_path(csv_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    track = MooseTrackConfig()
    data, samples = load_trajectory(csv_path, args.dt, track)
    render_animation(
        data,
        track,
        samples,
        output_path,
        args.fps,
        max(1, args.frame_stride),
        args.dpi,
        args.trail_length,
        args.zoom_width,
        args.zoom_height,
        args.angle_threshold,
        args.title,
    )

def assign_column_names(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = len(df.columns)
    if num_cols >= len(COL_NAMES_48): base_names = COL_NAMES_48
    elif num_cols >= len(COL_NAMES_38): base_names = COL_NAMES_38
    elif num_cols >= 21: base_names = COL_NAMES_32
    else: raise ValueError(f"CSV has {num_cols} cols. Need >= 21.")
    df.columns = base_names[:num_cols] + [f"col_{i}" for i in range(len(base_names), num_cols)]
    return df

def reference_y(track: MooseTrackConfig, x_values: np.ndarray) -> np.ndarray:
    x_array = np.asarray(x_values, dtype=float)
    y = np.zeros_like(x_array)
    m2 = (x_array > track.x1_end) & (x_array <= track.x2_end)
    if np.any(m2): y[m2] = track.lane_b_offset * (1.0 - np.cos(np.pi * (x_array[m2] - track.x1_end) / track.curve_ab_length)) / 2.0
    m3 = (x_array > track.x2_end) & (x_array <= track.x3_end)
    y[m3] = track.lane_b_offset
    m4 = (x_array > track.x3_end) & (x_array <= track.x4_end)
    if np.any(m4): y[m4] = track.lane_b_offset * (1.0 + np.cos(np.pi * (x_array[m4] - track.x3_end) / track.curve_bc_length)) / 2.0
    return y

def build_track_samples(track: MooseTrackConfig, x_stop: float) -> TrackSamples:
    x = np.linspace(-2.0, max(track.total_length, x_stop) + 5.0, 2000)
    y = reference_y(track, x)
    heading = np.arctan(np.gradient(y, x))
    hw = track.lane_width / 2.0
    return TrackSamples(
        x=x, y=y, heading=heading,
        left_x=x - hw * np.sin(heading), left_y=y + hw * np.cos(heading),
        right_x=x + hw * np.sin(heading), right_y=y - hw * np.cos(heading)
    )

def load_trajectory(csv_path: Path, dt: float, track: MooseTrackConfig) -> Tuple[TrajectoryData, TrackSamples]:
    df = pd.read_csv(csv_path, header=None)
    df = assign_column_names(df)
    x = df["x"].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    psi = df["psi"].to_numpy(dtype=float)
    speed_mps = df["v"].to_numpy(dtype=float) if "v" in df.columns else np.hypot(df["x_dot"], df["y_dot"])
    
    get_deg = lambda col: np.degrees(df[col].to_numpy(dtype=float)) if col in df.columns else np.zeros_like(x)
    track_samples = build_track_samples(track, float(np.max(x)))
    lateral_error = y - np.interp(x, track_samples.x, track_samples.y)
    
    return TrajectoryData(
        source=csv_path, frame_index=np.arange(len(x)), time=np.arange(len(x)) * dt,
        x=x, y=y, psi=psi, speed_mps=speed_mps, speed_kph=speed_mps * 3.6,
        delta_deg=get_deg("delta"), phi_deg=get_deg("phi"),
        theta_deg=get_deg("theta"), beta_deg=get_deg("beta"),
        lateral_error=lateral_error
    ), track_samples

# ==========================================
# 重构：更加逼真的车辆几何模型
# ==========================================
def create_vehicle_shapes(length: float = 4.2, width: float = 1.86) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """返回车身、挡风玻璃和四个车轮的坐标，形成更真实的俯视图轮廓"""
    # 流线型车身 (稍微收窄车头和车尾)
    body = np.array([
        [-length * 0.45, -width * 0.46],  # 右后
        [length * 0.40, -width * 0.46],   # 右前
        [length * 0.50, -width * 0.35],   # 车头右侧收缩
        [length * 0.50, width * 0.35],    # 车头左侧收缩
        [length * 0.40, width * 0.46],    # 左前
        [-length * 0.45, width * 0.46],   # 左后
    ])
    
    # 挡风玻璃 (梯形)
    windshield = np.array([
        [length * 0.05, -width * 0.38],
        [length * 0.25, -width * 0.32],
        [length * 0.25, width * 0.32],
        [length * 0.05, width * 0.38],
    ])
    
    # 四个独立车轮
    wheel_l, wheel_w = 0.65, 0.25
    fw_x, rw_x = length * 0.28, -length * 0.28
    tw_y = width * 0.5 + 0.02 # 微微突出车身
    
    def make_wheel(cx, cy):
        return np.array([
            [cx - wheel_l/2, cy - wheel_w/2],
            [cx + wheel_l/2, cy - wheel_w/2],
            [cx + wheel_l/2, cy + wheel_w/2],
            [cx - wheel_l/2, cy + wheel_w/2],
        ])
        
    wheels = [make_wheel(fw_x, -tw_y), make_wheel(fw_x, tw_y), 
              make_wheel(rw_x, -tw_y), make_wheel(rw_x, tw_y)]
              
    return body, windshield, wheels

def transform_shape(points: np.ndarray, x: float, y: float, psi: float) -> np.ndarray:
    c, s = np.cos(psi), np.sin(psi)
    return points @ np.array([[c, -s], [s, c]]).T + np.array([x, y])

def style_signal_axis(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, loc="left", color=TEXT_COLOR)
    ax.set_ylabel(ylabel)
    ax.grid(True, which="major", alpha=0.6)
    ax.minorticks_on()
    ax.tick_params(colors=MUTED_TEXT, direction="in")

def padded_limits(values: Sequence[float], pad_ratio: float = 0.1, min_span: float = 1.0) -> Tuple[float, float]:
    vmin, vmax = float(np.min(values)), float(np.max(values))
    pad = max(vmax - vmin, min_span) * pad_ratio
    return vmin - pad, vmax + pad

def render_animation(data: TrajectoryData, track: MooseTrackConfig, samples: TrackSamples, 
                     output_path: Path, fps: int, stride: int, dpi: int, trail_len: int,
                     z_width: float, z_height: float, angle_thresh: float, title: Optional[str]) -> None:
    
    indices = np.arange(0, data.num_steps, max(1, stride))
    if indices[-1] != data.num_steps - 1: indices = np.append(indices, data.num_steps - 1)
    
    body_base, wind_base, wheels_base = create_vehicle_shapes()
    
    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1.5, 1.0], height_ratios=[2.2, 1.4], wspace=0.2, hspace=0.25)
    
    ax_main = fig.add_subplot(gs[0, :2])
    ax_zoom = fig.add_subplot(gs[1, :2])
    right_gs = gs[:, 2].subgridspec(5, 1, height_ratios=[1.2, 1.0, 1.0, 1.0, 1.2], hspace=0.3)
    ax_info, ax_speed, ax_err, ax_beta, ax_ang = [fig.add_subplot(right_gs[i, 0]) for i in range(5)]

    # 绘制静态背景地图
    road_poly = np.vstack([np.column_stack([samples.left_x, samples.left_y]), 
                           np.column_stack([samples.right_x[::-1], samples.right_y[::-1]])])
    
    for ax in (ax_main, ax_zoom):
        ax.add_patch(Polygon(road_poly, closed=True, facecolor=ROAD_FILL, edgecolor="none", zorder=0))
        ax.plot(samples.left_x, samples.left_y, color=ROAD_EDGE, lw=1.5, zorder=1)
        ax.plot(samples.right_x, samples.right_y, color=ROAD_EDGE, lw=1.5, zorder=1)
        ax.plot(samples.x, samples.y, color=CENTERLINE_COLOR, lw=1.2, ls="--", zorder=2)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.4)

    ax_main.set_xlim(min(0.0, float(np.min(data.x))) - 2.0, track.total_length + 4.0)
    ax_main.set_ylim(*padded_limits(np.concatenate([samples.left_y, samples.right_y]), 0.15, 8.0))
    ax_main.set_title("Global Trajectory Evaluation", loc="left", color=TEXT_COLOR)

    # ==========================================
    # 修复核心点：利用混合坐标系解决标签遮挡问题
    # ==========================================
    stage_centers = [
        (track.x1_end / 2.0, "Lane A"),
        ((track.x1_end + track.x2_end) / 2.0, "A -> B"),
        ((track.x2_end + track.x3_end) / 2.0, "Lane B"),
        ((track.x3_end + track.x4_end) / 2.0, "B -> C"),
        ((track.x4_end + track.total_length) / 2.0, "Lane C"),
    ]
    # transData 决定 X 的位置，transAxes 决定 Y 的位置（1.02代表在坐标轴顶部外侧上方）
    blend_trans = mtransforms.blended_transform_factory(ax_main.transData, ax_main.transAxes)
    for cx, label in stage_centers:
        ax_main.text(cx, 1.03, label, transform=blend_trans, ha="center", va="bottom",
                     fontsize=9, color=TEXT_COLOR,
                     bbox=dict(boxstyle="round,pad=0.3", facecolor=CARD_COLOR, edgecolor=ROAD_EDGE, alpha=1.0))
        
    for sep in (track.x1_end, track.x2_end, track.x3_end, track.x4_end):
        ax_main.axvline(sep, color=ROAD_EDGE, lw=1.0, ls=":", zorder=1)

    ax_zoom.set_title("Vehicle Dynamic States", loc="left", color=TEXT_COLOR)

    # 初始化动态绘图元素
    main_recent, = ax_main.plot([], [], color=RECENT_COLOR, lw=2.5, zorder=5)
    zoom_recent, = ax_zoom.plot([], [], color=RECENT_COLOR, lw=3.0, zorder=4)
    
    # 渲染车辆补丁组
    def create_vehicle_patches(ax):
        body = Polygon(body_base, closed=True, fc=CAR_BODY_COLOR, ec="white", lw=1.0, zorder=7)
        wind = Polygon(wind_base, closed=True, fc=WINDSHIELD_COLOR, ec="none", zorder=8)
        wheels = [Polygon(w, closed=True, fc=WHEEL_COLOR, zorder=6) for w in wheels_base]
        ax.add_patch(body)
        ax.add_patch(wind)
        for w in wheels: ax.add_patch(w)
        return body, wind, wheels

    m_body, m_wind, m_wheels = create_vehicle_patches(ax_main)
    z_body, z_wind, z_wheels = create_vehicle_patches(ax_zoom)
    # 右侧信息与曲线面板初始化
    ax_info.set_axis_off()
    ax_info.text(0.05, 0.85, "Simulation Metrics", transform=ax_info.transAxes, fontsize=11, fontweight="bold")
    info_text = ax_info.text(0.05, 0.65, "", transform=ax_info.transAxes, fontsize=9, fontfamily="monospace", va="top")
    
    style_signal_axis(ax_speed, "Velocity", "km/h")
    style_signal_axis(ax_err, "Lat. Error", "m")
    style_signal_axis(ax_beta, "Sideslip (β)", "deg")
    style_signal_axis(ax_ang, "Attitude", "deg")
    
    for ax in [ax_speed, ax_err, ax_beta]: ax.tick_params(labelbottom=False)
    ax_ang.set_xlabel("Time (s)")
    t_max = float(data.time[-1]) if data.num_steps > 1 else 1.0
    for ax in [ax_speed, ax_err, ax_beta, ax_ang]: ax.set_xlim(0, t_max)

    spd_bg, = ax_speed.plot(data.time, data.speed_kph, color="#dee2e6", lw=1.5)
    spd_fg, = ax_speed.plot([], [], color=SPEED_COLOR, lw=2.0)
    spd_cur = ax_speed.axvline(0, color=MUTED_TEXT, ls=":", lw=1.0)
    ax_speed.set_ylim(*padded_limits(data.speed_kph, 0.1, 5.0))

    err_bg, = ax_err.plot(data.time, data.lateral_error, color="#dee2e6", lw=1.5)
    err_fg, = ax_err.plot([], [], color=ERROR_COLOR, lw=2.0)
    err_cur = ax_err.axvline(0, color=MUTED_TEXT, ls=":", lw=1.0)
    ax_err.axhline(0, color=MUTED_TEXT, ls="--", lw=0.8)
    ax_err.set_ylim(*padded_limits(data.lateral_error, 0.15, 1.0))

    beta_bg, = ax_beta.plot(data.time, data.beta_deg, color="#dee2e6", lw=1.5)
    beta_fg, = ax_beta.plot([], [], color=BETA_COLOR, lw=2.0)
    beta_cur = ax_beta.axvline(0, color=MUTED_TEXT, ls=":", lw=1.0)
    ax_beta.axhline(angle_thresh, color=THRESHOLD_COLOR, ls="--", lw=0.8)
    ax_beta.axhline(-angle_thresh, color=THRESHOLD_COLOR, ls="--", lw=0.8)
    ax_beta.set_ylim(*padded_limits(np.concatenate([data.beta_deg, [-angle_thresh, angle_thresh]]), 0.15, 4.0))

    ang_bg_phi, = ax_ang.plot(data.time, data.phi_deg, color="#dee2e6", lw=1.5)
    ang_bg_th, = ax_ang.plot(data.time, data.theta_deg, color="#e9ecef", lw=1.5)
    ang_fg_phi, = ax_ang.plot([], [], color=ROLL_COLOR, lw=2.0, label="Roll (φ)")
    ang_fg_th, = ax_ang.plot([], [], color=PITCH_COLOR, lw=2.0, label="Pitch (θ)")
    ang_cur = ax_ang.axvline(0, color=MUTED_TEXT, ls=":", lw=1.0)
    ax_ang.set_ylim(*padded_limits(np.concatenate([data.phi_deg, data.theta_deg, [-angle_thresh, angle_thresh]]), 0.15, 5.0))
    ax_ang.legend(loc="upper right", fontsize=8, framealpha=0.9)

    def update(frame: int):
        idx = int(indices[frame])
        start_idx = max(0, idx - trail_len)
        x, y, psi = float(data.x[idx]), float(data.y[idx]), float(data.psi[idx])
        t = float(data.time[idx])

        # 轨迹更新
        main_recent.set_data(data.x[start_idx:idx+1], data.y[start_idx:idx+1])
        zoom_recent.set_data(data.x[start_idx:idx+1], data.y[start_idx:idx+1])

        # 车辆位姿更新函数
        def update_vehicle(body, wind, wheels):
            body.set_xy(transform_shape(body_base, x, y, psi))
            wind.set_xy(transform_shape(wind_base, x, y, psi))
            for i, w in enumerate(wheels):
                w.set_xy(transform_shape(wheels_base[i], x, y, psi))

        update_vehicle(m_body, m_wind, m_wheels)
        update_vehicle(z_body, z_wind, z_wheels)

        # 镜头平移控制
        ax_zoom.set_xlim(x - z_width/2, x + z_width/2)
        ax_zoom.set_ylim(y - z_height/2, y + z_height/2)

        # 数据面板更新
        info_text.set_text(
            f"Time:      {t:6.2f} s\n"
            f"Speed:     {data.speed_kph[idx]:6.1f} km/h\n"
            f"Lat Err:   {data.lateral_error[idx]:+6.2f} m\n"
            f"Sideslip:  {data.beta_deg[idx]:+6.2f} deg\n"
            f"Roll Max:  {np.max(np.abs(data.phi_deg[:idx+1])):6.2f} deg\n"
            f"Pitch Max: {np.max(np.abs(data.theta_deg[:idx+1])):6.2f} deg"
        )
        
        spd_fg.set_data(data.time[:idx+1], data.speed_kph[:idx+1])
        err_fg.set_data(data.time[:idx+1], data.lateral_error[:idx+1])
        beta_fg.set_data(data.time[:idx+1], data.beta_deg[:idx+1])
        ang_fg_phi.set_data(data.time[:idx+1], data.phi_deg[:idx+1])
        ang_fg_th.set_data(data.time[:idx+1], data.theta_deg[:idx+1])
        
        spd_cur.set_xdata([t, t])
        err_cur.set_xdata([t, t])
        beta_cur.set_xdata([t, t])
        ang_cur.set_xdata([t, t])

        return [] # Blit is false so we don't strictly need to return artists

    writer = animation.FFMpegWriter(fps=fps, bitrate=2500) if output_path.suffix == ".mp4" else animation.PillowWriter(fps=fps)
    anim = animation.FuncAnimation(fig, update, frames=len(indices), interval=1000/fps, blit=False)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(output_path), writer=writer, dpi=dpi)
    plt.close(fig)

def main():
    configure_matplotlib()
    parser = build_parser()
    args = parser.parse_args()

    if args.all_csv:
        if args.output and args.output_dir:
            parser.error("Use only one of --output or --output-dir with --all-csv.")
        if args.output:
            parser.error("--output expects a single file path. Use --output-dir with --all-csv.")

        input_root, csv_files = collect_input_csvs(args.input_path, recursive=args.recursive)
        output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (input_root / "rendered_videos")

        print(f"Found {len(csv_files)} CSV file(s).")
        print(f"Saving rendered videos to: {output_dir}")
        for idx, csv_path in enumerate(csv_files, start=1):
            out_path = build_batch_output_path(csv_path, input_root, output_dir)
            print(f"[{idx}/{len(csv_files)}] Rendering {csv_path} -> {out_path}")
            render_csv_to_path(csv_path, out_path, args)
        print(f"Batch render completed: {output_dir}")
        return

    csv_path = resolve_input_csv(args.input_path)
    out_path = Path(args.output).expanduser().resolve() if args.output else csv_path.with_name(f"{csv_path.stem}_render.mp4")
    render_csv_to_path(csv_path, out_path, args)
    print(f"Render completed: {out_path}")

if __name__ == "__main__":
    main()
