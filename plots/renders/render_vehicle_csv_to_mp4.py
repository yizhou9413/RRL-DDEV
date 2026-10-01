from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FFMpegWriter, FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Polygon, Circle


# ============================================================
# Top-level configuration: edit these defaults if you want to
# hard-code your own common paths, field aliases or styles.
# ============================================================
DEFAULT_INPUT_CSV = "runs_moose/a2c_basic_continuous_moose_30000_rpiq_0_distured_2_0p5_2500_25000_22000_0_0_0_0_0_2026/rollouts/batchtest_20260418_150436_008727_007_cuda_2_a2c_continuous_moose_30000_rpiq_0_distured_1_0p5/env0.csv"
DEFAULT_OUTPUT_MP4 = "vehicle_animation.mp4"
DEFAULT_FPS = 25
DEFAULT_DPI = 150  
DEFAULT_DT = 0.01  

DEFAULT_VEHICLE_LENGTH = 4.8
DEFAULT_VEHICLE_WIDTH = 1.9
DEFAULT_FOLLOW_VIEW = False
DEFAULT_FOLLOW_VIEW_WIDTH = 35.0
DEFAULT_FOLLOW_VIEW_HEIGHT = 18.0
DEFAULT_SCENE_PADDING = 6.0
DEFAULT_LANE_WIDTH = 3.0
DEFAULT_TRACKING_ERROR_ALERT = 0.8
DEFAULT_INPUT_ANGLES_IN_DEGREES = False
DEFAULT_SCENE_CONTENT_ZOOM = 1.3
DEFAULT_ATTITUDE_Y_LIM = (-10.0, 10.0)
DEFAULT_TRACKING_ERROR_Y_LIM = (-0.5, 0.5)
DEFAULT_VEHICLE_RENDER_SCALE = 0.6

DEFAULT_ROAD_MODE = "auto"

# ------------------------------------------------------------
# Field alias mapping for normal CSVs with header.
# ------------------------------------------------------------
FIELD_ALIASES: Dict[str, List[str]] = {
    "t": ["t", "time", "timestamp", "sim_time", "time_s"],
    "x": ["x", "pos_x", "x_pos", "ego_x"],
    "y": ["y", "pos_y", "y_pos", "ego_y"],
    "yaw": ["yaw", "psi", "heading", "heading_yaw"],
    "beta": ["beta", "side_slip_angle", "sideslip_angle", "slip_angle"],
    "speed": ["speed", "v", "vel", "vehicle_speed", "speed_mps"],
    "pitch": ["pitch", "theta"],
    "roll": ["roll", "phi"],
    "tracking_error": [
        "tracking_error",
        "track_error",
        "cte",
        "cross_track_error",
        "lateral_error",
        "e_y",
    ],
    "ref_x": ["ref_x", "x_ref", "target_x", "reference_x"],
    "ref_y": ["ref_y", "y_ref", "target_y", "reference_y"],
    "vehicle_length": ["vehicle_length", "length", "ego_length"],
    "vehicle_width": ["vehicle_width", "width", "ego_width"],
}

# ------------------------------------------------------------
# Moose env0 column mapping.
# ------------------------------------------------------------
ENV0_COLUMNS_32 = [
    "x", "y", "psi", "x_dot", "y_dot", "psi_dot", "delta", "omega_fr", 
    "omega_fl", "omega_rr", "omega_rl", "f1", "f2", "f3", "f4", "r", 
    "beta", "v", "Zs", "phi", "theta", "Z11", "Z12", "Z13", "Z14", 
    "dZs", "dphi", "dtheta", "dZ11", "dZ12", "dZ13", "dZ14",
]
ENV0_COLUMNS_38 = ENV0_COLUMNS_32 + [
    "LLTR_front", "LLTR_rear", "gamma_fr", "gamma_fl", "gamma_rr", "gamma_rl",
]
ENV0_COLUMNS_48 = ENV0_COLUMNS_38 + [
    "fx_fr", "fx_fl", "fx_rr", "fx_rl", "fy_fr", "fy_fl", "fy_rr", "fy_rl", 
    "ddphi", "ddtheta",
]


@dataclass
class MooseRoadConfig:
    lane_a_length: float = 22.0
    curve_ab_length: float = 18.0
    lane_b_length: float = 11.0
    curve_bc_length: float = 16.0
    lane_c_length: float = 12.0
    lane_b_offset: float = 3.2
    lane_width: float = DEFAULT_LANE_WIDTH

    @property
    def total_length(self) -> float:
        return (
            self.lane_a_length
            + self.curve_ab_length
            + self.lane_b_length
            + self.curve_bc_length
            + self.lane_c_length
        )


@dataclass
class RoadGeometry:
    center_x: np.ndarray
    center_y: np.ndarray
    left_x: np.ndarray
    left_y: np.ndarray
    right_x: np.ndarray
    right_y: np.ndarray
    source: str


@dataclass
class TelemetryData:
    time_s: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw_rad: np.ndarray
    speed_mps: np.ndarray
    pitch_rad: np.ndarray
    roll_rad: np.ndarray
    tracking_error: np.ndarray
    vehicle_length: np.ndarray
    vehicle_width: np.ndarray
    beta_rad: Optional[np.ndarray] = None
    ref_x: Optional[np.ndarray] = None
    ref_y: Optional[np.ndarray] = None
    source_format: str = "named"
    warnings: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.time_s = np.asarray(self.time_s, dtype=float)
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        self.yaw_rad = np.asarray(self.yaw_rad, dtype=float)
        self.speed_mps = np.asarray(self.speed_mps, dtype=float)
        self.pitch_rad = np.asarray(self.pitch_rad, dtype=float)
        self.roll_rad = np.asarray(self.roll_rad, dtype=float)
        self.tracking_error = np.asarray(self.tracking_error, dtype=float)
        self.vehicle_length = np.asarray(self.vehicle_length, dtype=float)
        self.vehicle_width = np.asarray(self.vehicle_width, dtype=float)

        if self.beta_rad is None:
            self.beta_rad = np.zeros_like(self.yaw_rad, dtype=float)
        else:
            self.beta_rad = np.asarray(self.beta_rad, dtype=float)
            if self.beta_rad.shape != self.yaw_rad.shape:
                raise ValueError("beta_rad shape must match yaw_rad shape.")

        if self.ref_x is not None:
            self.ref_x = np.asarray(self.ref_x, dtype=float)
        if self.ref_y is not None:
            self.ref_y = np.asarray(self.ref_y, dtype=float)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render vehicle CSV telemetry to an MP4 animation.")
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_MP4, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="Video FPS.")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI, help="Figure DPI.")
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Fallback sample time.")
    parser.add_argument("--angles-in-degrees", action="store_true", default=DEFAULT_INPUT_ANGLES_IN_DEGREES)
    parser.add_argument("--road-mode", choices=["auto", "moose", "none"], default=DEFAULT_ROAD_MODE)
    parser.add_argument("--follow-view", action="store_true", default=DEFAULT_FOLLOW_VIEW)
    parser.add_argument("--follow-view-width", type=float, default=DEFAULT_FOLLOW_VIEW_WIDTH)
    parser.add_argument("--follow-view-height", type=float, default=DEFAULT_FOLLOW_VIEW_HEIGHT)
    parser.add_argument("--vehicle-length", type=float, default=DEFAULT_VEHICLE_LENGTH)
    parser.add_argument("--vehicle-width", type=float, default=DEFAULT_VEHICLE_WIDTH)
    parser.add_argument("--lane-width", type=float, default=DEFAULT_LANE_WIDTH)
    parser.add_argument("--scene-padding", type=float, default=DEFAULT_SCENE_PADDING)
    parser.add_argument("--tracking-error-alert", type=float, default=DEFAULT_TRACKING_ERROR_ALERT)
    return parser


def print_info(message: str) -> None:
    print(f"[INFO] {message}")


def print_warn(message: str) -> None:
    print(f"[WARN] {message}")


def has_named_header(csv_path: Path) -> bool:
    df = pd.read_csv(csv_path, nrows=0)
    normalized = [str(col).strip().lower() for col in df.columns]
    alias_pool = {alias.lower() for aliases in FIELD_ALIASES.values() for alias in aliases}
    match_count = sum(col in alias_pool for col in normalized)
    return match_count >= 3


def assign_env0_columns(df: pd.DataFrame) -> pd.DataFrame:
    num_cols = len(df.columns)
    if num_cols >= len(ENV0_COLUMNS_48):
        base = ENV0_COLUMNS_48
    elif num_cols >= len(ENV0_COLUMNS_38):
        base = ENV0_COLUMNS_38
    elif num_cols >= len(ENV0_COLUMNS_32):
        base = ENV0_COLUMNS_32
    else:
        raise ValueError(f"Headerless CSV has only {num_cols} columns.")
    extra = [f"col_{idx}" for idx in range(len(base), num_cols)]
    df.columns = base[:num_cols] + extra
    return df


def find_column(columns: Sequence[str], logical_name: str) -> Optional[str]:
    aliases = [item.lower() for item in FIELD_ALIASES[logical_name]]
    for column in columns:
        normalized = str(column).strip().lower()
        if normalized in aliases:
            return str(column)
    return None


def to_float_array(df: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(df[column], errors="raise").to_numpy(dtype=float)


def estimate_yaw_from_xy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.gradient(x)
    dy = np.gradient(y)
    if np.allclose(dx, 0.0) and np.allclose(dy, 0.0):
        raise ValueError("Cannot estimate yaw from x/y.")
    return np.arctan2(dy, dx)


def estimate_speed_from_xy(time_s: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx_dt = np.gradient(x, time_s)
    dy_dt = np.gradient(y, time_s)
    return np.hypot(dx_dt, dy_dt)


def interp_series(source_t: np.ndarray, source_values: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    return np.interp(target_t, source_t, source_values)


def interp_angle_rad(source_t: np.ndarray, source_angles_rad: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    unwrapped = np.unwrap(source_angles_rad)
    return np.interp(target_t, source_t, unwrapped)


def moose_reference_y(x: np.ndarray, road_cfg: MooseRoadConfig) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y = np.zeros_like(x)

    x1_end = road_cfg.lane_a_length
    x2_end = x1_end + road_cfg.curve_ab_length
    x3_end = x2_end + road_cfg.lane_b_length
    x4_end = x3_end + road_cfg.curve_bc_length

    mask_ab = (x > x1_end) & (x <= x2_end)
    if np.any(mask_ab):
        progress = (x[mask_ab] - x1_end) / road_cfg.curve_ab_length
        y[mask_ab] = road_cfg.lane_b_offset * (1.0 - np.cos(np.pi * progress)) / 2.0

    mask_b = (x > x2_end) & (x <= x3_end)
    y[mask_b] = road_cfg.lane_b_offset

    mask_bc = (x > x3_end) & (x <= x4_end)
    if np.any(mask_bc):
        progress = (x[mask_bc] - x3_end) / road_cfg.curve_bc_length
        y[mask_bc] = road_cfg.lane_b_offset * (1.0 + np.cos(np.pi * progress)) / 2.0

    return y


def build_lane_edges(center_x: np.ndarray, center_y: np.ndarray, lane_width: float) -> RoadGeometry:
    dx = np.gradient(center_x)
    dy = np.gradient(center_y)
    heading = np.arctan2(dy, dx)
    half_width = lane_width / 2.0

    left_x = center_x - half_width * np.sin(heading)
    left_y = center_y + half_width * np.cos(heading)
    right_x = center_x + half_width * np.sin(heading)
    right_y = center_y - half_width * np.cos(heading)

    return RoadGeometry(
        center_x=center_x, center_y=center_y, left_x=left_x, left_y=left_y, right_x=right_x, right_y=right_y, source="reference",
    )


def build_moose_road_geometry(x_max: float, road_cfg: MooseRoadConfig) -> RoadGeometry:
    center_x = np.linspace(0.0, max(x_max + 5.0, road_cfg.total_length + 3.0), 1500)
    center_y = moose_reference_y(center_x, road_cfg)
    geometry = build_lane_edges(center_x, center_y, road_cfg.lane_width)
    geometry.source = "moose"
    return geometry


def compute_signed_error_against_reference(x: np.ndarray, y: np.ndarray, ref_x: np.ndarray, ref_y: np.ndarray) -> np.ndarray:
    ref_points = np.column_stack([ref_x, ref_y])
    ref_dx = np.gradient(ref_x)
    ref_dy = np.gradient(ref_y)
    ref_heading = np.arctan2(ref_dy, ref_dx)
    ref_normals = np.column_stack([-np.sin(ref_heading), np.cos(ref_heading)])

    errors = np.zeros_like(x, dtype=float)
    same_length = len(ref_x) == len(x)

    if same_length:
        delta = np.column_stack([x - ref_x, y - ref_y])
        errors = np.sum(delta * ref_normals, axis=1)
        return errors

    for idx, (px, py) in enumerate(zip(x, y)):
        delta = ref_points - np.array([px, py], dtype=float)
        nearest_index = int(np.argmin(np.sum(delta * delta, axis=1)))
        vec = np.array([px - ref_x[nearest_index], py - ref_y[nearest_index]], dtype=float)
        errors[idx] = float(np.dot(vec, ref_normals[nearest_index]))

    return errors


def load_named_csv(
    csv_path: Path, dt_fallback: float, angles_in_degrees: bool,
    default_vehicle_length: float, default_vehicle_width: float,
    road_cfg: MooseRoadConfig, road_mode: str,
) -> TelemetryData:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    columns = list(df.columns)

    x_col = find_column(columns, "x")
    y_col = find_column(columns, "y")
    if not x_col or not y_col: raise ValueError("Missing required x/y columns.")

    t_col = find_column(columns, "t")
    yaw_col = find_column(columns, "yaw")
    beta_col = find_column(columns, "beta")
    speed_col = find_column(columns, "speed")
    pitch_col = find_column(columns, "pitch")
    roll_col = find_column(columns, "roll")
    tracking_error_col = find_column(columns, "tracking_error")
    ref_x_col = find_column(columns, "ref_x")
    ref_y_col = find_column(columns, "ref_y")
    veh_len_col = find_column(columns, "vehicle_length")
    veh_wid_col = find_column(columns, "vehicle_width")

    x = to_float_array(df, x_col)
    y = to_float_array(df, y_col)

    warnings: List[str] = []

    if t_col:
        time_s = to_float_array(df, t_col)
    else:
        time_s = np.arange(len(df), dtype=float) * dt_fallback
        warnings.append("Column 't' not found. Built time axis.")

    if yaw_col:
        yaw_raw = to_float_array(df, yaw_col)
        yaw_rad = np.deg2rad(yaw_raw) if angles_in_degrees else yaw_raw
    else:
        yaw_rad = estimate_yaw_from_xy(x, y)
        warnings.append("Column 'yaw' not found. Estimated.")

    if beta_col:
        beta_raw = to_float_array(df, beta_col)
        beta_rad = np.deg2rad(beta_raw) if angles_in_degrees else beta_raw
    else:
        beta_rad = np.zeros(len(df), dtype=float)
        warnings.append("Column 'beta' not found. Beta display will use zeros.")

    if speed_col: speed_mps = to_float_array(df, speed_col)
    else: speed_mps = estimate_speed_from_xy(time_s, x, y)

    if pitch_col:
        pitch_raw = to_float_array(df, pitch_col)
        pitch_rad = np.deg2rad(pitch_raw) if angles_in_degrees else pitch_raw
    else: pitch_rad = np.zeros(len(df), dtype=float)

    if roll_col:
        roll_raw = to_float_array(df, roll_col)
        roll_rad = np.deg2rad(roll_raw) if angles_in_degrees else roll_raw
    else: roll_rad = np.zeros(len(df), dtype=float)

    ref_x = to_float_array(df, ref_x_col) if ref_x_col else None
    ref_y = to_float_array(df, ref_y_col) if ref_y_col else None

    if tracking_error_col: tracking_error = to_float_array(df, tracking_error_col)
    elif ref_x is not None and ref_y is not None: tracking_error = compute_signed_error_against_reference(x, y, ref_x, ref_y)
    elif road_mode == "moose": tracking_error = y - moose_reference_y(x, road_cfg)
    else: tracking_error = np.zeros(len(df), dtype=float)

    vehicle_length = to_float_array(df, veh_len_col) if veh_len_col else np.full(len(df), default_vehicle_length, dtype=float)
    vehicle_width = to_float_array(df, veh_wid_col) if veh_wid_col else np.full(len(df), default_vehicle_width, dtype=float)

    return TelemetryData(
        time_s=time_s, x=x, y=y, yaw_rad=yaw_rad, beta_rad=beta_rad, speed_mps=speed_mps,
        pitch_rad=pitch_rad, roll_rad=roll_rad, tracking_error=tracking_error,
        vehicle_length=vehicle_length, vehicle_width=vehicle_width,
        ref_x=ref_x, ref_y=ref_y, source_format="named", warnings=warnings,
    )


def load_env0_csv(
    csv_path: Path, dt_fallback: float, default_vehicle_length: float,
    default_vehicle_width: float, road_cfg: MooseRoadConfig,
) -> TelemetryData:
    df = pd.read_csv(csv_path, header=None)
    df = assign_env0_columns(df)

    x = to_float_array(df, "x")
    y = to_float_array(df, "y")
    time_s = np.arange(len(df), dtype=float) * dt_fallback
    yaw_rad = to_float_array(df, "psi") if "psi" in df.columns else estimate_yaw_from_xy(x, y)
    beta_rad = to_float_array(df, "beta") if "beta" in df.columns else np.zeros(len(df), dtype=float)
    speed_mps = to_float_array(df, "v") if "v" in df.columns else estimate_speed_from_xy(time_s, x, y)
    pitch_rad = to_float_array(df, "theta") if "theta" in df.columns else np.zeros(len(df), dtype=float)
    roll_rad = to_float_array(df, "phi") if "phi" in df.columns else np.zeros(len(df), dtype=float)
    tracking_error = y - moose_reference_y(x, road_cfg)

    warnings = ["Headerless env0 CSV detected."]

    return TelemetryData(
        time_s=time_s, x=x, y=y, yaw_rad=yaw_rad, beta_rad=beta_rad, speed_mps=speed_mps,
        pitch_rad=pitch_rad, roll_rad=roll_rad, tracking_error=tracking_error,
        vehicle_length=np.full(len(df), default_vehicle_length, dtype=float),
        vehicle_width=np.full(len(df), default_vehicle_width, dtype=float),
        ref_x=None, ref_y=None, source_format="env0", warnings=warnings,
    )


def sanitize_telemetry(data: TelemetryData) -> TelemetryData:
    order = np.argsort(data.time_s)
    time_s = np.asarray(data.time_s, dtype=float)[order]
    unique_mask = np.ones_like(time_s, dtype=bool)
    unique_mask[1:] = np.diff(time_s) > 1e-12

    def dedup(array: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if array is None: return None
        return np.asarray(array, dtype=float)[order][unique_mask]

    sanitized = TelemetryData(
        time_s=time_s[unique_mask], x=dedup(data.x), y=dedup(data.y),
        yaw_rad=dedup(data.yaw_rad), beta_rad=dedup(data.beta_rad), speed_mps=dedup(data.speed_mps),
        pitch_rad=dedup(data.pitch_rad), roll_rad=dedup(data.roll_rad),
        tracking_error=dedup(data.tracking_error), vehicle_length=dedup(data.vehicle_length),
        vehicle_width=dedup(data.vehicle_width), ref_x=dedup(data.ref_x),
        ref_y=dedup(data.ref_y), source_format=data.source_format, warnings=list(data.warnings),
    )
    return sanitized


def resample_telemetry(data: TelemetryData, fps: int) -> TelemetryData:
    start_t, end_t = float(data.time_s[0]), float(data.time_s[-1])
    frame_count = max(2, int(np.ceil((end_t - start_t) * fps)) + 1)
    frame_time = np.linspace(start_t, end_t, frame_count)

    ref_x = interp_series(data.time_s, data.ref_x, frame_time) if data.ref_x is not None else None
    ref_y = interp_series(data.time_s, data.ref_y, frame_time) if data.ref_y is not None else None

    return TelemetryData(
        time_s=frame_time,
        x=interp_series(data.time_s, data.x, frame_time),
        y=interp_series(data.time_s, data.y, frame_time),
        yaw_rad=interp_angle_rad(data.time_s, data.yaw_rad, frame_time),
        beta_rad=interp_angle_rad(data.time_s, data.beta_rad, frame_time),
        speed_mps=interp_series(data.time_s, data.speed_mps, frame_time),
        pitch_rad=interp_angle_rad(data.time_s, data.pitch_rad, frame_time),
        roll_rad=interp_angle_rad(data.time_s, data.roll_rad, frame_time),
        tracking_error=interp_series(data.time_s, data.tracking_error, frame_time),
        vehicle_length=interp_series(data.time_s, data.vehicle_length, frame_time),
        vehicle_width=interp_series(data.time_s, data.vehicle_width, frame_time),
        ref_x=ref_x, ref_y=ref_y, source_format=data.source_format, warnings=list(data.warnings),
    )


def load_and_prepare_data(
    csv_path: Path, dt_fallback: float, angles_in_degrees: bool,
    default_vehicle_length: float, default_vehicle_width: float,
    road_cfg: MooseRoadConfig, fps: int, road_mode: str,
) -> TelemetryData:
    if has_named_header(csv_path):
        raw = load_named_csv(
            csv_path, dt_fallback, angles_in_degrees, default_vehicle_length,
            default_vehicle_width, road_cfg, road_mode
        )
    else:
        raw = load_env0_csv(csv_path, dt_fallback, default_vehicle_length, default_vehicle_width, road_cfg)

    sanitized = sanitize_telemetry(raw)
    resampled = resample_telemetry(sanitized, fps=fps)
    return resampled


def build_road_geometry(data: TelemetryData, road_mode: str, road_cfg: MooseRoadConfig) -> Optional[RoadGeometry]:
    if road_mode == "none": return None
    if data.ref_x is not None and data.ref_y is not None:
        geometry = build_lane_edges(data.ref_x, data.ref_y, road_cfg.lane_width)
        geometry.source = "csv_reference"
        return geometry
    if road_mode == "moose" or data.source_format == "env0":
        return build_moose_road_geometry(float(np.max(data.x)), road_cfg)
    return None


# ------------------------------------------------------------
# GEOMETRY OPTIMIZATION: Advanced vehicle shape computation
# ------------------------------------------------------------
def get_vehicle_transformed_geometry(
    center_x: float,
    center_y: float,
    yaw_rad: float,
    length: float,
    width: float,
    render_scale: float = 1.0,
) -> Dict[str, Any]:
    """Computes global coordinates of vehicle components (body, cabin, wheels, lights)."""
    half_l, half_w = length / 2.0, width / 2.0
    
    # Body (6 pts, rounded rear)
    body_local = np.array([
        [half_l - 0.2, half_w - 0.1],  # Front Right
        [half_l - 0.2, -half_w + 0.1], # Front Left
        [-half_l + 0.5, -half_w + 0.1], # Rear Left - corner point
        [-half_l, -half_w + 0.5],     # Rear Left - rounded
        [-half_l, half_w - 0.5],      # Rear Right - rounded
        [-half_l + 0.5, half_w - 0.1],  # Rear Right - corner point
    ], dtype=float)
    
    # Cabin: set inside body
    cabin_local = np.array([
        [half_l - 1.0, half_w - 0.2],
        [half_l - 1.0, -half_w + 0.2],
        [-half_l + 1.5, -half_w + 0.2],
        [-half_l + 1.5, half_w - 0.2]
    ], dtype=float)
    
    # Wheels centers
    wheel_fr_local = np.array([half_l - 1.2, half_w], dtype=float)
    wheel_fl_local = np.array([half_l - 1.2, -half_w], dtype=float)
    wheel_rr_local = np.array([-half_l + 1.2, half_w], dtype=float)
    wheel_rl_local = np.array([-half_l + 1.2, -half_w], dtype=float)
    
    # Lights (as polygons)
    hlight_r_local = np.array([[half_l - 0.1, half_w - 0.2], [half_l, half_w - 0.3], [half_l, half_w - 0.6]])
    hlight_l_local = np.array([[half_l - 0.1, -half_w + 0.2], [half_l, -half_w + 0.3], [half_l, -half_w + 0.6]])
    tlight_r_local = np.array([[-half_l, half_w - 0.2], [-half_l, half_w - 0.4], [-half_l - 0.1, half_w - 0.4], [-half_l - 0.1, half_w - 0.2]])
    tlight_l_local = np.array([[-half_l, -half_w + 0.2], [-half_l, -half_w + 0.4], [-half_l - 0.1, -half_w + 0.4], [-half_l - 0.1, -half_w + 0.2]])

    if render_scale != 1.0:
        body_local *= render_scale
        cabin_local *= render_scale
        wheel_fr_local *= render_scale
        wheel_fl_local *= render_scale
        wheel_rr_local *= render_scale
        wheel_rl_local *= render_scale
        hlight_r_local *= render_scale
        hlight_l_local *= render_scale
        tlight_r_local *= render_scale
        tlight_l_local *= render_scale

    rotation = np.array([[np.cos(yaw_rad), -np.sin(yaw_rad)], [np.sin(yaw_rad), np.cos(yaw_rad)]], dtype=float)
    center_global = np.array([center_x, center_y], dtype=float)

    def transform_polygon(local: np.ndarray) -> np.ndarray:
        return local @ rotation.T + center_global

    def transform_point(local: np.ndarray) -> np.ndarray:
        return local @ rotation.T + center_global

    return {
        "body": transform_polygon(body_local),
        "cabin": transform_polygon(cabin_local),
        "wheel_fr": transform_point(wheel_fr_local),
        "wheel_fl": transform_point(wheel_fl_local),
        "wheel_rr": transform_point(wheel_rr_local),
        "wheel_rl": transform_point(wheel_rl_local),
        "hlight_r": transform_polygon(hlight_r_local),
        "hlight_l": transform_polygon(hlight_l_local),
        "tlight_r": transform_polygon(tlight_r_local),
        "tlight_l": transform_polygon(tlight_l_local),
    }


def compute_scene_limits(data: TelemetryData, road_geometry: Optional[RoadGeometry], padding: float) -> Tuple[float, float, float, float]:
    xs, ys = [data.x], [data.y]
    if data.ref_x is not None and data.ref_y is not None:
        xs.append(data.ref_x); ys.append(data.ref_y)
    if road_geometry is not None:
        xs.extend([road_geometry.left_x, road_geometry.right_x])
        ys.extend([road_geometry.left_y, road_geometry.right_y])
    x_all, y_all = np.concatenate(xs), np.concatenate(ys)
    return float(np.min(x_all) - padding), float(np.max(x_all) + padding), float(np.min(y_all) - padding), float(np.max(y_all) + padding)


def zoom_scene_limits(
    limits: Tuple[float, float, float, float],
    zoom_factor: float,
) -> Tuple[float, float, float, float]:
    if zoom_factor <= 0.0:
        raise ValueError("zoom_factor must be positive.")

    xmin, xmax, ymin, ymax = limits
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    half_width = (xmax - xmin) / (2.0 * zoom_factor)
    half_height = (ymax - ymin) / (2.0 * zoom_factor)
    return (
        center_x - half_width,
        center_x + half_width,
        center_y - half_height,
        center_y + half_height,
    )


def ensure_ffmpeg_available() -> None:
    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError("ffmpeg is not available. Please install ffmpeg.")


def render_animation(
    data: TelemetryData, road_geometry: Optional[RoadGeometry], output_path: Path,
    fps: int, dpi: int, follow_view: bool, follow_view_width: float,
    follow_view_height: float, scene_padding: float, tracking_error_alert: float,
    curve_y_lims: Optional[Dict[str, Tuple[float, float]]] = None,
) -> None:
    # ------------------------------------------------------------
    # AESTHETIC TWEAKS: 科研级别排版配置
    # ------------------------------------------------------------
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.linewidth": 1.2,
        "grid.alpha": 0.4,
        "grid.color": "#a0a0a0",
        "grid.linestyle": "--",
    })

    yaw_deg = np.rad2deg(data.yaw_rad)
    beta_deg = np.rad2deg(data.beta_rad)
    pitch_deg = np.rad2deg(data.pitch_rad)
    roll_deg = np.rad2deg(data.roll_rad)

    fig = plt.figure(figsize=(12.8, 7.2), dpi=dpi, constrained_layout=False)
    gs = GridSpec(
        2,
        4,
        figure=fig,
        height_ratios=[2.3, 1.6],
        hspace=0.38,
        wspace=0.30,
    )

    ax_scene = fig.add_subplot(gs[0, :])
    curve_axes = [fig.add_subplot(gs[1, idx]) for idx in range(4)]

    fig.patch.set_facecolor("white")
    ax_scene.set_facecolor("#fbfbfc")
    ax_scene.set_aspect("equal", adjustable="box")
    ax_scene.grid(True)
    ax_scene.set_xlabel("X-coordinate [m]", weight="bold")
    ax_scene.set_ylabel("Y-coordinate [m]", weight="bold")
    ax_scene.set_title("Vehicle Trajectory & Top View", pad=12)

    scene_limits = zoom_scene_limits(
        compute_scene_limits(data, road_geometry, scene_padding),
        DEFAULT_SCENE_CONTENT_ZOOM,
    )
    if not follow_view:
        ax_scene.set_xlim(scene_limits[0], scene_limits[1])
        ax_scene.set_ylim(scene_limits[2], scene_limits[3])

    if road_geometry is not None:
        road_x = np.concatenate([road_geometry.left_x, road_geometry.right_x[::-1]])
        road_y = np.concatenate([road_geometry.left_y, road_geometry.right_y[::-1]])
        ax_scene.fill(road_x, road_y, color="#eef1f4", alpha=1.0, zorder=0)
        ax_scene.plot(road_geometry.left_x, road_geometry.left_y, color="#5f6b7a", linewidth=1.5, zorder=1)
        ax_scene.plot(road_geometry.right_x, road_geometry.right_y, color="#5f6b7a", linewidth=1.5, zorder=1)
        ax_scene.plot(
            road_geometry.center_x, road_geometry.center_y,
            color="#9aa4af", linestyle="--", linewidth=1.2, zorder=1, label="Reference Centerline"
        )

    if data.ref_x is not None and data.ref_y is not None:
        ax_scene.plot(
            data.ref_x, data.ref_y, color="#b0b0b0", linestyle="--", linewidth=1.4, zorder=2, label="Ref Trajectory"
        )

    # 画出浅色的完整轨迹底纹
    ax_scene.plot(data.x, data.y, color="#cbd5e1", linewidth=1.5, alpha=0.6, zorder=2, label="Full Trajectory")

    history_line, = ax_scene.plot([], [], color="#1d4ed8", linewidth=2.5, zorder=4, label="Driven Path")
    # current_point 改为更小的标记 "*", 用于指示车辆中心
    current_point, = ax_scene.plot([], [], marker="*", markersize=8, color="#1d4ed8", zorder=10)
    
    # 调整图例
    ax_scene.legend(loc="upper right", framealpha=0.9, edgecolor="#d0d7de")

    # ------------------------------------------------------------
    # VEHICLE SHAPE OPTIMIZATION: Advanced multi-patch composition
    # ------------------------------------------------------------
    # Body (6 pts, rounded rear)
    body_patch = Polygon(np.zeros((6, 2), dtype=float), closed=True, facecolor="#0f766e", edgecolor="#0f172a", linewidth=1.5, zorder=7)
    
    # Cabin
    cabin_patch = Polygon(np.zeros((4, 2), dtype=float), closed=True, facecolor="#475569", edgecolor="#1f2937", linewidth=1.0, alpha=0.9, zorder=8)
    
    # Wheels (r=0.38)
    wheel_fr = Circle((0, 0), radius=0.38 * DEFAULT_VEHICLE_RENDER_SCALE, facecolor="#1f2937", edgecolor="#0f172a", linewidth=1.0, zorder=6)
    wheel_fl = Circle((0, 0), radius=0.38 * DEFAULT_VEHICLE_RENDER_SCALE, facecolor="#1f2937", edgecolor="#0f172a", linewidth=1.0, zorder=6)
    wheel_rr = Circle((0, 0), radius=0.38 * DEFAULT_VEHICLE_RENDER_SCALE, facecolor="#1f2937", edgecolor="#0f172a", linewidth=1.0, zorder=6)
    wheel_rl = Circle((0, 0), radius=0.38 * DEFAULT_VEHICLE_RENDER_SCALE, facecolor="#1f2937", edgecolor="#0f172a", linewidth=1.0, zorder=6)
    
    # Lights
    hlight_r = Polygon(np.zeros((3, 2), dtype=float), closed=True, facecolor="#fbbf24", edgecolor="none", zorder=9)
    hlight_l = Polygon(np.zeros((3, 2), dtype=float), closed=True, facecolor="#fbbf24", edgecolor="none", zorder=9)
    tlight_r = Polygon(np.zeros((4, 2), dtype=float), closed=True, facecolor="#b91c1c", edgecolor="none", zorder=9)
    tlight_l = Polygon(np.zeros((4, 2), dtype=float), closed=True, facecolor="#b91c1c", edgecolor="none", zorder=9)

    # Add all patches to ax_scene
    ax_scene.add_patch(body_patch)
    ax_scene.add_patch(cabin_patch)
    ax_scene.add_patch(wheel_fr)
    ax_scene.add_patch(wheel_fl)
    ax_scene.add_patch(wheel_rr)
    ax_scene.add_patch(wheel_rl)
    ax_scene.add_patch(hlight_r)
    ax_scene.add_patch(hlight_l)
    ax_scene.add_patch(tlight_r)
    ax_scene.add_patch(tlight_l)
    
    # Heading arrow 已经被优化后的形状取代，无需再绘制
    
    hud_text = ax_scene.text(
        0.015, 0.965, "", transform=ax_scene.transAxes, va="top", ha="left",
        fontsize=10, family="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d0d7de", alpha=0.9)
    )

    curve_y_lims = curve_y_lims or {}
    curve_specs = [
        ("Pitch", pitch_deg, "deg", "#d97706", curve_y_lims.get("Pitch", DEFAULT_ATTITUDE_Y_LIM)),
        ("Roll", roll_deg, "deg", "#059669", curve_y_lims.get("Roll", DEFAULT_ATTITUDE_Y_LIM)),
        ("Beta", beta_deg, "deg", "#2563eb", curve_y_lims.get("Beta", DEFAULT_ATTITUDE_Y_LIM)),
        (
            "Tracking Error",
            data.tracking_error,
            "m",
            "#dc2626",
            curve_y_lims.get("Tracking Error", DEFAULT_TRACKING_ERROR_Y_LIM),
        ),
    ]
    
    curve_artists = []

    for ax, (title, values, unit, color, y_lim) in zip(curve_axes, curve_specs):
        ax.set_facecolor("white")
        
        # 移除上方和右方多余的脊柱线
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # 改为渐进式绘制线条 (初始化为空)
        prog_line, = ax.plot([], [], color=color, linewidth=2.0)
        
        vline = ax.axvline(data.time_s[0], color="#111827", linestyle=":", linewidth=1.2, alpha=0.6)
        marker, = ax.plot(
            [data.time_s[0]], [values[0]], marker="o", markersize=6, color=color, markeredgecolor="white", markeredgewidth=1.0
        )
        
        ax.set_xlim(float(data.time_s[0]), float(data.time_s[-1]))
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.grid(True)
        ax.set_xlabel("Time [s]", weight="bold")
        ax.set_ylabel(f"{title} [{unit}]", weight="bold")
        ax.set_title(f"{title}: {values[0]:.2f} {unit}", pad=8)
        
        # 存储线对象用于后续更新
        curve_artists.append((ax, prog_line, vline, marker, title, values, unit))

    def update(frame_idx: int):
        x_cur = float(data.x[frame_idx])
        y_cur = float(data.y[frame_idx])
        yaw_cur = float(data.yaw_rad[frame_idx])
        beta_cur = float(beta_deg[frame_idx])
        speed_cur = float(data.speed_mps[frame_idx])
        error_cur = float(data.tracking_error[frame_idx])
        length_cur = float(data.vehicle_length[frame_idx])
        width_cur = float(data.vehicle_width[frame_idx])

        # Follow view logic
        if follow_view:
            zoomed_follow_width = follow_view_width / DEFAULT_SCENE_CONTENT_ZOOM
            zoomed_follow_height = follow_view_height / DEFAULT_SCENE_CONTENT_ZOOM
            ax_scene.set_xlim(x_cur - zoomed_follow_width / 2.0, x_cur + zoomed_follow_width / 2.0)
            ax_scene.set_ylim(y_cur - zoomed_follow_height / 2.0, y_cur + zoomed_follow_height / 2.0)

        history_line.set_data(data.x[: frame_idx + 1], data.y[: frame_idx + 1])
        current_point.set_data([x_cur], [y_cur])

        # ------------------------------------------------------------
        # UPDATE OPTIMIZED VEHICLE SHAPE: Calculate & set component coordinates
        # ------------------------------------------------------------
        geo = get_vehicle_transformed_geometry(
            x_cur,
            y_cur,
            yaw_cur,
            length_cur,
            width_cur,
            render_scale=DEFAULT_VEHICLE_RENDER_SCALE,
        )
        body_patch.set_xy(geo["body"])
        cabin_patch.set_xy(geo["cabin"])
        wheel_fr.set_center(geo["wheel_fr"])
        wheel_fl.set_center(geo["wheel_fl"])
        wheel_rr.set_center(geo["wheel_rr"])
        wheel_rl.set_center(geo["wheel_rl"])
        hlight_r.set_xy(geo["hlight_r"])
        hlight_l.set_xy(geo["hlight_l"])
        tlight_r.set_xy(geo["tlight_r"])
        tlight_l.set_xy(geo["tlight_l"])

        # Update body color based on tracking error
        body_color = "#dc2626" if abs(error_cur) >= tracking_error_alert else "#0f766e"
        body_patch.set_facecolor(body_color)
        current_point.set_color(body_color) # '*' marker changes color too

        hud_text.set_text(
            "\n".join([
                f"t              = {data.time_s[frame_idx]:>6.2f} s",
                f"speed          = {speed_cur:>6.2f} m/s",
                f"beta           = {beta_cur:>6.2f} deg",
                f"tracking_error = {error_cur:>6.3f} m",
            ])
        )

        artists = [
            history_line, current_point, 
            body_patch, cabin_patch, wheel_fr, wheel_fl, wheel_rr, wheel_rl,
            hlight_r, hlight_l, tlight_r, tlight_l,
            hud_text
        ]
        
        # 动态更新子图
        for ax, prog_line, vline, marker, title, values, unit in curve_artists:
            x_line = float(data.time_s[frame_idx])
            y_line = float(values[frame_idx])
            
            # 渐进式绘制核心：更新从起点到当前帧的数据
            prog_line.set_data(data.time_s[:frame_idx + 1], values[:frame_idx + 1])
            
            vline.set_xdata([x_line, x_line])
            marker.set_data([x_line], [y_line])
            ax.set_title(f"{title}: {y_line:.2f} {unit}")
            artists.extend([prog_line, vline, marker])

        return artists

    anim = FuncAnimation(fig, update, frames=len(data.time_s), interval=1000.0 / fps, blit=False, repeat=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=4000, codec="libx264")  # 提高 bitrate 使得视频更加清晰
    anim.save(str(output_path), writer=writer)
    plt.close(fig)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    road_cfg = MooseRoadConfig(lane_width=args.lane_width)

    print_info(f"Input CSV: {input_path}")
    print_info(f"Output MP4: {output_path}")

    data = load_and_prepare_data(
        csv_path=input_path, dt_fallback=args.dt, angles_in_degrees=args.angles_in_degrees,
        default_vehicle_length=args.vehicle_length, default_vehicle_width=args.vehicle_width,
        road_cfg=road_cfg, fps=args.fps, road_mode=args.road_mode,
    )

    road_geometry = build_road_geometry(data=data, road_mode=args.road_mode, road_cfg=road_cfg)

    ensure_ffmpeg_available()
    print_info("ffmpeg is available. Start rendering MP4 ...")

    render_animation(
        data=data, road_geometry=road_geometry, output_path=output_path,
        fps=args.fps, dpi=args.dpi, follow_view=args.follow_view,
        follow_view_width=args.follow_view_width, follow_view_height=args.follow_view_height,
        scene_padding=args.scene_padding, tracking_error_alert=args.tracking_error_alert,
    )

    print_info(f"Video saved successfully: {output_path}")


if __name__ == "__main__":
    main()
