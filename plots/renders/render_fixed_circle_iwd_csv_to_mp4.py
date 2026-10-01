from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import render_vehicle_csv_to_mp4 as base


DEFAULT_INPUT_CSV = "env0.csv"
DEFAULT_OUTPUT_MP4 = "fixed_circle_iwd_animation.mp4"
DEFAULT_ROAD_MODE = "auto"
FIXED_CIRCLE_IWD_CURVE_Y_LIMS = {"Beta": (0.0, -30.0)}


@dataclass
class FixedCircleRoadConfig:
    center_x: float = 0.0
    center_y: float = 12.0
    radius: float = 12.0
    lane_width: float = base.DEFAULT_LANE_WIDTH
    num_points: int = 1440


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render fixed_circle_iwd CSV telemetry to an MP4 animation."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_CSV, help="Input CSV path.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_MP4, help="Output MP4 path.")
    parser.add_argument("--fps", type=int, default=base.DEFAULT_FPS, help="Video FPS.")
    parser.add_argument("--dpi", type=int, default=base.DEFAULT_DPI, help="Figure DPI.")
    parser.add_argument("--dt", type=float, default=base.DEFAULT_DT, help="Fallback sample time.")
    parser.add_argument(
        "--angles-in-degrees",
        action="store_true",
        default=base.DEFAULT_INPUT_ANGLES_IN_DEGREES,
        help="Treat yaw/pitch/roll as degrees in named CSV files.",
    )
    parser.add_argument(
        "--road-mode",
        choices=["auto", "fixed_circle", "none"],
        default=DEFAULT_ROAD_MODE,
        help="Road drawing mode.",
    )
    parser.add_argument("--follow-view", action="store_true", default=base.DEFAULT_FOLLOW_VIEW)
    parser.add_argument("--follow-view-width", type=float, default=base.DEFAULT_FOLLOW_VIEW_WIDTH)
    parser.add_argument("--follow-view-height", type=float, default=base.DEFAULT_FOLLOW_VIEW_HEIGHT)
    parser.add_argument("--vehicle-length", type=float, default=base.DEFAULT_VEHICLE_LENGTH)
    parser.add_argument("--vehicle-width", type=float, default=base.DEFAULT_VEHICLE_WIDTH)
    parser.add_argument("--lane-width", type=float, default=base.DEFAULT_LANE_WIDTH)
    parser.add_argument("--scene-padding", type=float, default=base.DEFAULT_SCENE_PADDING)
    parser.add_argument(
        "--tracking-error-alert",
        type=float,
        default=base.DEFAULT_TRACKING_ERROR_ALERT,
    )
    parser.add_argument("--circle-radius", type=float, default=12.0, help="Reference circle radius.")
    parser.add_argument("--circle-center-x", type=float, default=0.0, help="Reference circle center x.")
    parser.add_argument("--circle-center-y", type=float, default=12.0, help="Reference circle center y.")
    return parser


def fixed_circle_tracking_error(
    x: np.ndarray,
    y: np.ndarray,
    road_cfg: FixedCircleRoadConfig,
) -> np.ndarray:
    radial_distance = np.hypot(x - road_cfg.center_x, y - road_cfg.center_y)
    return radial_distance - road_cfg.radius


def build_fixed_circle_reference_path(
    road_cfg: FixedCircleRoadConfig,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(0.0, 2.0 * np.pi, road_cfg.num_points, endpoint=True)
    ref_x = road_cfg.center_x + road_cfg.radius * np.cos(theta)
    ref_y = road_cfg.center_y + road_cfg.radius * np.sin(theta)
    return ref_x, ref_y


def build_fixed_circle_road_geometry(road_cfg: FixedCircleRoadConfig) -> base.RoadGeometry:
    theta = np.linspace(0.0, 2.0 * np.pi, road_cfg.num_points, endpoint=True)
    center_x = road_cfg.center_x + road_cfg.radius * np.cos(theta)
    center_y = road_cfg.center_y + road_cfg.radius * np.sin(theta)
    outer_r = road_cfg.radius + road_cfg.lane_width / 2.0
    inner_r = max(road_cfg.radius - road_cfg.lane_width / 2.0, 0.1)
    left_x = road_cfg.center_x + outer_r * np.cos(theta)
    left_y = road_cfg.center_y + outer_r * np.sin(theta)
    right_x = road_cfg.center_x + inner_r * np.cos(theta)
    right_y = road_cfg.center_y + inner_r * np.sin(theta)
    return base.RoadGeometry(
        center_x=center_x,
        center_y=center_y,
        left_x=left_x,
        left_y=left_y,
        right_x=right_x,
        right_y=right_y,
        source="fixed_circle",
    )


def load_named_csv(
    csv_path: Path,
    dt_fallback: float,
    angles_in_degrees: bool,
    default_vehicle_length: float,
    default_vehicle_width: float,
    road_cfg: FixedCircleRoadConfig,
) -> base.TelemetryData:
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]
    columns = list(df.columns)

    x_col = base.find_column(columns, "x")
    y_col = base.find_column(columns, "y")
    if not x_col or not y_col:
        raise ValueError(f"Missing required x/y columns. Current columns: {columns}")

    x = base.to_float_array(df, x_col)
    y = base.to_float_array(df, y_col)
    warnings: list[str] = []

    t_col = base.find_column(columns, "t")
    if t_col:
        time_s = base.to_float_array(df, t_col)
    else:
        time_s = np.arange(len(df), dtype=float) * dt_fallback
        warnings.append(f"Column 't' not found. Built time axis from dt={dt_fallback:.6f}s.")

    yaw_col = base.find_column(columns, "yaw")
    beta_col = base.find_column(columns, "beta")
    if yaw_col:
        yaw_raw = base.to_float_array(df, yaw_col)
        yaw_rad = np.deg2rad(yaw_raw) if angles_in_degrees else yaw_raw
    else:
        yaw_rad = base.estimate_yaw_from_xy(x, y)
        warnings.append("Column 'yaw' not found. Estimated yaw from x/y gradients.")

    if beta_col:
        beta_raw = base.to_float_array(df, beta_col)
        beta_rad = np.deg2rad(beta_raw) if angles_in_degrees else beta_raw
    else:
        beta_rad = np.zeros(len(df), dtype=float)
        warnings.append("Column 'beta' not found. Beta display will use zeros.")

    speed_col = base.find_column(columns, "speed")
    if speed_col:
        speed_mps = base.to_float_array(df, speed_col)
    else:
        speed_mps = base.estimate_speed_from_xy(time_s, x, y)
        warnings.append("Column 'speed' not found. Estimated speed from x/y and time.")

    pitch_col = base.find_column(columns, "pitch")
    roll_col = base.find_column(columns, "roll")
    if not pitch_col or not roll_col:
        raise ValueError("Named CSV must provide pitch/roll columns for this renderer.")
    pitch_raw = base.to_float_array(df, pitch_col)
    roll_raw = base.to_float_array(df, roll_col)
    pitch_rad = np.deg2rad(pitch_raw) if angles_in_degrees else pitch_raw
    roll_rad = np.deg2rad(roll_raw) if angles_in_degrees else roll_raw

    tracking_error_col = base.find_column(columns, "tracking_error")
    if tracking_error_col:
        tracking_error = base.to_float_array(df, tracking_error_col)
    else:
        tracking_error = fixed_circle_tracking_error(x, y, road_cfg)
        warnings.append("Column 'tracking_error' not found. Computed as radial circle error.")

    veh_len_col = base.find_column(columns, "vehicle_length")
    veh_wid_col = base.find_column(columns, "vehicle_width")
    vehicle_length = (
        base.to_float_array(df, veh_len_col)
        if veh_len_col
        else np.full(len(df), default_vehicle_length, dtype=float)
    )
    vehicle_width = (
        base.to_float_array(df, veh_wid_col)
        if veh_wid_col
        else np.full(len(df), default_vehicle_width, dtype=float)
    )

    return base.TelemetryData(
        time_s=time_s,
        x=x,
        y=y,
        yaw_rad=yaw_rad,
        beta_rad=beta_rad,
        speed_mps=speed_mps,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        tracking_error=tracking_error,
        vehicle_length=vehicle_length,
        vehicle_width=vehicle_width,
        ref_x=None,
        ref_y=None,
        source_format="fixed_circle_named",
        warnings=warnings,
    )


def load_env0_csv(
    csv_path: Path,
    dt_fallback: float,
    default_vehicle_length: float,
    default_vehicle_width: float,
    road_cfg: FixedCircleRoadConfig,
) -> base.TelemetryData:
    df = pd.read_csv(csv_path, header=None)
    df = base.assign_env0_columns(df)

    x = base.to_float_array(df, "x")
    y = base.to_float_array(df, "y")
    time_s = np.arange(len(df), dtype=float) * dt_fallback
    yaw_rad = base.to_float_array(df, "psi") if "psi" in df.columns else base.estimate_yaw_from_xy(x, y)
    beta_rad = base.to_float_array(df, "beta") if "beta" in df.columns else np.zeros(len(df), dtype=float)
    speed_mps = base.to_float_array(df, "v") if "v" in df.columns else base.estimate_speed_from_xy(time_s, x, y)
    pitch_rad = base.to_float_array(df, "theta") if "theta" in df.columns else np.zeros(len(df), dtype=float)
    roll_rad = base.to_float_array(df, "phi") if "phi" in df.columns else np.zeros(len(df), dtype=float)
    tracking_error = fixed_circle_tracking_error(x, y, road_cfg)

    return base.TelemetryData(
        time_s=time_s,
        x=x,
        y=y,
        yaw_rad=yaw_rad,
        beta_rad=beta_rad,
        speed_mps=speed_mps,
        pitch_rad=pitch_rad,
        roll_rad=roll_rad,
        tracking_error=tracking_error,
        vehicle_length=np.full(len(df), default_vehicle_length, dtype=float),
        vehicle_width=np.full(len(df), default_vehicle_width, dtype=float),
        ref_x=None,
        ref_y=None,
        source_format="fixed_circle_env0",
        warnings=[
            f"Detected headerless env0/fixed_circle CSV with {len(df.columns)} columns.",
            f"Built time axis from dt={dt_fallback:.6f}s because env0 has no 't' column.",
            "Computed tracking_error as radial deviation from the reference circle.",
        ],
    )


def load_and_prepare_data(
    csv_path: Path,
    dt_fallback: float,
    angles_in_degrees: bool,
    default_vehicle_length: float,
    default_vehicle_width: float,
    road_cfg: FixedCircleRoadConfig,
    fps: int,
) -> base.TelemetryData:
    if base.has_named_header(csv_path):
        raw = load_named_csv(
            csv_path,
            dt_fallback,
            angles_in_degrees,
            default_vehicle_length,
            default_vehicle_width,
            road_cfg,
        )
    else:
        raw = load_env0_csv(
            csv_path,
            dt_fallback,
            default_vehicle_length,
            default_vehicle_width,
            road_cfg,
        )

    for item in raw.warnings:
        base.print_warn(item)

    sanitized = base.sanitize_telemetry(raw)
    resampled = base.resample_telemetry(sanitized, fps=fps)
    ref_x, ref_y = build_fixed_circle_reference_path(road_cfg)
    resampled.ref_x = ref_x
    resampled.ref_y = ref_y
    base.print_info(f"Loaded {len(sanitized.time_s)} raw samples.")
    base.print_info(f"Resampled to {len(resampled.time_s)} video frames at {fps} FPS.")
    return resampled


def build_road_geometry(
    road_mode: str,
    road_cfg: FixedCircleRoadConfig,
) -> Optional[base.RoadGeometry]:
    if road_mode == "none":
        return None
    return build_fixed_circle_road_geometry(road_cfg)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    road_cfg = FixedCircleRoadConfig(
        center_x=args.circle_center_x,
        center_y=args.circle_center_y,
        radius=args.circle_radius,
        lane_width=args.lane_width,
    )
    base.print_info(f"Input CSV: {input_path}")
    base.print_info(f"Output MP4: {output_path}")

    data = load_and_prepare_data(
        csv_path=input_path,
        dt_fallback=args.dt,
        angles_in_degrees=args.angles_in_degrees,
        default_vehicle_length=args.vehicle_length,
        default_vehicle_width=args.vehicle_width,
        road_cfg=road_cfg,
        fps=args.fps,
    )
    road_geometry = build_road_geometry(args.road_mode, road_cfg)

    base.ensure_ffmpeg_available()
    base.print_info("ffmpeg is available. Start rendering MP4 ...")
    base.render_animation(
        data=data,
        road_geometry=road_geometry,
        output_path=output_path,
        fps=args.fps,
        dpi=args.dpi,
        follow_view=args.follow_view,
        follow_view_width=args.follow_view_width,
        follow_view_height=args.follow_view_height,
        scene_padding=args.scene_padding,
        tracking_error_alert=args.tracking_error_alert,
        curve_y_lims=FIXED_CIRCLE_IWD_CURVE_Y_LIMS,
    )
    base.print_info(f"Video saved successfully: {output_path}")


if __name__ == "__main__":
    main()
