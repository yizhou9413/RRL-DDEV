from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Wedge
from PIL import Image

from plot_moose_test_csv_to_png import (
    ROAD_FILL_COLOR,
    Telemetry,
    add_path_arrow,
    configure_style,
    draw_vehicle,
    read_telemetry,
    resolve_path,
    transparent_crop_from_image,
)


DEFAULT_VEHICLE_ANGLES_DEG = [-90.0, -20.0, 50.0, 120.0, 190.0]
DEFAULT_TIME1_VEHICLE_ANGLE_DEG = 50.0
DEFAULT_CIRCLE_RADIUS = 12.0
DEFAULT_CIRCLE_CENTER_X = 0.0
DEFAULT_CIRCLE_CENTER_Y = 12.0
DEFAULT_LANE_WIDTH = 3.0
DEFAULT_SCENE_PADDING = 1.5
DEFAULT_ROAD_WIDTH_SCALE = 1.0
DEFAULT_VEHICLE_LENGTH = 4.8
DEFAULT_VEHICLE_SCALE = 0.65
DEFAULT_RADIUS_LABEL_ANGLE_DEG = 225.0
DEFAULT_RADIUS_LABEL_FONT_SIZE = 22.0
DEFAULT_SNAPSHOT_INTERVAL_SECONDS = 0.75
DEFAULT_SNAPSHOT_STEP_INTERVAL = 75
DEFAULT_CENTER_MARKER_RADIUS = 0.32
DEFAULT_DT = 0.01
DEFAULT_TIME_MIN = 0.0
DEFAULT_TIME_MAX = 10.0
DEFAULT_VEHICLE_TIME_MAX = 7.0


@dataclass
class FixedCircleConfig:
    center_x: float = DEFAULT_CIRCLE_CENTER_X
    center_y: float = DEFAULT_CIRCLE_CENTER_Y
    radius: float = DEFAULT_CIRCLE_RADIUS
    lane_width: float = DEFAULT_LANE_WIDTH
    num_points: int = 1440


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a fixed_circle_iwd CSV rollout to a PNG-style static visualization."
    )
    parser.add_argument("input_csv", type=str, help="Input rollout CSV.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <input_stem>_fixed_circle_iwd.png.",
    )
    parser.add_argument(
        "--vehicle-angle",
        type=float,
        nargs="+",
        default=DEFAULT_VEHICLE_ANGLES_DEG,
        help="Polar angles in degrees where vehicle snapshots are drawn. 0 is +x, 90 is +y.",
    )
    parser.add_argument(
        "--fixed-circle-source",
        type=str,
        default="envs/fixed_circle_iwd.py",
        help="Path to fixed_circle_iwd.py used to read the default radius.",
    )
    parser.add_argument(
        "--time1-image",
        type=str,
        default="time-1.png",
        help="Vehicle image used for the highlighted snapshot.",
    )
    parser.add_argument(
        "--times-image",
        type=str,
        default="times.png",
        help="Vehicle image used for non-highlighted snapshots.",
    )
    parser.add_argument(
        "--time1-vehicle-angle",
        type=float,
        default=DEFAULT_TIME1_VEHICLE_ANGLE_DEG,
        help="The vehicle-angle value that should use --time1-image.",
    )
    parser.add_argument(
        "--lane-width",
        type=float,
        default=None,
        help="Visual lane width before --road-width-scale. Defaults to 3.0 m.",
    )
    parser.add_argument(
        "--road-width-scale",
        type=float,
        default=DEFAULT_ROAD_WIDTH_SCALE,
        help="Scale applied to the visual lane width.",
    )
    parser.add_argument("--circle-radius", type=float, default=None, help="Reference circle radius.")
    parser.add_argument("--circle-center-x", type=float, default=None, help="Reference circle center x.")
    parser.add_argument("--circle-center-y", type=float, default=None, help="Reference circle center y.")
    parser.add_argument("--x-min", type=float, default=None, help="Left x limit.")
    parser.add_argument("--x-max", type=float, default=None, help="Right x limit.")
    parser.add_argument("--y-min", type=float, default=None, help="Bottom y limit.")
    parser.add_argument("--y-max", type=float, default=None, help="Top y limit.")
    parser.add_argument(
        "--scene-padding",
        type=float,
        default=DEFAULT_SCENE_PADDING,
        help="Padding around the circular road when axis limits are automatic.",
    )
    parser.add_argument(
        "--vehicle-scale",
        type=float,
        default=DEFAULT_VEHICLE_SCALE,
        help="Scale applied to the vehicle image/vector size.",
    )
    parser.add_argument(
        "--vehicle-placement",
        choices=["steps", "angles"],
        default="steps",
        help="Use CSV time-step snapshots or fixed polar-angle snapshots.",
    )
    parser.add_argument(
        "--snapshot-step-interval",
        type=int,
        default=DEFAULT_SNAPSHOT_STEP_INTERVAL,
        help="Draw one vehicle every N time steps after the first frame.",
    )
    parser.add_argument(
        "--snapshot-interval-seconds",
        type=float,
        default=DEFAULT_SNAPSHOT_INTERVAL_SECONDS,
        help="Draw one vehicle every N seconds after the first frame when --vehicle-placement=steps.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=DEFAULT_DT,
        help="Sample time in seconds for headerless rollout CSVs.",
    )
    parser.add_argument(
        "--time-min",
        type=float,
        default=DEFAULT_TIME_MIN,
        help="Start time in seconds for the plotted CSV trajectory.",
    )
    parser.add_argument(
        "--time-max",
        type=float,
        default=DEFAULT_TIME_MAX,
        help="End time in seconds for the plotted CSV trajectory.",
    )
    parser.add_argument(
        "--vehicle-time-max",
        type=float,
        default=DEFAULT_VEHICLE_TIME_MAX,
        help="End time in seconds for vehicle snapshots.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    parser.add_argument("--fig-width", type=float, default=7.2, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=7.2, help="Figure height in inches.")
    parser.add_argument(
        "--hide-radius-label",
        action="store_true",
        help="Hide the radius dimension label.",
    )
    parser.add_argument(
        "--radius-label-angle",
        type=float,
        default=DEFAULT_RADIUS_LABEL_ANGLE_DEG,
        help="Angle in degrees where the radius dimension line is drawn.",
    )
    parser.add_argument(
        "--radius-label-font-size",
        type=float,
        default=DEFAULT_RADIUS_LABEL_FONT_SIZE,
        help="Font size for the radius label.",
    )
    parser.add_argument(
        "--center-marker-radius",
        type=float,
        default=DEFAULT_CENTER_MARKER_RADIUS,
        help="Radius of the small center marker circle.",
    )
    parser.add_argument(
        "--plot-reference",
        action="store_true",
        help="Also draw the ideal fixed-circle reference as a pale dashed line.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=["auto", "csv", "fixed_circle"],
        default="auto",
        help="Green path source. Auto uses CSV when it has enough samples.",
    )
    parser.add_argument(
        "--reference-start-angle",
        type=float,
        default=-90.0,
        help="Start angle in degrees for the ideal reference path.",
    )
    parser.add_argument(
        "--reference-sweep",
        type=float,
        default=360.0,
        help="Absolute sweep angle in degrees for the ideal reference path.",
    )
    parser.add_argument(
        "--direction",
        choices=["ccw", "cw"],
        default="ccw",
        help="Direction used for reference headings and reference-path arrows.",
    )
    return parser


def extract_fixed_circle_config(fixed_circle_source: Path) -> FixedCircleConfig:
    config = FixedCircleConfig()
    if not fixed_circle_source.exists():
        print(f"[WARN] Fixed-circle source not found, using defaults: {fixed_circle_source}")
        return config

    text = fixed_circle_source.read_text(encoding="utf-8", errors="ignore")
    radius_match = re.search(r"\bself\.radius\s*=\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))", text)
    if radius_match:
        config.radius = float(radius_match.group(1))
        config.center_y = config.radius

    return config


def load_transparent_vehicle_image(image_path: Path) -> Optional[Image.Image]:
    if not image_path.exists():
        print(f"[WARN] Vehicle image not found, falling back to vector car: {image_path}")
        return None
    return transparent_crop_from_image(image_path)


def draw_vehicle_image_transparent(
    ax: plt.Axes,
    image: Image.Image,
    x: float,
    y: float,
    yaw: float,
    *,
    length: float,
    zorder: int,
) -> None:
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    rotated = image.rotate(
        math.degrees(yaw),
        resample=resampling,
        expand=True,
        fillcolor=(255, 255, 255, 0),
    )
    scale = length / max(1, image.width)
    width_m = rotated.width * scale
    height_m = rotated.height * scale
    ax.imshow(
        np.asarray(rotated),
        extent=(x - width_m / 2.0, x + width_m / 2.0, y - height_m / 2.0, y + height_m / 2.0),
        origin="upper",
        interpolation="lanczos",
        zorder=zorder,
    )


def direction_sign(direction: str) -> float:
    return -1.0 if direction == "cw" else 1.0


def reference_xy(config: FixedCircleConfig, angles_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return (
        config.center_x + config.radius * np.cos(angles_rad),
        config.center_y + config.radius * np.sin(angles_rad),
    )


def reference_heading(angles_rad: np.ndarray, direction: str) -> np.ndarray:
    return angles_rad + direction_sign(direction) * math.pi / 2.0


def build_fixed_circle_reference_path(
    config: FixedCircleConfig,
    start_angle_deg: float,
    sweep_deg: float,
    direction: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    sweep_rad = math.radians(abs(float(sweep_deg)))
    start_rad = math.radians(float(start_angle_deg))
    theta = start_rad + direction_sign(direction) * np.linspace(
        0.0,
        sweep_rad,
        config.num_points,
        endpoint=False,
    )
    ref_x, ref_y = reference_xy(config, theta)
    return ref_x, ref_y, theta


def circle_angles_for_telemetry(telemetry: Telemetry, config: FixedCircleConfig) -> np.ndarray:
    raw_angles = np.arctan2(telemetry.y - config.center_y, telemetry.x - config.center_x)
    return np.unwrap(raw_angles)


def align_query_angles_to_telemetry(
    query_angles: np.ndarray,
    telemetry_angles: np.ndarray,
) -> np.ndarray:
    if len(telemetry_angles) < 2:
        return query_angles

    angle_min = float(np.nanmin(telemetry_angles))
    angle_max = float(np.nanmax(telemetry_angles))
    middle = (angle_min + angle_max) / 2.0
    aligned = []
    for angle in np.asarray(query_angles, dtype=float):
        turns = round((middle - float(angle)) / (2.0 * math.pi))
        aligned.append(float(angle) + turns * 2.0 * math.pi)
    return np.asarray(aligned, dtype=float)


def interpolate_by_angle(
    angles: np.ndarray,
    values: np.ndarray,
    query_angles: Sequence[float],
) -> np.ndarray:
    order = np.argsort(angles)
    angle_sorted = np.asarray(angles, dtype=float)[order]
    value_sorted = np.asarray(values, dtype=float)[order]
    unique_angles, unique_indices = np.unique(angle_sorted, return_index=True)
    unique_values = value_sorted[unique_indices]
    if unique_angles.shape[0] < 2:
        return np.full(len(query_angles), float(unique_values[0]))
    return np.interp(np.asarray(query_angles, dtype=float), unique_angles, unique_values)


def draw_fixed_circle_road(
    ax: plt.Axes,
    config: FixedCircleConfig,
    visual_lane_width: float,
) -> float:
    outer_radius = config.radius + visual_lane_width / 2.0
    inner_radius = max(config.radius - visual_lane_width / 2.0, 0.05)

    ax.add_patch(
        Wedge(
            (config.center_x, config.center_y),
            outer_radius,
            0.0,
            360.0,
            width=outer_radius - inner_radius,
            facecolor=ROAD_FILL_COLOR,
            edgecolor="none",
            zorder=0,
        )
    )

    theta = np.linspace(0.0, 2.0 * math.pi, config.num_points + 1)
    for radius in (outer_radius, inner_radius):
        ax.plot(
            config.center_x + radius * np.cos(theta),
            config.center_y + radius * np.sin(theta),
            color="black",
            linewidth=2.0,
            zorder=2,
        )

    ax.plot(
        config.center_x + config.radius * np.cos(theta),
        config.center_y + config.radius * np.sin(theta),
        color="black",
        linewidth=1.45,
        linestyle=(0, (5, 4)),
        zorder=2,
    )
    return outer_radius


def add_radius_annotation(
    ax: plt.Axes,
    config: FixedCircleConfig,
    angle_deg: float,
    font_size: float,
) -> None:
    angle = math.radians(float(angle_deg))
    start = np.array([config.center_x, config.center_y], dtype=float)
    end = np.array(
        [
            config.center_x + config.radius * math.cos(angle),
            config.center_y + config.radius * math.sin(angle),
        ],
        dtype=float,
    )
    color = "#4f4f4f"
    ax.add_patch(
        FancyArrowPatch(
            tuple(start),
            tuple(end),
            arrowstyle="<|-|>",
            mutation_scale=12,
            linewidth=1.15,
            color=color,
            shrinkA=0,
            shrinkB=0,
            zorder=8,
        )
    )

    label_xy = start + 0.54 * (end - start)
    ax.text(
        float(label_xy[0]),
        float(label_xy[1]),
        f"R = {config.radius:g} m",
        color="#222222",
        ha="center",
        va="center",
        fontsize=float(font_size),
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="none", alpha=0.82),
        zorder=9,
    )


def add_center_marker(
    ax: plt.Axes,
    config: FixedCircleConfig,
    marker_radius: float,
) -> None:
    ax.add_patch(
        Circle(
            (config.center_x, config.center_y),
            radius=max(float(marker_radius), 0.0),
            facecolor="#4f4f4f",
            edgecolor="white",
            linewidth=0.8,
            zorder=10,
        )
    )


def build_step_snapshot_indices(num_samples: int, step_interval: int) -> np.ndarray:
    if num_samples <= 0:
        return np.asarray([], dtype=int)

    interval = max(int(step_interval), 1)
    indices = [0]
    indices.extend(range(interval, num_samples, interval))
    return np.asarray(indices, dtype=int)


def snapshot_step_interval_from_seconds(interval_seconds: float, dt: float, fallback_steps: int) -> int:
    if interval_seconds <= 0.0 or dt <= 0.0:
        return max(int(fallback_steps), 1)
    return max(int(round(interval_seconds / dt)), 1)


def filter_telemetry_by_time(
    telemetry: Telemetry,
    dt: float,
    time_min: float,
    time_max: Optional[float],
) -> Telemetry:
    if dt <= 0.0:
        raise ValueError("--dt must be positive when filtering by time.")

    times = np.arange(len(telemetry.x), dtype=float) * float(dt)
    lower = min(float(time_min), float(time_max)) if time_max is not None else float(time_min)
    upper = max(float(time_min), float(time_max)) if time_max is not None else None

    mask = times >= lower - 1e-12
    if upper is not None:
        mask &= times <= upper + 1e-12

    if np.count_nonzero(mask) < 2:
        upper_text = "end" if upper is None else f"{upper:g}s"
        raise ValueError(f"Time window {lower:g}s to {upper_text} contains fewer than two samples.")

    upper_label = "end" if upper is None else f"{upper:g}"
    return Telemetry(
        x=telemetry.x[mask],
        y=telemetry.y[mask],
        psi=telemetry.psi[mask],
        source_format=f"{telemetry.source_format}_time_{lower:g}_{upper_label}",
    )


def automatic_limits(
    config: FixedCircleConfig,
    outer_radius: float,
    scene_padding: float,
    path_x: np.ndarray,
    path_y: np.ndarray,
) -> Tuple[float, float, float, float]:
    finite = np.isfinite(path_x) & np.isfinite(path_y)
    x_values = [config.center_x - outer_radius, config.center_x + outer_radius]
    y_values = [config.center_y - outer_radius, config.center_y + outer_radius]
    if np.any(finite):
        x_values.extend([float(np.min(path_x[finite])), float(np.max(path_x[finite]))])
        y_values.extend([float(np.min(path_y[finite])), float(np.max(path_y[finite]))])

    padding = float(scene_padding)
    return (
        min(x_values) - padding,
        max(x_values) + padding,
        min(y_values) - padding,
        max(y_values) + padding,
    )


def resolve_limits(
    auto_limits: Tuple[float, float, float, float],
    x_min: Optional[float],
    x_max: Optional[float],
    y_min: Optional[float],
    y_max: Optional[float],
) -> Tuple[float, float, float, float]:
    auto_x_min, auto_x_max, auto_y_min, auto_y_max = auto_limits
    return (
        auto_x_min if x_min is None else float(x_min),
        auto_x_max if x_max is None else float(x_max),
        auto_y_min if y_min is None else float(y_min),
        auto_y_max if y_max is None else float(y_max),
    )


def render_image(
    telemetry: Telemetry,
    vehicle_telemetry: Telemetry,
    config: FixedCircleConfig,
    vehicle_angles_deg: Sequence[float],
    output_path: Path,
    lane_width: Optional[float],
    road_width_scale: float,
    axis_limits: Tuple[Optional[float], Optional[float], Optional[float], Optional[float]],
    scene_padding: float,
    vehicle_scale: float,
    vehicle_placement: str,
    snapshot_step_interval: int,
    dpi: int,
    fig_size: Tuple[float, float],
    show_radius_label: bool,
    radius_label_angle_deg: float,
    radius_label_font_size: float,
    center_marker_radius: float,
    plot_reference: bool,
    trajectory_source: str,
    reference_start_angle_deg: float,
    reference_sweep_deg: float,
    direction: str,
    time1_vehicle_angle_deg: float,
    time1_image: Optional[Image.Image],
    times_image: Optional[Image.Image],
) -> None:
    configure_style()

    source_lane_width = float(lane_width if lane_width is not None else config.lane_width)
    visual_lane_width = source_lane_width * float(road_width_scale)
    telemetry_angles = circle_angles_for_telemetry(telemetry, config)
    vehicle_angles = circle_angles_for_telemetry(vehicle_telemetry, config)

    ref_x, ref_y, _ = build_fixed_circle_reference_path(
        config,
        reference_start_angle_deg,
        reference_sweep_deg,
        direction,
    )
    has_csv_path = len(telemetry.x) >= 3 and np.count_nonzero(np.isfinite(telemetry.x + telemetry.y)) >= 3
    if trajectory_source == "auto":
        path_source = "csv" if has_csv_path else "fixed_circle_reference"
    elif trajectory_source == "fixed_circle":
        path_source = "fixed_circle_reference"
    else:
        path_source = "csv"

    if path_source == "fixed_circle_reference":
        path_x = ref_x
        path_y = ref_y
    else:
        path_x = telemetry.x
        path_y = telemetry.y

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor("white")

    outer_radius = draw_fixed_circle_road(ax, config, visual_lane_width)

    if plot_reference and path_source != "fixed_circle_reference":
        ax.plot(
            ref_x,
            ref_y,
            color="#98d998",
            linewidth=1.0,
            linestyle=(0, (3, 3)),
            alpha=0.8,
            zorder=3,
        )

    green = "#1eb51b"
    ax.plot(path_x, path_y, color=green, linewidth=2.1, solid_capstyle="round", zorder=4)
    add_path_arrow(ax, path_x, path_y, green)

    if show_radius_label:
        add_radius_annotation(ax, config, radius_label_angle_deg, radius_label_font_size)

    add_center_marker(ax, config, center_marker_radius)

    if vehicle_placement == "steps":
        snapshot_indices = build_step_snapshot_indices(len(vehicle_telemetry.x), snapshot_step_interval)
        car_x = vehicle_telemetry.x[snapshot_indices]
        car_y = vehicle_telemetry.y[snapshot_indices]
        car_psi = np.unwrap(vehicle_telemetry.psi)[snapshot_indices]
        highlight_index = 0 if len(snapshot_indices) else -1
    else:
        requested_angles_deg = np.asarray(vehicle_angles_deg, dtype=float)
        requested_angles = np.deg2rad(requested_angles_deg)
        aligned_angles = align_query_angles_to_telemetry(requested_angles, vehicle_angles)

        car_x = interpolate_by_angle(vehicle_angles, vehicle_telemetry.x, aligned_angles)
        car_y = interpolate_by_angle(vehicle_angles, vehicle_telemetry.y, aligned_angles)
        car_psi = interpolate_by_angle(vehicle_angles, np.unwrap(vehicle_telemetry.psi), aligned_angles)

        angle_min = float(np.nanmin(vehicle_angles))
        angle_max = float(np.nanmax(vehicle_angles))
        outside = (aligned_angles < angle_min) | (aligned_angles > angle_max)
        if np.any(outside):
            fallback_x, fallback_y = reference_xy(config, aligned_angles[outside])
            car_x[outside] = fallback_x
            car_y[outside] = fallback_y
            car_psi[outside] = reference_heading(aligned_angles[outside], direction)

        highlight_index = int(np.argmin(np.abs(requested_angles_deg - time1_vehicle_angle_deg))) if len(requested_angles_deg) else -1
    for idx, (cx, cy, yaw) in enumerate(zip(car_x, car_y, car_psi)):
        vehicle_image = time1_image if idx == highlight_index else times_image
        if vehicle_image is not None:
            draw_vehicle_image_transparent(
                ax,
                vehicle_image,
                float(cx),
                float(cy),
                float(yaw),
                length=DEFAULT_VEHICLE_LENGTH * float(vehicle_scale),
                zorder=10 + idx,
            )
        else:
            is_highlight = idx == highlight_index
            fill = "#88a96a" if is_highlight else "#f2fff0"
            edge = "#202020" if is_highlight else green
            alpha = 0.95 if is_highlight else 0.78
            draw_vehicle(ax, float(cx), float(cy), float(yaw), fill=fill, edge=edge, alpha=alpha, zorder=10 + idx)

    x_min, x_max, y_min, y_max = resolve_limits(
        automatic_limits(config, outer_radius, scene_padding, np.asarray(path_x), np.asarray(path_y)),
        *axis_limits,
    )
    ax.set_xlim(min(x_min, x_max), max(x_min, x_max))
    ax.set_ylim(min(y_min, y_max), max(y_min, y_max))
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, pad_inches=0)
    plt.close(fig)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = build_parser()
    args = parser.parse_args()

    input_csv = resolve_path(args.input_csv, Path.cwd())
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    output_path = (
        resolve_path(args.output, Path.cwd())
        if args.output
        else input_csv.with_name(f"{input_csv.stem}_fixed_circle_iwd.png")
    )
    fixed_circle_source = resolve_path(args.fixed_circle_source, script_dir)
    time1_image_path = resolve_path(args.time1_image, Path.cwd())
    times_image_path = resolve_path(args.times_image, Path.cwd())

    config = extract_fixed_circle_config(fixed_circle_source)
    if args.circle_radius is not None:
        config.radius = float(args.circle_radius)
        if args.circle_center_y is None:
            config.center_y = config.radius
    if args.circle_center_x is not None:
        config.center_x = float(args.circle_center_x)
    if args.circle_center_y is not None:
        config.center_y = float(args.circle_center_y)
    if args.lane_width is not None:
        config.lane_width = float(args.lane_width)

    raw_telemetry = read_telemetry(input_csv)
    telemetry = filter_telemetry_by_time(
        raw_telemetry,
        dt=args.dt,
        time_min=args.time_min,
        time_max=args.time_max,
    )
    vehicle_telemetry = filter_telemetry_by_time(
        raw_telemetry,
        dt=args.dt,
        time_min=args.time_min,
        time_max=args.vehicle_time_max,
    )
    snapshot_step_interval = snapshot_step_interval_from_seconds(
        args.snapshot_interval_seconds,
        args.dt,
        args.snapshot_step_interval,
    )
    time1_image = load_transparent_vehicle_image(time1_image_path)
    times_image = load_transparent_vehicle_image(times_image_path)
    render_image(
        telemetry=telemetry,
        vehicle_telemetry=vehicle_telemetry,
        config=config,
        vehicle_angles_deg=args.vehicle_angle,
        output_path=output_path,
        lane_width=args.lane_width,
        road_width_scale=args.road_width_scale,
        axis_limits=(args.x_min, args.x_max, args.y_min, args.y_max),
        scene_padding=args.scene_padding,
        vehicle_scale=args.vehicle_scale,
        vehicle_placement=args.vehicle_placement,
        snapshot_step_interval=snapshot_step_interval,
        dpi=args.dpi,
        fig_size=(args.fig_width, args.fig_height),
        show_radius_label=not args.hide_radius_label,
        radius_label_angle_deg=args.radius_label_angle,
        radius_label_font_size=args.radius_label_font_size,
        center_marker_radius=args.center_marker_radius,
        plot_reference=args.plot_reference,
        trajectory_source=args.trajectory_source,
        reference_start_angle_deg=args.reference_start_angle,
        reference_sweep_deg=args.reference_sweep,
        direction=args.direction,
        time1_vehicle_angle_deg=args.time1_vehicle_angle,
        time1_image=time1_image,
        times_image=times_image,
    )

    print(f"Saved fixed_circle_iwd visualization: {output_path}")


if __name__ == "__main__":
    main()
