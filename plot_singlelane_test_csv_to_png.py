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
from PIL import Image

from plot_moose_test_csv_to_png import (
    ROAD_FILL_COLOR,
    Telemetry,
    add_path_arrow,
    clip_path_to_x_range,
    configure_style,
    draw_vehicle,
    draw_vehicle_image,
    interpolate_by_x,
    load_vehicle_image,
    read_telemetry,
    resolve_path,
)


DEFAULT_VEHICLE_X = [5.0, 18.0, 31.0, 44.0, 57.0, 70.0]
DEFAULT_X_MIN = 0.0
DEFAULT_X_MAX = 75.0
DEFAULT_TRAJECTORY_X_MIN = DEFAULT_X_MIN
DEFAULT_TRAJECTORY_X_MAX = DEFAULT_X_MAX
DEFAULT_TIME1_VEHICLE_X = 40.0
DEFAULT_FINAL_LENGTH = 35.0
DEFAULT_ROAD_WIDTH_SCALE = 1.0
DEFAULT_VEHICLE_LENGTH = 4.08


@dataclass
class SingleLaneConfig:
    target_speed: float = 20.0
    lane_a_length: float = 20.0
    curve_ab_length: float = 20.0
    lane_b_length: float = 20.0
    lane_b_offset: float = 3.5
    max_steps: int = 550
    dt: float = 0.01

    @property
    def x1_end(self) -> float:
        return self.lane_a_length

    @property
    def x2_end(self) -> float:
        return self.x1_end + self.curve_ab_length

    @property
    def x3_end(self) -> float:
        return self.x2_end + self.lane_b_length

    @property
    def nominal_total_length(self) -> float:
        return self.x3_end

    @property
    def rollout_total_length(self) -> float:
        return self.target_speed * self.dt * max(self.max_steps - 1, 0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a singlelane CSV rollout to a PNG-style static visualization."
    )
    parser.add_argument("input_csv", type=str, help="Input rollout CSV.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <input_stem>_singlelane_test.png.",
    )
    parser.add_argument(
        "--vehicle-x",
        type=float,
        nargs="+",
        default=DEFAULT_VEHICLE_X,
        help="Vehicle x positions to draw.",
    )
    parser.add_argument(
        "--singlelane-source",
        type=str,
        default="envs/singlelane.py",
        help="Path to singlelane.py used for lane offset and segment lengths.",
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
        "--time1-vehicle-x",
        type=float,
        default=DEFAULT_TIME1_VEHICLE_X,
        help="Deprecated. The first vehicle now always uses --time1-image.",
    )
    parser.add_argument(
        "--lane-width",
        type=float,
        default=None,
        help="Visual lane width. Defaults to lane_b_offset from singlelane.py.",
    )
    parser.add_argument(
        "--road-width-scale",
        type=float,
        default=DEFAULT_ROAD_WIDTH_SCALE,
        help="Scale applied to the visual lane width.",
    )
    parser.add_argument("--x-min", type=float, default=DEFAULT_X_MIN, help="Left x limit.")
    parser.add_argument("--x-max", type=float, default=DEFAULT_X_MAX, help="Right x limit.")
    parser.add_argument(
        "--trajectory-x-min",
        type=float,
        default=DEFAULT_TRAJECTORY_X_MIN,
        help="Left x limit for the green trajectory line.",
    )
    parser.add_argument(
        "--trajectory-x-max",
        type=float,
        default=DEFAULT_TRAJECTORY_X_MAX,
        help="Right x limit for the green trajectory line.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI.")
    parser.add_argument("--fig-width", type=float, default=10.0, help="Figure width in inches.")
    parser.add_argument("--fig-height", type=float, default=2.2, help="Figure height in inches.")
    parser.add_argument(
        "--hide-segments",
        action="store_true",
        help="Hide singlelane length and lane-center distance annotations.",
    )
    parser.add_argument(
        "--plot-reference",
        action="store_true",
        help="Also draw the ideal singlelane reference trajectory as a pale dashed line.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=["auto", "csv", "singlelane"],
        default="auto",
        help="Green path source. Auto uses CSV when it covers all requested vehicle x positions.",
    )
    return parser


def _extract_assignment(scope: str, name: str) -> Optional[float]:
    pattern = rf"\b{name}\s*=\s*([-+]?\d+(?:\.\d+)?)"
    match = re.search(pattern, scope)
    return float(match.group(1)) if match else None


def extract_singlelane_config(singlelane_source: Path) -> SingleLaneConfig:
    config = SingleLaneConfig()
    if not singlelane_source.exists():
        print(f"[WARN] Singlelane source not found, using defaults: {singlelane_source}")
        return config

    text = singlelane_source.read_text(encoding="utf-8", errors="ignore")
    trajectory_match = re.search(
        r"def _generate_perfect_trajectory\(self\):(.*?)(?:\n    def |\Z)",
        text,
        flags=re.S,
    )
    trajectory_scope = trajectory_match.group(1) if trajectory_match else text

    for name in (
        "target_speed",
        "lane_a_length",
        "curve_ab_length",
        "lane_b_length",
        "lane_b_offset",
    ):
        value = _extract_assignment(trajectory_scope, name)
        if value is not None:
            setattr(config, name, value)

    max_steps_match = re.search(r"\bself\.max_steps\s*=\s*(\d+)", text)
    if max_steps_match:
        config.max_steps = int(max_steps_match.group(1))

    dt_match = re.search(r"\bself\.dt\s*=\s*kwargs\.get\(['\"]dt['\"],\s*([-+]?\d+(?:\.\d+)?)\)", text)
    if dt_match:
        config.dt = float(dt_match.group(1))

    return config


def reference_y(config: SingleLaneConfig, x_values: np.ndarray) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    y = np.zeros_like(x)

    mask_curve = (x > config.x1_end) & (x <= config.x2_end)
    if np.any(mask_curve):
        progress = (x[mask_curve] - config.x1_end) / config.curve_ab_length
        y[mask_curve] = config.lane_b_offset * (1.0 - np.cos(np.pi * progress)) / 2.0

    mask_lane_b = x > config.x2_end
    y[mask_lane_b] = config.lane_b_offset
    return y


def reference_heading(config: SingleLaneConfig, x_values: np.ndarray) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    dy_dx = np.zeros_like(x)

    mask_curve = (x > config.x1_end) & (x <= config.x2_end)
    if np.any(mask_curve):
        progress = (x[mask_curve] - config.x1_end) / config.curve_ab_length
        dy_dx[mask_curve] = (
            config.lane_b_offset
            * np.pi
            / config.curve_ab_length
            * np.sin(np.pi * progress)
        )

    return np.arctan2(dy_dx, np.ones_like(dy_dx))


def draw_singlelane_road(
    ax: plt.Axes,
    x_window: Tuple[float, float],
    lane_w: float,
    lane_b_offset: float,
) -> Tuple[float, float]:
    x_min, x_max = x_window
    road_bottom = -lane_w / 2.0
    road_top = lane_b_offset + lane_w / 2.0
    divider_y = lane_b_offset / 2.0

    ax.axhspan(road_bottom, road_top, color=ROAD_FILL_COLOR, zorder=0)
    ax.plot([x_min, x_max], [road_top, road_top], color="black", linewidth=2.0, zorder=2)
    ax.plot([x_min, x_max], [road_bottom, road_bottom], color="black", linewidth=2.0, zorder=2)
    ax.plot(
        [x_min, x_max],
        [divider_y, divider_y],
        color="black",
        linewidth=1.45,
        linestyle=(0, (5, 4)),
        zorder=2,
    )
    return road_bottom, road_top


def add_lane_center_distance_annotation(ax: plt.Axes, config: SingleLaneConfig, x_pos: float) -> None:
    y0 = 0.0
    y1 = config.lane_b_offset
    color = "#303030"
    ax.hlines([y0, y1], x_pos - 0.45, x_pos + 0.45, colors=color, linewidth=0.9, alpha=0.9, zorder=6)
    ax.annotate(
        "",
        xy=(x_pos, y1),
        xytext=(x_pos, y0),
        arrowprops=dict(arrowstyle="<->", color=color, linewidth=1.0, shrinkA=0, shrinkB=0),
        zorder=7,
    )
    ax.text(
        x_pos + 0.65,
        (y0 + y1) / 2.0,
        f"{config.lane_b_offset:g} m",
        color=color,
        ha="left",
        va="center",
        fontsize=11.25,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=0.8),
        zorder=8,
    )


def add_singlelane_key_lengths(ax: plt.Axes, config: SingleLaneConfig, y_pos: float) -> None:
    final_right = config.x2_end + DEFAULT_FINAL_LENGTH
    segments = [
        (0.0, config.x1_end, config.lane_a_length),
        (config.x1_end, config.x2_end, config.curve_ab_length),
        (config.x2_end, final_right, DEFAULT_FINAL_LENGTH),
    ]
    color = "#444444"
    for left, right, length in segments:
        ax.annotate(
            "",
            xy=(right, y_pos),
            xytext=(left, y_pos),
            arrowprops=dict(arrowstyle="<->", color=color, linewidth=0.75, shrinkA=1, shrinkB=1),
            zorder=7,
        )
        ax.text(
            (left + right) / 2.0,
            y_pos + 0.14,
            f"{length:g} m",
            color=color,
            ha="center",
            va="bottom",
            fontsize=15.0,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.78, pad=0.8),
            zorder=8,
        )

    for marker in (0.0, config.x1_end, config.x2_end, final_right):
        ax.vlines(marker, y_pos - 0.14, y_pos + 0.14, colors=color, linewidth=0.7, zorder=7)


def render_image(
    telemetry: Telemetry,
    config: SingleLaneConfig,
    vehicle_x: Sequence[float],
    output_path: Path,
    lane_width: Optional[float],
    road_width_scale: float,
    x_window: Tuple[float, float],
    trajectory_window: Tuple[float, float],
    dpi: int,
    fig_size: Tuple[float, float],
    hide_segments: bool,
    plot_reference: bool,
    trajectory_source: str,
    time1_vehicle_x: float,
    time1_image: Optional[Image.Image],
    times_image: Optional[Image.Image],
) -> None:
    configure_style()

    lane_w = float(lane_width if lane_width is not None else config.lane_b_offset) * float(road_width_scale)

    x_min, x_max = sorted((float(x_window[0]), float(x_window[1])))
    traj_x_min, traj_x_max = sorted((float(trajectory_window[0]), float(trajectory_window[1])))

    ref_x = np.linspace(traj_x_min, traj_x_max, 1600)
    ref_y = reference_y(config, ref_x)
    requested_x = np.asarray(vehicle_x, dtype=float)

    csv_covers_requested = bool(
        len(requested_x) == 0
        or (
            float(np.min(telemetry.x)) <= float(np.min(requested_x)) + 1e-6
            and float(np.max(telemetry.x)) >= float(np.max(requested_x)) - 1e-6
        )
    )
    if trajectory_source == "auto":
        path_source = "csv" if csv_covers_requested else "singlelane_reference"
    elif trajectory_source == "singlelane":
        path_source = "singlelane_reference"
    else:
        path_source = "csv"

    if path_source == "singlelane_reference":
        path_x = ref_x
        path_y = ref_y
    else:
        path_x = telemetry.x
        path_y = telemetry.y
        path_x, path_y = clip_path_to_x_range(path_x, path_y, traj_x_min, traj_x_max)

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor("white")

    road_bottom, road_top = draw_singlelane_road(ax, (x_min, x_max), lane_w, config.lane_b_offset)
    length_label_y = road_top + 0.3
    if not hide_segments:
        add_singlelane_key_lengths(ax, config, length_label_y)
        add_lane_center_distance_annotation(ax, config, x_min + 2.0)

    if plot_reference and path_source != "singlelane_reference":
        ax.plot(ref_x, ref_y, color="#98d998", linewidth=1.0, linestyle=(0, (3, 3)), alpha=0.8, zorder=3)

    green = "#1eb51b"
    ax.plot(path_x, path_y, color=green, linewidth=2.1, solid_capstyle="round", zorder=4)
    add_path_arrow(ax, path_x, path_y, green)

    car_y = interpolate_by_x(telemetry.x, telemetry.y, requested_x)
    car_psi = interpolate_by_x(telemetry.x, np.unwrap(telemetry.psi), requested_x)

    outside = (requested_x < float(np.min(telemetry.x))) | (requested_x > float(np.max(telemetry.x)))
    if np.any(outside):
        car_y[outside] = reference_y(config, requested_x[outside])
        car_psi[outside] = reference_heading(config, requested_x[outside])

    highlight_index = 0 if len(requested_x) else -1
    for idx, (cx, cy, yaw) in enumerate(zip(requested_x, car_y, car_psi)):
        vehicle_image = time1_image if idx == highlight_index else times_image
        if vehicle_image is not None:
            draw_vehicle_image(
                ax,
                vehicle_image,
                float(cx),
                float(cy),
                float(yaw),
                length=DEFAULT_VEHICLE_LENGTH,
                zorder=10 + idx,
            )
        else:
            is_highlight = idx == highlight_index
            fill = "#88a96a" if is_highlight else "#f2fff0"
            edge = "#202020" if is_highlight else green
            alpha = 0.95 if is_highlight else 0.78
            draw_vehicle(ax, float(cx), float(cy), float(yaw), fill=fill, edge=edge, alpha=alpha, zorder=10 + idx)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(road_bottom - 0.35, max(road_top + 0.35, length_label_y + 0.55))
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
        else input_csv.with_name(f"{input_csv.stem}_singlelane_test.png")
    )
    singlelane_source = resolve_path(args.singlelane_source, script_dir)
    time1_image_path = resolve_path(args.time1_image, Path.cwd())
    times_image_path = resolve_path(args.times_image, Path.cwd())

    config = extract_singlelane_config(singlelane_source)
    telemetry = read_telemetry(input_csv)
    time1_image = load_vehicle_image(time1_image_path)
    times_image = load_vehicle_image(times_image_path)
    render_image(
        telemetry=telemetry,
        config=config,
        vehicle_x=args.vehicle_x,
        output_path=output_path,
        lane_width=args.lane_width,
        road_width_scale=args.road_width_scale,
        x_window=(args.x_min, args.x_max),
        trajectory_window=(args.trajectory_x_min, args.trajectory_x_max),
        dpi=args.dpi,
        fig_size=(args.fig_width, args.fig_height),
        hide_segments=args.hide_segments,
        plot_reference=args.plot_reference,
        trajectory_source=args.trajectory_source,
        time1_vehicle_x=args.time1_vehicle_x,
        time1_image=time1_image,
        times_image=times_image,
    )

    print(f"Saved singlelane test visualization: {output_path}")


if __name__ == "__main__":
    main()
