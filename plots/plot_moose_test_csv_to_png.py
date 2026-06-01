from __future__ import annotations

import argparse
import csv
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Polygon
from PIL import Image


# Headerless rollout CSV column order. This matches the writer.writerow call in
# xcar-simulation/gpu_vectorized_car_env.py.
ENV0_COLUMNS_32 = [
    "x",
    "y",
    "psi",
    "x_dot",
    "y_dot",
    "psi_dot",
    "delta",
    "omega_fr",
    "omega_fl",
    "omega_rr",
    "omega_rl",
    "f1",
    "f2",
    "f3",
    "f4",
    "r",
    "beta",
    "v",
    "Zs",
    "phi",
    "theta",
    "Z11",
    "Z12",
    "Z13",
    "Z14",
    "dZs",
    "dphi",
    "dtheta",
    "dZ11",
    "dZ12",
    "dZ13",
    "dZ14",
]
ENV0_COLUMNS_38 = ENV0_COLUMNS_32 + [
    "LLTR_front",
    "LLTR_rear",
    "gamma_fr",
    "gamma_fl",
    "gamma_rr",
    "gamma_rl",
]
ENV0_COLUMNS_48 = ENV0_COLUMNS_38 + [
    "fx_fr",
    "fx_fl",
    "fx_rr",
    "fx_rl",
    "fy_fr",
    "fy_fl",
    "fy_rr",
    "fy_rl",
    "ddphi",
    "ddtheta",
]

DEFAULT_VEHICLE_X = [15.0, 30.0, 45.0, 60.0, 75.0]
ROAD_FILL_COLOR = "#eeeeee"
ROAD_FILL_RGBA = (238, 238, 238, 255)
DEFAULT_ROAD_WIDTH_SCALE = 1.0
DEFAULT_VEHICLE_LENGTH = 4.08
DEFAULT_VEHICLE_WIDTH = 1.615
DEFAULT_X_MIN = 0.0
DEFAULT_X_MAX = 90.0
DEFAULT_TRAJECTORY_X_MIN = DEFAULT_X_MIN
DEFAULT_TRAJECTORY_X_MAX = DEFAULT_X_MAX


@dataclass
class MooseConfig:
    target_speed: float = 20.0
    lane_a_length: float = 22.0
    curve_ab_length: float = 18.0
    lane_b_length: float = 11.0
    curve_bc_length: float = 16.0
    lane_c_length: float = 12.0
    lane_b_offset: float = 3.2

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
    def x4_end(self) -> float:
        return self.x3_end + self.curve_bc_length

    @property
    def total_length(self) -> float:
        return self.x4_end + self.lane_c_length


@dataclass
class Telemetry:
    x: np.ndarray
    y: np.ndarray
    psi: np.ndarray
    source_format: str


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a moose-test CSV rollout to a PNG-style static visualization."
    )
    parser.add_argument("input_csv", type=str, help="Input rollout CSV.")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output image path. Defaults to <input_stem>_moose_test.png.",
    )
    parser.add_argument(
        "--vehicle-x",
        type=float,
        nargs="+",
        default=DEFAULT_VEHICLE_X,
        help="Vehicle x positions to draw.",
    )
    parser.add_argument(
        "--moose-source",
        type=str,
        default="envs/moose.py",
        help="Path to moose.py used for lane offset and segment lengths.",
    )
    parser.add_argument(
        "--time1-image",
        type=str,
        default="time-1.png",
        help="Vehicle image used for the first vehicle snapshot.",
    )
    parser.add_argument(
        "--times-image",
        type=str,
        default="times.png",
        help="Vehicle image used for non-highlighted snapshots.",
    )
    parser.add_argument(
        "--lane-width",
        type=float,
        default=None,
        help="Visual lane width. Defaults to lane_b_offset from moose.py.",
    )
    parser.add_argument(
        "--road-width-scale",
        type=float,
        default=DEFAULT_ROAD_WIDTH_SCALE,
        help="Scale applied to the visual lane width.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=DEFAULT_X_MIN,
        help="Left x limit for the rendered scene.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=DEFAULT_X_MAX,
        help="Right x limit for the rendered scene.",
    )
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
        help="Deprecated; annotations are hidden by default.",
    )
    parser.add_argument(
        "--plot-reference",
        action="store_true",
        help="Also draw the ideal moose reference trajectory as a pale dashed line.",
    )
    parser.add_argument(
        "--trajectory-source",
        choices=["auto", "csv", "moose"],
        default="moose",
        help="Green path source. Defaults to the moose.py reference trajectory.",
    )
    return parser


def resolve_path(path_like: str, base_dir: Path) -> Path:
    path = Path(path_like).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def crop_rgba_array_to_alpha(arr: np.ndarray, pad: int = 4) -> np.ndarray:
    alpha = arr[:, :, 3] > 0
    if not np.any(alpha):
        return arr

    rows, cols = np.where(alpha)
    top = max(0, int(rows.min()) - pad)
    bottom = min(arr.shape[0], int(rows.max()) + pad + 1)
    left = max(0, int(cols.min()) - pad)
    right = min(arr.shape[1], int(cols.max()) + pad + 1)
    return arr[top:bottom, left:right].copy()


def clear_edge_connected_halo(arr: np.ndarray) -> np.ndarray:
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    alpha = arr[:, :, 3]
    max_rgb = np.max(rgb, axis=2)
    min_rgb = np.min(rgb, axis=2)
    saturation = np.zeros_like(max_rgb)
    np.divide(max_rgb - min_rgb, max_rgb, out=saturation, where=max_rgb > 1e-6)

    # Remove only pixels connected to the outside, so interior windows and
    # styling lines survive. This targets exported shadow/halo pixels.
    halo_candidate = (alpha == 0) | (
        (saturation < 0.30)
        & (max_rgb < 0.92)
    )

    height, width = halo_candidate.shape
    outside = np.zeros((height, width), dtype=bool)
    queue: deque[Tuple[int, int]] = deque()

    def enqueue(row: int, col: int) -> None:
        if halo_candidate[row, col] and not outside[row, col]:
            outside[row, col] = True
            queue.append((row, col))

    for col in range(width):
        enqueue(0, col)
        enqueue(height - 1, col)
    for row in range(height):
        enqueue(row, 0)
        enqueue(row, width - 1)

    while queue:
        row, col = queue.popleft()
        if row > 0:
            enqueue(row - 1, col)
        if row + 1 < height:
            enqueue(row + 1, col)
        if col > 0:
            enqueue(row, col - 1)
        if col + 1 < width:
            enqueue(row, col + 1)

    arr[outside, :3] = 255
    arr[outside, 3] = 0
    return arr


def extract_moose_config(moose_source: Path) -> MooseConfig:
    config = MooseConfig()
    if not moose_source.exists():
        print(f"[WARN] Moose source not found, using defaults: {moose_source}")
        return config

    text = moose_source.read_text(encoding="utf-8", errors="ignore")
    values = {}
    for name in (
        "target_speed",
        "lane_a_length",
        "curve_ab_length",
        "lane_b_length",
        "curve_bc_length",
        "lane_c_length",
        "lane_b_offset",
    ):
        match = re.search(rf"\b{name}\s*=\s*([-+]?\d+(?:\.\d+)?)", text)
        if match:
            values[name] = float(match.group(1))

    for name, value in values.items():
        setattr(config, name, value)
    return config


def is_float(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


def has_numeric_row(row: Sequence[str]) -> bool:
    return bool(row) and all(is_float(item.strip()) for item in row if item.strip() != "")


def read_rows(csv_path: Path) -> List[List[str]]:
    try:
        handle = csv_path.open("r", encoding="utf-8-sig", newline="")
        rows = list(csv.reader(handle))
        handle.close()
    except UnicodeDecodeError:
        handle = csv_path.open("r", encoding="gbk", newline="")
        rows = list(csv.reader(handle))
        handle.close()

    return [[cell.strip() for cell in row] for row in rows if any(cell.strip() for cell in row)]


def assign_headerless_columns(width: int) -> List[str]:
    if width >= len(ENV0_COLUMNS_48):
        base = ENV0_COLUMNS_48
    elif width >= len(ENV0_COLUMNS_38):
        base = ENV0_COLUMNS_38
    elif width >= len(ENV0_COLUMNS_32):
        base = ENV0_COLUMNS_32
    elif width >= 3:
        base = ENV0_COLUMNS_32
    else:
        raise ValueError(f"CSV has only {width} columns; need at least x, y, psi.")

    return base[:width] + [f"col_{idx}" for idx in range(len(base), width)]


def rows_to_columns(rows: Sequence[Sequence[str]], columns: Sequence[str]) -> Dict[str, np.ndarray]:
    width = len(columns)
    parsed: List[List[float]] = []
    for idx, row in enumerate(rows, start=1):
        if len(row) < min(3, width):
            continue
        values = []
        for col_idx in range(width):
            if col_idx >= len(row) or row[col_idx] == "":
                values.append(np.nan)
            else:
                values.append(float(row[col_idx]))
        parsed.append(values)

    if not parsed:
        raise ValueError("CSV contains no numeric data rows.")

    matrix = np.asarray(parsed, dtype=float)
    return {name: matrix[:, idx] for idx, name in enumerate(columns)}


def normalize_name(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", name.strip().lower())


def read_telemetry(csv_path: Path) -> Telemetry:
    rows = read_rows(csv_path)
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    first_row_is_numeric = has_numeric_row(rows[0])
    if first_row_is_numeric:
        width = len(rows[0])
        columns = assign_headerless_columns(width)
        data = rows_to_columns(rows, columns)
        source_format = f"headerless_{width}_col"
    else:
        header = [normalize_name(item) for item in rows[0]]
        data = rows_to_columns(rows[1:], header)
        source_format = "headered"

    def get_column(candidates: Iterable[str], required: bool = True) -> Optional[np.ndarray]:
        for name in candidates:
            normalized = normalize_name(name)
            if normalized in data:
                return data[normalized]
        if required:
            raise ValueError(f"Missing required column. Tried: {', '.join(candidates)}")
        return None

    x = get_column(["x", "pos_x", "x_pos", "ego_x"])
    y = get_column(["y", "pos_y", "y_pos", "ego_y"])
    psi = get_column(["psi", "yaw", "heading"], required=False)
    if psi is None:
        psi = estimate_heading(x, y)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(psi)
    if np.count_nonzero(finite) < 2:
        raise ValueError("Need at least two finite x/y/psi samples.")

    return Telemetry(x=x[finite], y=y[finite], psi=psi[finite], source_format=source_format)


def estimate_heading(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = np.gradient(x)
    dy = np.gradient(y)
    return np.arctan2(dy, dx)


def reference_y(config: MooseConfig, x_values: np.ndarray) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    y = np.zeros_like(x)

    mask_ab = (x > config.x1_end) & (x <= config.x2_end)
    if np.any(mask_ab):
        progress = (x[mask_ab] - config.x1_end) / config.curve_ab_length
        y[mask_ab] = config.lane_b_offset * (1.0 - np.cos(np.pi * progress)) / 2.0

    mask_b = (x > config.x2_end) & (x <= config.x3_end)
    y[mask_b] = config.lane_b_offset

    mask_bc = (x > config.x3_end) & (x <= config.x4_end)
    if np.any(mask_bc):
        progress = (x[mask_bc] - config.x3_end) / config.curve_bc_length
        y[mask_bc] = config.lane_b_offset * (1.0 + np.cos(np.pi * progress)) / 2.0

    return y


def reference_heading(config: MooseConfig, x_values: np.ndarray) -> np.ndarray:
    x = np.asarray(x_values, dtype=float)
    dy_dx = np.zeros_like(x)

    mask_ab = (x > config.x1_end) & (x <= config.x2_end)
    if np.any(mask_ab):
        progress = (x[mask_ab] - config.x1_end) / config.curve_ab_length
        dy_dx[mask_ab] = (
            config.lane_b_offset
            * np.pi
            / config.curve_ab_length
            * np.sin(np.pi * progress)
        )

    mask_bc = (x > config.x3_end) & (x <= config.x4_end)
    if np.any(mask_bc):
        progress = (x[mask_bc] - config.x3_end) / config.curve_bc_length
        dy_dx[mask_bc] = (
            -config.lane_b_offset
            * np.pi
            / config.curve_bc_length
            * np.sin(np.pi * progress)
        )

    return np.arctan2(dy_dx, np.ones_like(dy_dx))


def scale_heading_for_visual_y(heading: np.ndarray, y_scale: float) -> np.ndarray:
    return np.arctan2(np.sin(heading) * y_scale, np.cos(heading))


def interpolate_by_x(x: np.ndarray, values: np.ndarray, query_x: Sequence[float]) -> np.ndarray:
    order = np.argsort(x)
    x_sorted = x[order]
    values_sorted = values[order]
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_values = values_sorted[unique_indices]
    if unique_x.shape[0] < 2:
        return np.full(len(query_x), float(unique_values[0]))
    return np.interp(np.asarray(query_x, dtype=float), unique_x, unique_values)


def clip_path_to_x_range(x: np.ndarray, y: np.ndarray, x_min: float, x_max: float) -> Tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = np.asarray(x, dtype=float)[order]
    y_sorted = np.asarray(y, dtype=float)[order]
    unique_x, unique_indices = np.unique(x_sorted, return_index=True)
    unique_y = y_sorted[unique_indices]

    if unique_x.shape[0] < 2:
        return unique_x, unique_y

    left = max(float(x_min), float(unique_x[0]))
    right = min(float(x_max), float(unique_x[-1]))
    if left >= right:
        return unique_x, unique_y

    inside = (unique_x > left) & (unique_x < right)
    clipped_x = np.concatenate([[left], unique_x[inside], [right]])
    clipped_y = np.concatenate(
        [
            [float(np.interp(left, unique_x, unique_y))],
            unique_y[inside],
            [float(np.interp(right, unique_x, unique_y))],
        ]
    )
    return clipped_x, clipped_y


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "axes.linewidth": 0.0,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def transform_points(points: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=float)
    return points @ rotation.T + np.array([x, y], dtype=float)


def transparent_crop_from_image(image_path: Path) -> Image.Image:
    raw = Image.open(image_path)
    image = raw.convert("RGBA")
    arr = np.asarray(image, dtype=np.uint8).copy()

    original_alpha = arr[:, :, 3]
    if raw.mode in ("RGBA", "LA") or "transparency" in raw.info:
        # Clean transparent-edge RGB before rotation. Some editors leave black
        # RGB values under transparent pixels; PIL interpolation can pull those
        # into the visible edge as a dotted dark border.
        arr[original_alpha < 128, 3] = 0
        arr[arr[:, :, 3] == 0, :3] = 255
        arr = crop_rgba_array_to_alpha(arr)
        arr = clear_edge_connected_halo(arr)
        arr = crop_rgba_array_to_alpha(arr)
        return Image.fromarray(arr, mode="RGBA")

    rgb = arr[:, :, :3].astype(np.float32) / 255.0

    max_rgb = np.max(rgb, axis=2)
    min_rgb = np.min(rgb, axis=2)
    saturation = np.zeros_like(max_rgb)
    np.divide(max_rgb - min_rgb, max_rgb, out=saturation, where=max_rgb > 1e-6)

    # Uploaded PNGs use a baked checkerboard background. Treat neutral, bright
    # pixels connected to the image edge as background so pale car interiors stay.
    background_candidate = (saturation < 0.28) & (max_rgb > 0.50)
    height, width = background_candidate.shape
    background = np.zeros((height, width), dtype=bool)
    queue: deque[Tuple[int, int]] = deque()

    def enqueue(row: int, col: int) -> None:
        if background_candidate[row, col] and not background[row, col]:
            background[row, col] = True
            queue.append((row, col))

    for col in range(width):
        enqueue(0, col)
        enqueue(height - 1, col)
    for row in range(height):
        enqueue(row, 0)
        enqueue(row, width - 1)

    while queue:
        row, col = queue.popleft()
        if row > 0:
            enqueue(row - 1, col)
        if row + 1 < height:
            enqueue(row + 1, col)
        if col > 0:
            enqueue(row, col - 1)
        if col + 1 < width:
            enqueue(row, col + 1)

    arr[background, :3] = 255
    arr[:, :, 3] = np.where(background, 0, 255).astype(np.uint8)
    alpha = arr[:, :, 3] > 0
    if not np.any(alpha):
        return image

    rows, cols = np.where(alpha)
    pad = 4
    top = max(0, int(rows.min()) - pad)
    bottom = min(height, int(rows.max()) + pad + 1)
    left = max(0, int(cols.min()) - pad)
    right = min(width, int(cols.max()) + pad + 1)
    return Image.fromarray(arr, mode="RGBA").crop((left, top, right, bottom))


def load_vehicle_image(image_path: Path) -> Optional[Image.Image]:
    if not image_path.exists():
        print(f"[WARN] Vehicle image not found, falling back to vector car: {image_path}")
        return None
    image = transparent_crop_from_image(image_path)
    background = Image.new("RGBA", image.size, ROAD_FILL_RGBA)
    background.alpha_composite(image)
    return background


def draw_vehicle_image(
    ax: plt.Axes,
    image: Image.Image,
    x: float,
    y: float,
    yaw: float,
    *,
    length: float = DEFAULT_VEHICLE_LENGTH,
    zorder: int,
) -> None:
    resampling = getattr(getattr(Image, "Resampling", Image), "BICUBIC")
    rotated = image.rotate(
        math.degrees(yaw),
        resample=resampling,
        expand=True,
        fillcolor=ROAD_FILL_RGBA,
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


def vehicle_shapes(
    length: float = DEFAULT_VEHICLE_LENGTH,
    width: float = DEFAULT_VEHICLE_WIDTH,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    body = np.array(
        [
            [-0.48 * length, -0.43 * width],
            [0.35 * length, -0.43 * width],
            [0.50 * length, -0.25 * width],
            [0.50 * length, 0.25 * width],
            [0.35 * length, 0.43 * width],
            [-0.48 * length, 0.43 * width],
        ],
        dtype=float,
    )
    cabin = np.array(
        [
            [-0.12 * length, -0.31 * width],
            [0.20 * length, -0.27 * width],
            [0.29 * length, 0.00 * width],
            [0.20 * length, 0.27 * width],
            [-0.12 * length, 0.31 * width],
            [-0.24 * length, 0.00 * width],
        ],
        dtype=float,
    )

    wheel_length = 0.72
    wheel_width = 0.24
    front_x = 0.26 * length
    rear_x = -0.30 * length
    side_y = 0.52 * width

    def wheel(cx: float, cy: float) -> np.ndarray:
        return np.array(
            [
                [cx - wheel_length / 2, cy - wheel_width / 2],
                [cx + wheel_length / 2, cy - wheel_width / 2],
                [cx + wheel_length / 2, cy + wheel_width / 2],
                [cx - wheel_length / 2, cy + wheel_width / 2],
            ],
            dtype=float,
        )

    wheels = [
        wheel(front_x, -side_y),
        wheel(front_x, side_y),
        wheel(rear_x, -side_y),
        wheel(rear_x, side_y),
    ]
    return body, cabin, wheels


def draw_vehicle(
    ax: plt.Axes,
    x: float,
    y: float,
    yaw: float,
    *,
    fill: str,
    edge: str,
    alpha: float,
    zorder: int,
) -> None:
    body, cabin, wheels = vehicle_shapes()

    for wheel in wheels:
        ax.add_patch(
            Polygon(
                transform_points(wheel, x, y, yaw),
                closed=True,
                facecolor="#242424",
                edgecolor="none",
                alpha=0.92 * alpha,
                zorder=zorder,
            )
        )

    ax.add_patch(
        Polygon(
            transform_points(body, x, y, yaw),
            closed=True,
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.2,
            alpha=alpha,
            joinstyle="round",
            zorder=zorder + 1,
        )
    )
    ax.add_patch(
        Polygon(
            transform_points(cabin, x, y, yaw),
            closed=True,
            facecolor="#ecffe9",
            edgecolor=edge,
            linewidth=0.8,
            alpha=0.85 * alpha,
            joinstyle="round",
            zorder=zorder + 2,
        )
    )


def add_path_arrow(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 3:
        return

    end_idx = len(x) - 1
    start_idx = max(0, end_idx - 8)
    while start_idx > 0 and math.hypot(x[end_idx] - x[start_idx], y[end_idx] - y[start_idx]) < 1.0:
        start_idx -= 1

    arrow = FancyArrowPatch(
        (float(x[start_idx]), float(y[start_idx])),
        (float(x[end_idx]), float(y[end_idx])),
        arrowstyle="-|>",
        mutation_scale=14,
        linewidth=0,
        color=color,
        zorder=5,
    )
    ax.add_patch(arrow)


def add_lane_center_distance_annotation(ax: plt.Axes, config: MooseConfig, x_pos: float) -> None:
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


def add_segment_labels(ax: plt.Axes, config: MooseConfig, y_pos: float) -> None:
    segments = [
        (0.0, config.x1_end, f"{config.lane_a_length:g} m"),
        (config.x1_end, config.x2_end, f"{config.curve_ab_length:g} m"),
        (config.x2_end, config.x3_end, f"{config.lane_b_length:g} m"),
        (config.x3_end, config.x4_end, f"{config.curve_bc_length:g} m"),
        (config.x4_end, config.total_length, f"{config.lane_c_length:g} m"),
    ]

    color = "#555555"
    for left, right, label in segments:
        ax.annotate(
            "",
            xy=(right, y_pos),
            xytext=(left, y_pos),
            arrowprops=dict(arrowstyle="<->", color=color, linewidth=0.75, shrinkA=1, shrinkB=1),
            zorder=7,
        )
        ax.text(
            (left + right) / 2.0,
            y_pos + 0.18,
            label,
            color=color,
            ha="center",
            va="bottom",
            fontsize=7.5,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=1.0),
            zorder=8,
        )

    for marker in (config.x1_end, config.x2_end, config.x3_end, config.x4_end, config.total_length):
        ax.vlines(marker, y_pos - 0.16, y_pos + 0.16, colors=color, linewidth=0.75, zorder=7)


def add_moose_key_lengths(ax: plt.Axes, config: MooseConfig, y_pos: float, final_right: float) -> None:
    final_right = max(float(final_right), config.x4_end)
    segments = [
        (0.0, config.x1_end, config.lane_a_length),
        (config.x1_end, config.x2_end, config.curve_ab_length),
        (config.x2_end, config.x3_end, config.lane_b_length),
        (config.x3_end, config.x4_end, config.curve_bc_length),
        (config.x4_end, final_right, final_right - config.x4_end),
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
    for marker in (0.0, config.x1_end, config.x2_end, config.x3_end, config.x4_end, final_right):
        ax.vlines(marker, y_pos - 0.14, y_pos + 0.14, colors=color, linewidth=0.7, zorder=7)


def add_moose_length_dimensions(ax: plt.Axes, config: MooseConfig, road_bottom: float) -> float:
    segments = [
        (config.x1_end, config.x2_end, config.curve_ab_length),
        (config.x2_end, config.x3_end, config.lane_b_length),
        (config.x3_end, config.x4_end, config.curve_bc_length),
    ]

    # 使用更柔和的高级灰，避免纯黑带来的生硬感
    color = "#666666" 
    
    # 调整标注线基准高度
    dimension_y = road_bottom - 1.0 
    
    # 工程制图规范：辅助线不应直接接触被测物体（留出 gap），并向下略微伸出标注线
    extension_top = road_bottom - 0.15
    extension_bottom = dimension_y - 0.25

    for left, right, length in segments:
        # 1. 绘制垂直辅助线 (Extension lines)
        ax.vlines([left, right], extension_top, extension_bottom, colors=color, linewidth=0.8, zorder=7)
        
        # 2. 绘制带专业 CAD 箭头的水平尺寸线 (Dimension line)
        ax.annotate(
            "",
            xy=(left, dimension_y),
            xytext=(right, dimension_y),
            arrowprops=dict(
                arrowstyle="<|-|>",   # 使用实心三角箭头
                color=color,
                linewidth=0.8,
                mutation_scale=10,    # 控制箭头大小
                shrinkA=0, 
                shrinkB=0
            ),
            zorder=7,
        )
        
        # 3. 绘制尺寸文本，使用圆角背景框打断线条，更显精致
        ax.text(
            (left + right) / 2.0,
            dimension_y,
            f"{length:g} m",
            color="#222222",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=1.0),
            zorder=8,
        )
        
    return dimension_y


def render_image(
    telemetry: Telemetry,
    config: MooseConfig,
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
    time1_image: Optional[Image.Image],
    times_image: Optional[Image.Image],
) -> None:
    configure_style()

    lane_w = float(lane_width if lane_width is not None else config.lane_b_offset) * float(road_width_scale)
    road_bottom = -lane_w / 2.0
    road_top = config.lane_b_offset + lane_w / 2.0
    divider_y = config.lane_b_offset / 2.0

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
        path_source = "csv" if csv_covers_requested else "moose_reference"
    elif trajectory_source == "moose":
        path_source = "moose_reference"
    else:
        path_source = "csv"

    if path_source == "moose_reference":
        path_x = ref_x
        path_y = ref_y
    else:
        path_x = telemetry.x
        path_y = telemetry.y
        path_x, path_y = clip_path_to_x_range(path_x, path_y, traj_x_min, traj_x_max)

    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.set_facecolor("white")

    # Road fill and edge lines in the environment's original y coordinates.
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
    length_label_y = road_top + 0.3
    if not hide_segments:
        add_moose_key_lengths(ax, config, length_label_y, x_max)
        add_lane_center_distance_annotation(ax, config, x_min + 2.0)

    if plot_reference and path_source != "moose_reference":
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
        else input_csv.with_name(f"{input_csv.stem}_moose_test.png")
    )
    moose_source = resolve_path(args.moose_source, script_dir)
    time1_image_path = resolve_path(args.time1_image, Path.cwd())
    times_image_path = resolve_path(args.times_image, Path.cwd())

    config = extract_moose_config(moose_source)
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
        time1_image=time1_image,
        times_image=times_image,
    )

    print(f"Saved moose test visualization: {output_path}")


if __name__ == "__main__":
    main()
