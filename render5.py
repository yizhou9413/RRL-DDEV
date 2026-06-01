import argparse
from dataclasses import dataclass, field
import glob
import os
from pathlib import Path
import re
import textwrap
import zipfile
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

try:
    import tkinter as tk
except Exception:
    tk = None


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


MetricFunction = Callable[[Dict[str, object]], float]
MetricDefinition = Tuple[str, str, MetricFunction]
FAILURE_MODE_COMBINED = "combined"
FAILURE_MODE_TRAJECTORY_ERROR_ONLY = "trajectory_error_only"


COL_NAMES_32 = [
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
COL_NAMES_EXTRA = [
    "LLTR_front",
    "LLTR_rear",
    "gamma_fr",
    "gamma_fl",
    "gamma_rr",
    "gamma_rl",
]
COL_NAMES_38 = COL_NAMES_32 + COL_NAMES_EXTRA
RENDER_NPZ_NAMES = ("render_rollout.npz",)
SUPPORTED_ROLLOUT_SUFFIXES = (".npz", ".csv")

IEEE_SINGLE_COLUMN_WIDTH = 3.5
IEEE_DOUBLE_COLUMN_WIDTH = 7.16
DEFAULT_FIGURE_HEIGHT = 6.35
DEFAULT_OUTPUT_NAME = "trajectory_analysis"
DEFAULT_INPUT_DIR = "/home/zhouyi/lane-change/runs/a2c_continuous_singlelane_default"
DEFAULT_RENDER_DESCRIPTION = "Render publication-quality trajectory analysis figures."
DEFAULT_ROLLOUT_DT = 0.01
DEFAULT_SUMMARY_ONLY_TRAJECTORY_THRESHOLD = 100
BCDF_STACK_PANEL_STEM = "g_bcdf_stack"

SUCCESS_COLOR = "#006d77"
FAILURE_COLOR = "#bc6c25"
MEAN_COLOR = "#0b4f6c"
BAND_COLOR = "#76b7b2"
TRAJECTORY_COLOR = "#87a8c7"
THRESHOLD_COLOR = "#b22222"
GRID_COLOR = "#d9d9d9"
SPINE_COLOR = "#5a5a5a"
TEXT_MUTED = "#4f4f4f"
DEFAULT_PUBLICATION_STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIX Two Text", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "axes.linewidth": 0.9,
    "axes.edgecolor": SPINE_COLOR,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.minor.width": 0.5,
    "ytick.minor.width": 0.5,
    "legend.fontsize": 8,
    "legend.frameon": True,
    "legend.framealpha": 0.96,
    "legend.edgecolor": "#c7c7c7",
    "grid.color": GRID_COLOR,
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
    "lines.linewidth": 1.6,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
SUMMARY_MEAN_RADAR_COLORS = (
    "#4C78A8",
    "#F58518",
    "#54A24B",
    "#E45756",
    "#72B7B2",
    "#B279A2",
    "#FF9DA6",
    "#9D755D",
    "#BAB0AC",
    "#5F4B8B",
    "#0081A7",
    "#D62828",
)
SUMMARY_MEAN_RADAR_RANGES = {
    "max_beta": (5.0, 9.0),
    "max_phi": (1.0, 7.5),
    "max_theta": (1.0, 6.5),
    "rms_beta": (2.5, 5.5),
    "rms_phi": (0.3, 2.7),
    "rms_theta": (0.5, 2.6),
    "traj_dev_rmse": (0.05, 0.14),
    "traj_dev_max": (0.15, 0.38),
}
SUPPORTED_ENV_NAMES = ("singlelane", "moose", "fixed_circle_iwd")
TRAJECTORY_DEVIATION_METRIC_KEY = "traj_dev_rmse"
TRAJECTORY_DEVIATION_MAX_METRIC_KEY = "traj_dev_max"
SINGLELANE_LANE_A_LENGTH = 20.0
SINGLELANE_CURVE_AB_LENGTH = 20.0
SINGLELANE_LANE_B_OFFSET = 3.5
MOOSE_LANE_A_LENGTH = 22.0
MOOSE_CURVE_AB_LENGTH = 18.0
MOOSE_LANE_B_LENGTH = 11.0
MOOSE_CURVE_BC_LENGTH = 16.0
MOOSE_LANE_B_OFFSET = 3.2
FIXED_CIRCLE_CENTER_X = 0.0
FIXED_CIRCLE_CENTER_Y = 12.0
FIXED_CIRCLE_RADIUS = 12.0
FIXED_CIRCLE_RADIUS_ERROR_THRESHOLD = 4.0
RPIQ_ENABLED_PATTERN = re.compile(r"(?:^|[_-])rpiq[_-]?1(?:$|[_-])")
DISTURBED_PATTERN = re.compile(r"distur(?:ed|bed)")
RISKQ_TOKEN_PATTERN = re.compile(r"(?:^|[_\-\s])riskq(?:$|[_\-\s])")
SAMPLED_ADV_SCALE_TOKEN_PATTERN = re.compile(r"(?:^|[_\-\s])sampled_adv_scale(?:$|[_\-\s])")
RISKQ_TRUE_PATTERNS = (
    re.compile(r"use_risk_q\s*[:=]\s*(?:true|1)\b"),
    re.compile(r"use_rararl_q\s*[:=]\s*(?:true|1)\b"),
    re.compile(r"use_risk_q[_\-\s]+(?:true|1)\b"),
    re.compile(r"use_rararl_q[_\-\s]+(?:true|1)\b"),
)
SAMPLED_ADV_SCALE_TRUE_PATTERNS = (
    re.compile(r"rarl_branch\s*[:=]\s*sampled_adv_scale\b"),
    re.compile(r"branch\s*[:=]\s*sampled_adv_scale\b"),
)

RUN_LOG_HINT_CACHE: Dict[str, str] = {}
CHECKPOINT_HINT_CACHE: Dict[str, str] = {}
HIDE_PANEL_TITLES = False


def metric_from_data(data: Dict[str, object], metric_key: str, series_key: str, mode: str) -> float:
    precomputed = data.get(metric_key)
    if precomputed is not None:
        return float(precomputed)

    series = np.asarray(data[series_key], dtype=float)
    if mode == "max_abs":
        return float(np.max(np.abs(series)))
    if mode == "rms":
        return float(np.sqrt(np.mean(np.square(series))))
    raise ValueError(f"Unknown metric mode: {mode}")


METRIC_DEFINITIONS: Tuple[MetricDefinition, ...] = (
    ("max_beta", r"$\max(|\beta|)$", lambda data: metric_from_data(data, "max_beta", "beta", "max_abs")),
    ("max_phi", r"$\max(|\phi|)$", lambda data: metric_from_data(data, "max_phi", "phi", "max_abs")),
    ("max_theta", r"$\max(|\theta|)$", lambda data: metric_from_data(data, "max_theta", "theta", "max_abs")),
    ("rms_beta", r"$\mathrm{RMS}(\beta)$", lambda data: metric_from_data(data, "rms_beta", "beta", "rms")),
    ("rms_phi", r"$\mathrm{RMS}(\phi)$", lambda data: metric_from_data(data, "rms_phi", "phi", "rms")),
    ("rms_theta", r"$\mathrm{RMS}(\theta)$", lambda data: metric_from_data(data, "rms_theta", "theta", "rms")),
)
PRECOMPUTED_TRAJECTORY_VALUE_KEYS = tuple(key for key, _, _ in METRIC_DEFINITIONS) + ("final_x",)
AxisPostprocessor = Callable[[plt.Axes, Dict[str, object]], None]
FigurePostprocessor = Callable[[plt.Figure, Dict[str, object]], None]
ParserCustomizer = Callable[[argparse.ArgumentParser], None]


@dataclass(frozen=True)
class SurvivalPanelConfig:
    stem: str = "a_survival_distance"
    panel_label: str = "(a)"
    title: str = "Survival distance"
    xlabel: str = "Longitudinal distance (m)"
    ylabel: str = "Trajectory rank"
    axis_postprocessor: Optional[AxisPostprocessor] = None
    xlim: Optional[Tuple[float, float]] = None
    reference_line_x: Optional[float] = None
    show_reference_line: bool = True
    show_threshold_label: bool = True
    show_stats_box: bool = True


@dataclass(frozen=True)
class AnglePanelConfig:
    stem: str
    key: str
    ylabel: str
    panel_label: str
    title: str
    show_legend: bool = False
    xlabel: str = "Time step"
    axis_postprocessor: Optional[AxisPostprocessor] = None
    x_key: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None
    show_threshold_lines: bool = True
    threshold_value: Optional[float] = None
    failure_lower: Optional[float] = None
    failure_upper: Optional[float] = None


@dataclass(frozen=True)
class TrajectoryErrorPanelConfig:
    stem: str = "e_trajectory_error"
    panel_label: str = "(e)"
    title: str = "Trajectory error"
    xlabel: Optional[str] = None
    ylabel: str = r"$|e_{\mathrm{traj}}|$ (m)"
    axis_postprocessor: Optional[AxisPostprocessor] = None
    x_key: Optional[str] = None
    xlim: Optional[Tuple[float, float]] = None
    ylim: Tuple[float, float] = (0.0, 0.5)
    show_legend: bool = False
    use_absolute_value: bool = True
    reference_line_values: Tuple[float, ...] = ()


@dataclass(frozen=True)
class MetricsTableConfig:
    panel_label: str = "(e)"
    title: str = "Trajectory-level angle summary"
    axis_postprocessor: Optional[AxisPostprocessor] = None
    sample_label: str = "Successful trajectories"


@dataclass(frozen=True)
class SummaryTableConfig:
    title: str = "Cross-model test summary"
    note_with_trajectory_deviation: str = (
        "Angle metrics are in degrees; trajectory deviation metrics are in meters. "
        "Values are reported as mean $\\pm$ SD."
    )
    note_without_trajectory_deviation: str = (
        "Each row summarizes one algorithm/test folder. Values are in degrees and "
        "reported as mean $\\pm$ SD."
    )
    wrap_width: int = 18
    metric_order: Tuple[str, ...] = ()
    figure_postprocessor: Optional[FigurePostprocessor] = None


@dataclass(frozen=True)
class SummaryMeanRadarConfig:
    title: str = "Summary mean radar"
    subtitle: str = (
        "Mean values normalized against PPO as algorithm / PPO."
    )
    colors: Tuple[str, ...] = SUMMARY_MEAN_RADAR_COLORS
    ranges: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: dict(SUMMARY_MEAN_RADAR_RANGES)
    )
    metric_order: Tuple[str, ...] = ()
    normalized_min_radius: float = 0.35
    expand_ranges_to_data: bool = True
    baseline_algorithm_name: Optional[str] = "PPO"
    ratio_range: Tuple[float, float] = (0.0, 2.0)
    figure_postprocessor: Optional[FigurePostprocessor] = None


@dataclass(frozen=True)
class FigureLayoutConfig:
    gridspec_height_ratios: Tuple[float, ...] = (1.0, 1.0, 0.74)
    gridspec_hspace: float = 0.42
    gridspec_wspace: float = 0.26
    combined_left: float = 0.08
    combined_right: float = 0.985
    combined_bottom: float = 0.07
    combined_top: float = 0.95
    split_panel_min_width: float = 3.55
    split_panel_width_ratio: float = 0.52
    split_panel_min_height: float = 2.65
    split_panel_height_ratio: float = 0.44
    split_panel_left: float = 0.16
    split_panel_right: float = 0.985
    split_panel_bottom: float = 0.16
    split_panel_top_with_label: float = 0.82
    split_panel_top_without_label: float = 0.88
    combined_figure_postprocessor: Optional[FigurePostprocessor] = None
    split_panel_figure_postprocessor: Optional[FigurePostprocessor] = None


DEFAULT_ANGLE_PANELS = (
    AnglePanelConfig("b_roll_angle", "phi", r"$\phi$ (deg)", "(b)", "Roll angle"),
    AnglePanelConfig("c_pitch_angle", "theta", r"$\theta$ (deg)", "(c)", "Pitch angle"),
    AnglePanelConfig("d_sideslip_angle", "beta", r"$\beta$ (deg)", "(d)", "Sideslip angle"),
)


@dataclass(frozen=True)
class RenderProfile:
    env_name: Optional[str] = None
    default_input_dir: str = DEFAULT_INPUT_DIR
    description: str = DEFAULT_RENDER_DESCRIPTION
    default_figure_width: float = IEEE_DOUBLE_COLUMN_WIDTH
    default_figure_height: float = DEFAULT_FIGURE_HEIGHT
    default_output_name: str = DEFAULT_OUTPUT_NAME
    default_threshold_x: float = 90.0
    failure_mode: str = FAILURE_MODE_COMBINED
    summary_x_range: Optional[Tuple[float, float]] = None
    summary_time_range: Optional[Tuple[float, float]] = None
    metric_definitions: Optional[Tuple[MetricDefinition, ...]] = None
    summary_metric_units: Dict[str, str] = field(default_factory=dict)
    metrics_use_successful_trajectories_only: bool = True
    style_overrides: Dict[str, object] = field(default_factory=dict)
    parser_customizer: Optional[ParserCustomizer] = None
    survival_panel: SurvivalPanelConfig = field(default_factory=SurvivalPanelConfig)
    angle_panels: Tuple[AnglePanelConfig, ...] = DEFAULT_ANGLE_PANELS
    trajectory_error_panel: Optional[TrajectoryErrorPanelConfig] = field(
        default_factory=TrajectoryErrorPanelConfig
    )
    signed_trajectory_error_panel: Optional[TrajectoryErrorPanelConfig] = field(
        default_factory=lambda: TrajectoryErrorPanelConfig(
            stem="f_signed_trajectory_error",
            panel_label="(f)",
            title="Trajectory error",
            ylabel=r"$e_{\mathrm{traj}}$ (m)",
            ylim=(-0.5, 0.5),
            use_absolute_value=False,
            reference_line_values=(-0.5, 0.5),
        )
    )
    metrics_table: MetricsTableConfig = field(default_factory=MetricsTableConfig)
    summary_table: SummaryTableConfig = field(default_factory=SummaryTableConfig)
    summary_mean_radar: SummaryMeanRadarConfig = field(default_factory=SummaryMeanRadarConfig)
    layout: FigureLayoutConfig = field(default_factory=FigureLayoutConfig)
    create_figure_fn: Optional[Callable[..., plt.Figure]] = None
    create_split_panel_figures_fn: Optional[Callable[..., List[Tuple[str, plt.Figure]]]] = None
    create_summary_table_figure_fn: Optional[Callable[..., plt.Figure]] = None
    create_summary_mean_radar_figure_fn: Optional[Callable[..., Optional[plt.Figure]]] = None


DEFAULT_RENDER_PROFILE = RenderProfile()


def resolve_render_profile(profile: Optional[RenderProfile] = None) -> RenderProfile:
    return profile or DEFAULT_RENDER_PROFILE


def get_publication_style(profile: Optional[RenderProfile] = None) -> Dict[str, object]:
    resolved_profile = resolve_render_profile(profile)
    style = dict(DEFAULT_PUBLICATION_STYLE)
    style.update(resolved_profile.style_overrides)
    return style


def get_split_panel_stems(profile: Optional[RenderProfile] = None) -> List[str]:
    resolved_profile = resolve_render_profile(profile)
    stems = [
        resolved_profile.survival_panel.stem,
        *(panel.stem for panel in resolved_profile.angle_panels),
    ]
    trajectory_error_panel = resolved_profile.trajectory_error_panel
    if trajectory_error_panel is not None:
        stems.append(trajectory_error_panel.stem)
    signed_trajectory_error_panel = resolved_profile.signed_trajectory_error_panel
    if signed_trajectory_error_panel is not None:
        stems.append(signed_trajectory_error_panel.stem)
    if should_create_bcdf_stack_panel(resolved_profile):
        stems.append(BCDF_STACK_PANEL_STEM)
    return stems


def get_split_panel_filenames(profile: Optional[RenderProfile] = None) -> List[str]:
    return [f"{stem}.png" for stem in get_split_panel_stems(profile)]


def should_create_bcdf_stack_panel(profile: Optional[RenderProfile] = None) -> bool:
    resolved_profile = resolve_render_profile(profile)
    return (
        len(resolved_profile.angle_panels) >= 3
        and resolved_profile.signed_trajectory_error_panel is not None
    )


def build_parser_for_profile(profile: RenderProfile) -> argparse.ArgumentParser:
    return build_parser(profile=profile)


def summarize_model_source_for_profile(
    profile: RenderProfile,
    model_source: Dict[str, object],
    args: argparse.Namespace,
) -> Optional[Dict[str, object]]:
    return summarize_model_source(model_source, args, profile=profile)


def export_summary_table_figure_for_profile(
    profile: RenderProfile,
    summary_records: Sequence[Dict[str, object]],
    output_dir: str,
    output_name: str,
    dpi: int,
    save_pdf: bool,
) -> List[str]:
    return export_summary_table_figure(
        summary_records=summary_records,
        output_dir=output_dir,
        output_name=output_name,
        dpi=dpi,
        save_pdf=save_pdf,
        profile=profile,
    )


def export_summary_mean_radar_figure_for_profile(
    profile: RenderProfile,
    summary_records: Sequence[Dict[str, object]],
    output_dir: str,
    output_name: str,
    dpi: int,
    save_pdf: bool,
) -> List[str]:
    return export_summary_mean_radar_figure(
        summary_records=summary_records,
        output_dir=output_dir,
        output_name=output_name,
        dpi=dpi,
        save_pdf=save_pdf,
        profile=profile,
    )


def apply_publication_style_for_profile(profile: RenderProfile) -> None:
    apply_publication_style(profile)


def main_for_profile(profile: RenderProfile) -> None:
    main(profile=profile)


def apply_publication_style(profile: Optional[RenderProfile] = None) -> None:
    plt.rcParams.update(get_publication_style(profile))


def get_screen_size() -> Tuple[int, int]:
    if tk is None:
        return 1920, 1080
    try:
        root = tk.Tk()
        root.withdraw()
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        root.destroy()
        return screen_width, screen_height
    except Exception:
        return 1920, 1080


def build_parser(
    *,
    profile: Optional[RenderProfile] = None,
    description: str = DEFAULT_RENDER_DESCRIPTION,
    default_input_dir: str = DEFAULT_INPUT_DIR,
    default_env_name: Optional[str] = None,
) -> argparse.ArgumentParser:
    resolved_profile = resolve_render_profile(profile)
    resolved_description = description
    if description == DEFAULT_RENDER_DESCRIPTION:
        resolved_description = resolved_profile.description
    resolved_default_input_dir = default_input_dir
    if default_input_dir == DEFAULT_INPUT_DIR:
        resolved_default_input_dir = resolved_profile.default_input_dir
    resolved_default_env_name = default_env_name
    if default_env_name is None:
        resolved_default_env_name = resolved_profile.env_name
    parser = argparse.ArgumentParser(
        description=resolved_description
    )
    parser.add_argument(
        "input_dir",
        type=str,
        nargs="?",
        default=resolved_default_input_dir,
        help="Input directory that contains render_rollout.npz/CSV files or a rollouts subdirectory.",
    )
    parser.add_argument(
        "--num-folders",
        "-n",
        type=int,
        default=10,
        help="Process the latest N rollout folders when input_dir contains rollouts.",
    )
    parser.add_argument(
        "--all",
        "-a",
        action="store_true",
        help="Process all rollout folders instead of only the latest N.",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Only save figures without opening interactive windows.",
    )
    parser.add_argument(
        "--threshold-x",
        type=float,
        default=resolved_profile.default_threshold_x,
        help="Minimum longitudinal distance required by the task.",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=10.0,
        help="Maximum allowed absolute roll, pitch, and sideslip angle in degrees.",
    )
    parser.add_argument(
        "--figure-width",
        type=float,
        default=resolved_profile.default_figure_width,
        help="Figure width in inches. IEEE double-column width is about 7.16 in.",
    )
    parser.add_argument(
        "--figure-height",
        type=float,
        default=resolved_profile.default_figure_height,
        help="Figure height in inches.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Raster export resolution for PNG output.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=resolved_profile.default_output_name,
        help="Base name of the exported figure files.",
    )
    parser.add_argument(
        "--save-pdf",
        action="store_true",
        help="Also export a vector PDF copy beside the PNG.",
    )
    parser.add_argument(
        "--show-run-label",
        action="store_true",
        help="Add the folder name as a compact figure header.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory where all exported images are saved. Defaults to <input_dir>/analysis_outputs.",
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default=resolved_default_env_name,
        help="Optional environment hint for trajectory-deviation metrics.",
    )
    parser.add_argument(
        "--extra-input-dirs",
        nargs="*",
        default=[],
        help="Additional rollout data folders to merge with input_dir into one aggregated figure.",
    )
    parser.add_argument(
        "--model-label",
        type=str,
        default=None,
        help="Override the display/export label used for a merged model figure.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only export the cross-model summary figure and skip per-model figures.",
    )
    parser.add_argument(
        "--skip-summary-table",
        action="store_true",
        help="Skip exporting the standalone summary-table figure.",
    )
    parser.add_argument(
        "--max-plot-trajectories",
        type=int,
        default=600,
        help=(
            "Maximum number of individual trajectories to draw per angle panel. "
            "Use 0 to draw every trajectory; statistics still use all loaded trajectories."
        ),
    )
    parser.add_argument(
        "--summary-only-threshold",
        type=int,
        default=DEFAULT_SUMMARY_ONLY_TRAJECTORY_THRESHOLD,
        help=(
            "Automatically skip per-trajectory figures when a loaded source has more than this many "
            "trajectories. Use 0 to disable."
        ),
    )
    parser.add_argument(
        "--split-panels",
        dest="split_panels",
        action="store_true",
        default=False,
        help="Export trajectory analysis as separate panel images inside a subdirectory.",
    )
    parser.add_argument(
        "--no-split-panels",
        dest="split_panels",
        action="store_false",
        help="Export the trajectory analysis as one combined figure.",
    )
    parser.add_argument(
        "--hide-panel-titles",
        action="store_true",
        help="Hide panel titles on exported figures.",
    )
    if resolved_profile.parser_customizer is not None:
        resolved_profile.parser_customizer(parser)
    return parser


TIMESTAMP_SEED_PATTERN = re.compile(r"^(?P<base>.+?)_(?P<date>\d{4,8})_(?P<time>\d{6})_(?P<seed>\d+)$")


def strip_seed_run_suffix(name: str) -> str:
    match = TIMESTAMP_SEED_PATTERN.match(name)
    if match:
        return str(match.group("base"))
    return name


def make_model_source(
    label: str,
    model_dirs: Sequence[str],
    primary_dir: Optional[str] = None,
    env_name: Optional[str] = None,
) -> Dict[str, object]:
    normalized_dirs = [
        os.path.abspath(os.path.normpath(path)) for path in model_dirs if folder_contains_data(path)
    ]
    if not normalized_dirs:
        raise ValueError(f"No rollout data folders found for model source: {label}")
    unique_dirs = list(dict.fromkeys(normalized_dirs))
    resolved_primary = os.path.abspath(os.path.normpath(primary_dir or unique_dirs[0]))
    resolved_env_name = canonicalize_env_name(
        env_name or infer_env_name_from_strings(label, *unique_dirs, resolved_primary)
    )
    display_name = infer_algorithm_display_name(label, *unique_dirs, resolved_primary)
    return {
        "label": label,
        "model_dirs": unique_dirs,
        "primary_dir": resolved_primary,
        "env_name": resolved_env_name,
        "display_name": display_name,
    }


def infer_model_label(model_dir: str) -> str:
    normalized = os.path.abspath(os.path.normpath(model_dir))
    if is_rollout_leaf_dir(normalized):
        run_dir_name = os.path.basename(os.path.dirname(os.path.dirname(normalized)))
        return strip_seed_run_suffix(run_dir_name)
    return strip_seed_run_suffix(os.path.basename(normalized))


def canonicalize_env_name(env_name: Optional[str]) -> Optional[str]:
    if env_name is None:
        return None
    lowered = str(env_name).lower()
    if "singlelane" in lowered:
        return "singlelane"
    if "moose" in lowered:
        return "moose"
    if "fixed_circle" in lowered:
        return "fixed_circle_iwd"
    return lowered


def infer_env_name_from_strings(*values: object) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        lowered = str(value).lower()
        for env_name in SUPPORTED_ENV_NAMES:
            if env_name in lowered:
                return canonicalize_env_name(env_name)
        if "fixed_circle" in lowered:
            return "fixed_circle_iwd"
    return None


def get_singlelane_target_y(x: np.ndarray) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    y = np.zeros_like(x_array, dtype=float)
    x1_end = SINGLELANE_LANE_A_LENGTH
    x2_end = x1_end + SINGLELANE_CURVE_AB_LENGTH

    curve_mask = (x_array > x1_end) & (x_array <= x2_end)
    if np.any(curve_mask):
        progress = np.clip((x_array[curve_mask] - x1_end) / SINGLELANE_CURVE_AB_LENGTH, 0.0, 1.0)
        y[curve_mask] = SINGLELANE_LANE_B_OFFSET * (1.0 - np.cos(np.pi * progress)) / 2.0

    final_mask = x_array > x2_end
    y[final_mask] = SINGLELANE_LANE_B_OFFSET
    return y


def get_moose_target_y(x: np.ndarray) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    y = np.zeros_like(x_array, dtype=float)
    x1_end = MOOSE_LANE_A_LENGTH
    x2_end = x1_end + MOOSE_CURVE_AB_LENGTH
    x3_end = x2_end + MOOSE_LANE_B_LENGTH
    x4_end = x3_end + MOOSE_CURVE_BC_LENGTH

    mask_ab = (x_array > x1_end) & (x_array <= x2_end)
    if np.any(mask_ab):
        progress = np.clip((x_array[mask_ab] - x1_end) / MOOSE_CURVE_AB_LENGTH, 0.0, 1.0)
        y[mask_ab] = MOOSE_LANE_B_OFFSET * (1.0 - np.cos(np.pi * progress)) / 2.0

    mask_b = (x_array > x2_end) & (x_array <= x3_end)
    y[mask_b] = MOOSE_LANE_B_OFFSET

    mask_bc = (x_array > x3_end) & (x_array <= x4_end)
    if np.any(mask_bc):
        progress = np.clip((x_array[mask_bc] - x3_end) / MOOSE_CURVE_BC_LENGTH, 0.0, 1.0)
        y[mask_bc] = MOOSE_LANE_B_OFFSET * (1.0 + np.cos(np.pi * progress)) / 2.0

    return y


def infer_algorithm_display_name(*values: object) -> str:
    parts = [str(value) for value in values if value is not None]
    combined = " ".join(parts).lower()
    extra_hints = " ".join(load_algorithm_hint_text(value) for value in values if value is not None).lower()
    combined_with_hints = f"{combined} {extra_hints}".strip()

    is_rarl = bool(
        "rarl_" in combined_with_hints
        or "rarl_continuous" in combined_with_hints
        or re.search(r"(?:^|[_/\\\\\-\s])rarl(?:$|[_/\\\\\-\s])", combined_with_hints)
    )
    has_riskq = bool(RISKQ_TOKEN_PATTERN.search(combined_with_hints)) or any(
        pattern.search(combined_with_hints) for pattern in RISKQ_TRUE_PATTERNS
    )
    has_sampled_adv_scale = bool(SAMPLED_ADV_SCALE_TOKEN_PATTERN.search(combined_with_hints)) or any(
        pattern.search(combined_with_hints) for pattern in SAMPLED_ADV_SCALE_TRUE_PATTERNS
    )

    if is_rarl and has_riskq:
        return "riskq-RARL"
    if is_rarl and has_sampled_adv_scale:
        return "K-RARL"
    if is_rarl:
        return "RARL"
    if "a2c_continuous" in combined or "a2c_basic_continuous" in combined or "ppo" in combined:
        has_rpiq = bool(RPIQ_ENABLED_PATTERN.search(combined))
        has_disturbed = bool(DISTURBED_PATTERN.search(combined))
        if has_rpiq and has_disturbed:
            return "rpiq-PPO-distured"
        if has_rpiq:
            return "rpiq-PPO"
        if has_disturbed:
            return "PPO-distured"
        return "PPO"
    return parts[0] if parts else "Model"


def load_algorithm_hint_text(value: object) -> str:
    path = safe_path_from_value(value)
    if path is None:
        return ""

    if path.is_file():
        if path.name.lower() == "run.log":
            return read_run_log_hint_text(path)
        if path.suffix.lower() == ".pth":
            return read_checkpoint_hint_text(path)
        return ""

    if path.is_dir():
        run_dir = resolve_run_dir_from_path(path)
        if run_dir is None:
            return ""
        return read_run_log_hint_text(run_dir / "run.log")

    return ""


def safe_path_from_value(value: object) -> Optional[Path]:
    try:
        raw = str(value)
    except Exception:
        return None
    if not raw:
        return None
    if not any(sep in raw for sep in (os.sep, "/", "\\")):
        return None
    try:
        return Path(raw)
    except Exception:
        return None


def resolve_run_dir_from_path(path_value: Path) -> Optional[Path]:
    path = Path(path_value)
    if path.is_file():
        if path.name.lower() == "run.log":
            return path.parent
        if path.parent.name == "nn":
            return path.parent.parent
        return None

    normalized_parts = [part.lower() for part in path.parts]
    if "rollouts" in normalized_parts:
        rollouts_index = normalized_parts.index("rollouts")
        if rollouts_index >= 1:
            return Path(*path.parts[:rollouts_index])
    if (path / "run.log").exists():
        return path
    if (path / "nn").exists() or (path / "rollouts").exists():
        return path
    return None


def read_run_log_hint_text(run_log_path: Path) -> str:
    cache_key = str(run_log_path.resolve()) if run_log_path.exists() else str(run_log_path)
    if cache_key in RUN_LOG_HINT_CACHE:
        return RUN_LOG_HINT_CACHE[cache_key]

    text = ""
    if run_log_path.exists():
        try:
            text = run_log_path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            text = ""
    RUN_LOG_HINT_CACHE[cache_key] = text
    return text


def read_checkpoint_hint_text(checkpoint_path: Path) -> str:
    cache_key = str(checkpoint_path.resolve()) if checkpoint_path.exists() else str(checkpoint_path)
    if cache_key in CHECKPOINT_HINT_CACHE:
        return CHECKPOINT_HINT_CACHE[cache_key]

    text = ""
    if checkpoint_path.exists():
        try:
            with zipfile.ZipFile(checkpoint_path, "r") as archive:
                pkl_name = next((name for name in archive.namelist() if name.endswith("data.pkl")), None)
                if pkl_name is not None:
                    text = archive.read(pkl_name).decode("latin1", errors="ignore")
        except Exception:
            text = ""
    CHECKPOINT_HINT_CACHE[cache_key] = text
    return text


def resolve_model_sources(
    input_dir: str,
    extra_input_dirs: Sequence[str],
    model_label: Optional[str],
    num_folders: int,
    process_all: bool,
) -> List[Dict[str, object]]:
    normalized_input = os.path.abspath(os.path.normpath(input_dir))
    normalized_extra = [
        os.path.abspath(os.path.normpath(path)) for path in extra_input_dirs
    ]

    if normalized_extra:
        merged_dirs = [normalized_input, *normalized_extra]
        label = model_label or infer_model_label(normalized_input)
        return [make_model_source(label=label, model_dirs=merged_dirs, primary_dir=normalized_input)]

    model_dirs = discover_model_dirs(input_dir=normalized_input, num_folders=num_folders, process_all=process_all)
    resolved_sources: List[Dict[str, object]] = []
    for model_dir in model_dirs:
        resolved_sources.append(
            make_model_source(
                label=model_label or infer_model_label(model_dir),
                model_dirs=[model_dir],
                primary_dir=model_dir,
            )
        )
    return resolved_sources


def discover_model_dirs(input_dir: str, num_folders: int, process_all: bool) -> List[str]:
    direct_files = discover_rollout_data_files(input_dir)
    if direct_files:
        print(f"Found {len(direct_files)} rollout data file(s) directly inside: {input_dir}")
        return [input_dir]

    rollouts_dir = os.path.join(input_dir, "rollouts")
    if not os.path.exists(rollouts_dir):
        raise FileNotFoundError(f"No rollout data files found under: {input_dir}")

    rollout_subdirs = [
        path for path in glob.glob(os.path.join(rollouts_dir, "*")) if os.path.isdir(path)
    ]
    if not rollout_subdirs:
        raise FileNotFoundError(f"No rollout folders found under: {rollouts_dir}")

    rollout_subdirs.sort(key=os.path.getmtime, reverse=True)
    selected_dirs = rollout_subdirs if process_all else rollout_subdirs[:num_folders]

    model_dirs = [
        rollout_dir
        for rollout_dir in selected_dirs
        if folder_contains_data(rollout_dir)
    ]
    if not model_dirs:
        raise FileNotFoundError(f"No rollout data files found in the selected rollout folders under: {rollouts_dir}")

    print(f"Selected {len(model_dirs)} rollout folders:")
    for index, path in enumerate(model_dirs, start=1):
        print(f"  {index}. {os.path.basename(path)}")
    return model_dirs


def folder_contains_csv(dir_path: str) -> bool:
    return bool(glob.glob(os.path.join(dir_path, "*.csv")))


def discover_rollout_data_files(dir_path: str) -> List[str]:
    npz_files = []
    for name in RENDER_NPZ_NAMES:
        candidate = os.path.join(dir_path, name)
        if os.path.isfile(candidate):
            npz_files.append(candidate)
    if npz_files:
        return sorted(npz_files)
    csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
    return csv_files


def folder_contains_data(dir_path: str) -> bool:
    return bool(discover_rollout_data_files(dir_path))


def is_rollout_leaf_dir(dir_path: str) -> bool:
    normalized = os.path.abspath(os.path.normpath(dir_path))
    return folder_contains_data(normalized) and os.path.basename(os.path.dirname(normalized)) == "rollouts"


def resolve_output_dir(input_dir: str, output_dir: Optional[str]) -> str:
    if output_dir:
        resolved = os.path.abspath(output_dir)
    elif folder_contains_data(input_dir):
        resolved = os.path.abspath(input_dir)
    else:
        resolved = os.path.abspath(os.path.join(input_dir, "analysis_outputs"))
    os.makedirs(resolved, exist_ok=True)
    return resolved


def resolve_summary_output_dir(input_dir: str, output_dir: Optional[str], default_output_dir: str) -> str:
    if output_dir:
        resolved = os.path.abspath(output_dir)
    elif is_rollout_leaf_dir(input_dir):
        resolved = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(input_dir)), "analysis_outputs"))
    else:
        resolved = default_output_dir
    os.makedirs(resolved, exist_ok=True)
    return resolved


def make_export_stem(model_label: str, output_name: str) -> str:
    return f"{model_label}_{output_name}"


def discover_summary_dirs(input_dir: str, processed_dirs: Sequence[str]) -> List[str]:
    if is_rollout_leaf_dir(input_dir):
        rollouts_dir = os.path.dirname(os.path.abspath(input_dir))
        sibling_dirs = [
            path for path in glob.glob(os.path.join(rollouts_dir, "*"))
            if os.path.isdir(path) and folder_contains_data(path)
        ]
        sibling_dirs.sort(key=os.path.getmtime)
        if sibling_dirs:
            print(f"Summary scope: {len(sibling_dirs)} rollout folders under {rollouts_dir}")
            return sibling_dirs
    return list(processed_dirs)


def load_csv(csv_file: str) -> Optional[Dict[str, np.ndarray]]:
    try:
        df = pd.read_csv(csv_file, header=None)
    except Exception as exc:
        print(f"  Skip {os.path.basename(csv_file)}: failed to read CSV ({exc})")
        return None

    num_cols = len(df.columns)
    if num_cols >= len(COL_NAMES_38):
        df.columns = COL_NAMES_38[:num_cols] + [
            f"col_{i}" for i in range(len(COL_NAMES_38), num_cols)
        ]
    elif num_cols == 32:
        df.columns = COL_NAMES_32
    elif num_cols >= 21:
        df.columns = COL_NAMES_32[:num_cols] + [
            f"col_{i}" for i in range(len(COL_NAMES_32), num_cols)
        ]
    else:
        print(f"  Skip {os.path.basename(csv_file)}: only {num_cols} columns")
        return None

    required_columns = {"x", "phi", "theta"}
    if not required_columns.issubset(df.columns):
        print(f"  Skip {os.path.basename(csv_file)}: missing required columns")
        return None

    beta = df["beta"].to_numpy() if "beta" in df.columns else np.zeros(len(df))
    return {
        "file": csv_file,
        "time": np.arange(len(df), dtype=float) * DEFAULT_ROLLOUT_DT,
        "x": df["x"].to_numpy(),
        "y": df["y"].to_numpy() if "y" in df.columns else np.zeros(len(df)),
        "phi": np.degrees(df["phi"].to_numpy()),
        "theta": np.degrees(df["theta"].to_numpy()),
        "beta": np.degrees(beta),
    }


def load_npz(npz_file: str) -> List[Dict[str, np.ndarray]]:
    try:
        with np.load(npz_file, allow_pickle=False) as archive:
            if "data" not in archive.files:
                print(f"  Skip {os.path.basename(npz_file)}: missing 'data' array")
                return []
            data = np.asarray(archive["data"])
            if "columns" in archive.files:
                raw_columns = archive["columns"]
                columns = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in raw_columns
                ]
            else:
                columns = ["x", "y", "phi", "theta", "beta"]
            if "axis_order" in archive.files:
                axis_order = [
                    item.decode("utf-8") if isinstance(item, bytes) else str(item)
                    for item in archive["axis_order"]
                ]
            else:
                axis_order = ["env", "time", "column"]
            precomputed_metrics = {
                key: np.asarray(archive[key], dtype=float)
                for key in (
                    "max_phi",
                    "max_theta",
                    "max_beta",
                    "rms_phi",
                    "rms_theta",
                    "rms_beta",
                    "final_x",
                )
                if key in archive.files
            }
            precomputed_angle_stats = None
            if {"angle_stat_keys", "angle_step", "angle_time", "angle_mean", "angle_ci"}.issubset(archive.files):
                raw_angle_keys = archive["angle_stat_keys"]
                precomputed_angle_stats = {
                    "keys": [
                        item.decode("utf-8") if isinstance(item, bytes) else str(item)
                        for item in raw_angle_keys
                    ],
                    "step": np.asarray(archive["angle_step"], dtype=float),
                    "time": np.asarray(archive["angle_time"], dtype=float),
                    "mean": np.asarray(archive["angle_mean"], dtype=float),
                    "ci": np.asarray(archive["angle_ci"], dtype=float),
                    "count": (
                        np.asarray(archive["angle_count"], dtype=float)
                        if "angle_count" in archive.files
                        else np.full(np.asarray(archive["angle_mean"]).shape[0], data.shape[0], dtype=float)
                    ),
                }
                if {"angle_x", "angle_x_mean", "angle_x_ci"}.issubset(archive.files):
                    precomputed_angle_stats["x"] = np.asarray(archive["angle_x"], dtype=float)
                    precomputed_angle_stats["x_mean"] = np.asarray(archive["angle_x_mean"], dtype=float)
                    precomputed_angle_stats["x_ci"] = np.asarray(archive["angle_x_ci"], dtype=float)
                    precomputed_angle_stats["x_count"] = (
                        np.asarray(archive["angle_x_count"], dtype=float)
                        if "angle_x_count" in archive.files
                        else np.full(np.asarray(archive["angle_x_mean"]).shape[:2], data.shape[0], dtype=float)
                    )
    except Exception as exc:
        print(f"  Skip {os.path.basename(npz_file)}: failed to read NPZ ({exc})")
        return []

    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    if data.ndim != 3:
        print(f"  Skip {os.path.basename(npz_file)}: expected 3D data, got shape {data.shape}")
        return []

    if len(axis_order) >= 3 and axis_order[:3] == ["time", "env", "column"]:
        data = np.transpose(data, (1, 0, 2))

    column_index = {name: index for index, name in enumerate(columns)}
    required_columns = {"x", "phi", "theta"}
    if not required_columns.issubset(column_index):
        print(f"  Skip {os.path.basename(npz_file)}: missing required columns")
        return []

    beta_index = column_index.get("beta")
    y_index = column_index.get("y")
    loaded: List[Dict[str, np.ndarray]] = []
    for env_idx in range(data.shape[0]):
        env_data = data[env_idx]
        if env_data.shape[0] == 0:
            continue
        beta = (
            env_data[:, beta_index]
            if beta_index is not None
            else np.zeros(env_data.shape[0], dtype=env_data.dtype)
        )
        y = (
            env_data[:, y_index]
            if y_index is not None
            else np.zeros(env_data.shape[0], dtype=env_data.dtype)
        )
        item: Dict[str, object] = {
            "file": f"{npz_file}::env{env_idx}",
            "time": np.arange(env_data.shape[0], dtype=float) * DEFAULT_ROLLOUT_DT,
            "x": np.asarray(env_data[:, column_index["x"]], dtype=float),
            "y": np.asarray(y, dtype=float),
            "phi": np.degrees(np.asarray(env_data[:, column_index["phi"]], dtype=float)),
            "theta": np.degrees(np.asarray(env_data[:, column_index["theta"]], dtype=float)),
            "beta": np.degrees(np.asarray(beta, dtype=float)),
            "_npz_source": npz_file,
            "_npz_env_count": data.shape[0],
        }
        if precomputed_angle_stats is not None:
            item["_precomputed_angle_stats"] = precomputed_angle_stats
        for metric_key, values in precomputed_metrics.items():
            if env_idx < len(values):
                item[metric_key] = float(values[env_idx])
        loaded.append(item)
    return loaded


def load_rollout_data_file(data_file: str) -> List[Dict[str, np.ndarray]]:
    suffix = os.path.splitext(data_file)[1].lower()
    if suffix == ".npz":
        return load_npz(data_file)
    if suffix == ".csv":
        loaded = load_csv(data_file)
        return [loaded] if loaded is not None else []
    print(f"  Skip {os.path.basename(data_file)}: unsupported rollout data format")
    return []


def interpolate_series_at_x(
    x_series: np.ndarray,
    y_series: np.ndarray,
    target_x: np.ndarray,
) -> np.ndarray:
    finite_mask = np.isfinite(x_series) & np.isfinite(y_series)
    if not np.any(finite_mask):
        return np.full(len(target_x), np.nan)

    finite_x = x_series[finite_mask]
    finite_y = y_series[finite_mask]
    sort_order = np.argsort(finite_x)
    finite_x = finite_x[sort_order]
    finite_y = finite_y[sort_order]
    finite_x, unique_indices = np.unique(finite_x, return_index=True)
    finite_y = finite_y[unique_indices]
    if len(finite_x) == 1:
        return np.full(len(target_x), finite_y[0])
    return np.interp(target_x, finite_x, finite_y)


def clip_trajectory_to_x_range(
    data: Dict[str, np.ndarray],
    x_range: Optional[Tuple[float, float]],
) -> Optional[Dict[str, np.ndarray]]:
    if x_range is None:
        return data
    if "x" not in data or len(data["x"]) == 0:
        return None

    x_min, x_max = x_range
    if x_min > x_max:
        x_min, x_max = x_max, x_min

    x_series = np.asarray(data["x"], dtype=float)
    finite_x = x_series[np.isfinite(x_series)]
    if finite_x.size == 0:
        return None

    lower = max(float(x_min), float(np.min(finite_x)))
    upper = min(float(x_max), float(np.max(finite_x)))
    if lower > upper:
        return None

    in_range = x_series[(x_series >= lower) & (x_series <= upper) & np.isfinite(x_series)]
    target_x = np.unique(np.concatenate(([lower, upper], in_range)))
    if target_x.size == 0:
        return None

    clipped: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        if key in PRECOMPUTED_TRAJECTORY_VALUE_KEYS:
            continue
        if isinstance(value, np.ndarray) and len(value) == len(x_series):
            value_series = np.asarray(value, dtype=float)
            if key == "x":
                clipped[key] = target_x
            else:
                clipped[key] = interpolate_series_at_x(x_series, value_series, target_x)
        else:
            clipped[key] = value
    return clipped


def clip_trajectories_to_x_range(
    data_list: Sequence[Dict[str, np.ndarray]],
    x_range: Optional[Tuple[float, float]],
) -> List[Dict[str, np.ndarray]]:
    if x_range is None:
        return list(data_list)
    clipped_data: List[Dict[str, np.ndarray]] = []
    for data in data_list:
        clipped = clip_trajectory_to_x_range(data, x_range)
        if clipped is not None:
            clipped_data.append(clipped)
    return clipped_data


def clip_trajectory_to_time_range(
    data: Dict[str, np.ndarray],
    time_range: Optional[Tuple[float, float]],
) -> Optional[Dict[str, np.ndarray]]:
    if time_range is None:
        return data

    time_min, time_max = time_range
    if time_min > time_max:
        time_min, time_max = time_max, time_min

    time_series = compute_time_series(data)
    if len(time_series) == 0:
        return None

    finite_time = time_series[np.isfinite(time_series)]
    if finite_time.size == 0:
        return None

    lower = max(float(time_min), float(np.min(finite_time)))
    upper = min(float(time_max), float(np.max(finite_time)))
    if lower > upper:
        return None

    in_range = time_series[
        (time_series >= lower) & (time_series <= upper) & np.isfinite(time_series)
    ]
    target_time = np.unique(np.concatenate(([lower, upper], in_range)))
    if target_time.size == 0:
        return None

    clipped: Dict[str, object] = {}
    for key, value in data.items():
        if key in PRECOMPUTED_TRAJECTORY_VALUE_KEYS:
            continue
        if isinstance(value, np.ndarray) and len(value) == len(time_series):
            value_series = np.asarray(value, dtype=float)
            if key == "time":
                clipped[key] = target_time
            else:
                clipped[key] = interpolate_series_at_x(time_series, value_series, target_time)
        else:
            clipped[key] = value
    if "time" not in clipped:
        clipped["time"] = target_time
    return clipped


def clip_trajectories_to_time_range(
    data_list: Sequence[Dict[str, np.ndarray]],
    time_range: Optional[Tuple[float, float]],
) -> List[Dict[str, np.ndarray]]:
    if time_range is None:
        return list(data_list)
    clipped_data: List[Dict[str, np.ndarray]] = []
    for data in data_list:
        clipped = clip_trajectory_to_time_range(data, time_range)
        if clipped is not None:
            clipped_data.append(clipped)
    return clipped_data


def prepare_summary_data_for_profile(
    profile: Optional[RenderProfile],
    all_data: Sequence[Dict[str, np.ndarray]],
    threshold_x: float,
    angle_threshold: float,
    env_name: Optional[str] = None,
) -> Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]:
    resolved_profile = resolve_render_profile(profile)
    resolved_env_name = canonicalize_env_name(env_name) or canonicalize_env_name(resolved_profile.env_name)
    summary_data = clip_trajectories_to_x_range(all_data, resolved_profile.summary_x_range)
    summary_data = clip_trajectories_to_time_range(summary_data, resolved_profile.summary_time_range)
    summary_valid_data = [
        data
        for data in summary_data
        if is_successful_trajectory(
            data,
            threshold_x=threshold_x,
            angle_threshold=angle_threshold,
            env_name=resolved_env_name,
            profile=resolved_profile,
        )
    ]
    return summary_data, summary_valid_data


def compute_fixed_circle_radius_error(data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    if "x" not in data or "y" not in data or len(data["x"]) == 0 or len(data["y"]) == 0:
        return None
    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    radial_distance = np.hypot(x - FIXED_CIRCLE_CENTER_X, y - FIXED_CIRCLE_CENTER_Y)
    return np.asarray(radial_distance - FIXED_CIRCLE_RADIUS, dtype=float)


def compute_time_series(data: Dict[str, np.ndarray]) -> np.ndarray:
    if "time" in data and len(data["time"]) > 0:
        return np.asarray(data["time"], dtype=float)
    length = len(data["x"]) if "x" in data else 0
    return np.arange(length, dtype=float) * DEFAULT_ROLLOUT_DT


def get_angle_failure_thresholds(
    angle_threshold: float,
    profile: Optional[RenderProfile] = None,
) -> Dict[str, float]:
    resolved_profile = resolve_render_profile(profile)
    thresholds: Dict[str, float] = {}
    for panel in resolved_profile.angle_panels:
        if panel.failure_lower is not None or panel.failure_upper is not None:
            continue
        if not panel.show_threshold_lines:
            continue
        thresholds[panel.key] = float(
            panel.threshold_value
            if panel.threshold_value is not None
            else angle_threshold
        )
    return thresholds


def get_angle_failure_bounds(
    profile: Optional[RenderProfile] = None,
) -> Dict[str, Tuple[Optional[float], Optional[float]]]:
    resolved_profile = resolve_render_profile(profile)
    bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = {}
    for panel in resolved_profile.angle_panels:
        if panel.failure_lower is None and panel.failure_upper is None:
            continue
        bounds[panel.key] = (
            float(panel.failure_lower) if panel.failure_lower is not None else None,
            float(panel.failure_upper) if panel.failure_upper is not None else None,
        )
    return bounds


def compute_angle_failure_bound_mask(
    data: Dict[str, np.ndarray],
    key: str,
    lower: Optional[float],
    upper: Optional[float],
) -> Optional[np.ndarray]:
    if key not in data:
        return None

    values = np.asarray(data[key], dtype=float)
    if values.size == 0:
        return np.zeros(0, dtype=bool)

    finite_mask = np.isfinite(values)
    failure_mask = np.zeros(values.shape, dtype=bool)
    if lower is not None:
        failure_mask |= finite_mask & (values < float(lower))
    if upper is not None:
        failure_mask |= finite_mask & (values > float(upper))
    return failure_mask


def has_angle_panel_failure(
    data: Dict[str, np.ndarray],
    angle_threshold: float,
    profile: Optional[RenderProfile] = None,
) -> bool:
    angle_thresholds = get_angle_failure_thresholds(angle_threshold, profile=profile)
    for key, threshold in angle_thresholds.items():
        if key not in data:
            return True
        metric_key = f"max_{key}"
        if metric_key in data:
            max_value = metric_from_data(data, metric_key, key, "max_abs")
            if max_value >= float(threshold):
                return True
            continue
        values = np.asarray(data[key], dtype=float)
        if values.size == 0:
            return True
        finite_mask = np.isfinite(values)
        if np.any(finite_mask & (np.abs(values) >= float(threshold))):
            return True

    angle_failure_bounds = get_angle_failure_bounds(profile=profile)
    for key, (lower, upper) in angle_failure_bounds.items():
        mask = compute_angle_failure_bound_mask(
            data,
            key=key,
            lower=lower,
            upper=upper,
        )
        if mask is None or mask.size == 0 or bool(np.any(mask)):
            return True

    return False


def get_trajectory_error_failure_bounds(
    profile: Optional[RenderProfile] = None,
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    resolved_profile = resolve_render_profile(profile)
    signed_panel = resolve_trajectory_error_panel_config(
        resolved_profile,
        panel_config=resolved_profile.signed_trajectory_error_panel,
    )
    if signed_panel is not None and signed_panel.reference_line_values:
        reference_values = tuple(float(value) for value in signed_panel.reference_line_values)
        lower_candidates = [value for value in reference_values if value < 0.0]
        upper_candidates = [value for value in reference_values if value > 0.0]
        lower = max(lower_candidates) if lower_candidates else None
        upper = min(upper_candidates) if upper_candidates else None
        if lower is not None or upper is not None:
            return lower, upper
    error_panel = resolve_trajectory_error_panel_config(
        resolved_profile,
        panel_config=resolved_profile.trajectory_error_panel,
    )
    if error_panel is not None and error_panel.use_absolute_value:
        y_low, y_high = error_panel.ylim
        if y_low >= 0.0:
            threshold = float(y_high)
            return -threshold, threshold
    return None


def get_trajectory_error_failure_threshold(
    profile: Optional[RenderProfile] = None,
) -> Optional[float]:
    bounds = get_trajectory_error_failure_bounds(profile)
    if bounds is None:
        return None
    lower, upper = bounds
    candidates = [abs(value) for value in (lower, upper) if value is not None]
    return max(candidates) if candidates else None


def compute_trajectory_error_failure_mask(
    data: Dict[str, np.ndarray],
    env_name: Optional[str],
    profile: Optional[RenderProfile] = None,
) -> Optional[np.ndarray]:
    bounds = get_trajectory_error_failure_bounds(profile)
    if bounds is None:
        return None
    deviation = compute_trajectory_deviation_values(data, env_name=env_name)
    if deviation is None:
        return None
    lower, upper = bounds
    mask = np.zeros(len(deviation), dtype=bool)
    if lower is not None:
        mask |= np.asarray(deviation, dtype=float) <= float(lower)
    if upper is not None:
        mask |= np.asarray(deviation, dtype=float) >= float(upper)
    return mask


def resolve_success_env_name(data: Dict[str, np.ndarray], env_name: Optional[str]) -> Optional[str]:
    return canonicalize_env_name(env_name) or infer_env_name_from_strings(data.get("file", ""))


def uses_trajectory_error_only_failure(
    data: Dict[str, np.ndarray],
    env_name: Optional[str],
    profile: Optional[RenderProfile],
) -> bool:
    resolved_profile = resolve_render_profile(profile)
    return resolved_profile.failure_mode == FAILURE_MODE_TRAJECTORY_ERROR_ONLY


def compute_failure_progress_series(
    data: Dict[str, np.ndarray],
    env_name: Optional[str],
) -> np.ndarray:
    if resolve_success_env_name(data, env_name) == "fixed_circle_iwd":
        return compute_time_series(data)
    if "x" in data and len(data["x"]) > 0:
        return np.asarray(data["x"], dtype=float)
    return compute_time_series(data)


def value_at_failure_progress_index(
    data: Dict[str, np.ndarray],
    index: int,
    env_name: Optional[str],
) -> float:
    progress = compute_failure_progress_series(data, env_name)
    if index < len(progress) and np.isfinite(progress[index]):
        return float(progress[index])
    if len(progress) > 0 and np.isfinite(progress[-1]):
        return float(progress[-1])
    return float(index * DEFAULT_ROLLOUT_DT)


def truncate_xy_series_at_y_bounds(
    x_values: np.ndarray,
    y_values: np.ndarray,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=float)
    y = np.asarray(y_values, dtype=float)
    common_len = min(len(x), len(y))
    x = x[:common_len]
    y = y[:common_len]
    if common_len == 0 or (lower is None and upper is None):
        return x, y

    def crossed(value: float) -> bool:
        if not np.isfinite(value):
            return False
        return bool(
            (upper is not None and value >= upper)
            or (lower is not None and value <= lower)
        )

    def segment_crossing(
        previous_x: float,
        previous_y: float,
        current_x: float,
        current_y: float,
    ) -> Tuple[float, float]:
        candidates: List[Tuple[float, float, float]] = []
        for boundary in (lower, upper):
            if boundary is None:
                continue
            if not (
                np.isfinite(previous_y)
                and np.isfinite(current_y)
                and min(previous_y, current_y) <= boundary <= max(previous_y, current_y)
            ):
                continue
            denominator = current_y - previous_y
            fraction = 0.0 if np.isclose(denominator, 0.0) else (boundary - previous_y) / denominator
            if 0.0 <= fraction <= 1.0:
                candidates.append((float(fraction), float(boundary), float(previous_x + fraction * (current_x - previous_x))))
        if not candidates:
            clipped_y = float(np.clip(current_y, lower if lower is not None else -np.inf, upper if upper is not None else np.inf))
            return float(current_x), clipped_y
        fraction, boundary, crossing_x = min(candidates, key=lambda item: item[0])
        return crossing_x, boundary

    for index, current_y in enumerate(y):
        if not crossed(float(current_y)):
            continue
        if index == 0:
            clipped_y = float(np.clip(current_y, lower if lower is not None else -np.inf, upper if upper is not None else np.inf))
            return np.asarray([x[0]], dtype=float), np.asarray([clipped_y], dtype=float)
        crossing_x, crossing_y = segment_crossing(
            float(x[index - 1]),
            float(y[index - 1]),
            float(x[index]),
            float(y[index]),
        )
        return (
            np.concatenate([x[:index], np.asarray([crossing_x], dtype=float)]),
            np.concatenate([y[:index], np.asarray([crossing_y], dtype=float)]),
        )
    return x, y


def compute_survival_distance(
    data: Dict[str, np.ndarray],
    angle_threshold: float,
    env_name: Optional[str] = None,
    profile: Optional[RenderProfile] = None,
) -> float:
    if uses_trajectory_error_only_failure(data, env_name, profile):
        trajectory_error_mask = compute_trajectory_error_failure_mask(
            data,
            env_name=env_name,
            profile=profile,
        )
        if trajectory_error_mask is not None:
            exceed_indices = np.where(trajectory_error_mask)[0]
            if len(exceed_indices) > 0:
                return value_at_failure_progress_index(
                    data,
                    int(exceed_indices[0]),
                    env_name,
                )
            progress = compute_failure_progress_series(data, env_name)
            if len(progress) > 0 and np.isfinite(progress[-1]):
                return float(progress[-1])
        if "x" in data and len(data["x"]) > 0:
            return float(data["x"][-1])
        return 0.0

    exceed_masks = []
    angle_thresholds = get_angle_failure_thresholds(angle_threshold, profile=profile)
    for key, threshold in angle_thresholds.items():
        if key not in data:
            continue
        exceed_masks.append(np.abs(np.asarray(data[key], dtype=float)) >= threshold)
    angle_failure_bounds = get_angle_failure_bounds(profile=profile)
    for key, (lower, upper) in angle_failure_bounds.items():
        mask = compute_angle_failure_bound_mask(
            data,
            key=key,
            lower=lower,
            upper=upper,
        )
        if mask is not None:
            exceed_masks.append(mask)

    trajectory_error_mask = compute_trajectory_error_failure_mask(
        data,
        env_name=env_name,
        profile=profile,
    )
    if trajectory_error_mask is not None:
        exceed_masks.append(trajectory_error_mask)

    if exceed_masks:
        min_len = min(len(mask) for mask in exceed_masks)
        exceed_mask = np.zeros(min_len, dtype=bool)
        for mask in exceed_masks:
            exceed_mask |= np.asarray(mask[:min_len], dtype=bool)
        exceed_indices = np.where(exceed_mask)[0]
    else:
        exceed_indices = np.array([], dtype=int)
    if len(exceed_indices) > 0:
        return value_at_failure_progress_index(
            data,
            int(exceed_indices[0]),
            env_name,
        )
    progress = compute_failure_progress_series(data, env_name)
    if len(progress) > 0 and np.isfinite(progress[-1]):
        return float(progress[-1])
    if "x" in data and len(data["x"]) > 0:
        return float(data["x"][-1])
    return 0.0


def is_successful_trajectory(
    data: Dict[str, np.ndarray],
    threshold_x: float,
    angle_threshold: float,
    env_name: Optional[str] = None,
    profile: Optional[RenderProfile] = None,
) -> bool:
    if uses_trajectory_error_only_failure(data, env_name, profile):
        trajectory_error_mask = compute_trajectory_error_failure_mask(
            data,
            env_name=env_name,
            profile=profile,
        )
        if trajectory_error_mask is None:
            return False
        return not bool(np.any(trajectory_error_mask))

    angle_thresholds = get_angle_failure_thresholds(angle_threshold, profile=profile)
    for key, threshold in angle_thresholds.items():
        if key not in data:
            return False
        metric_key = f"max_{key}"
        max_value = metric_from_data(data, metric_key, key, "max_abs")
        if max_value >= threshold:
            return False
    angle_failure_bounds = get_angle_failure_bounds(profile=profile)
    for key, (lower, upper) in angle_failure_bounds.items():
        mask = compute_angle_failure_bound_mask(
            data,
            key=key,
            lower=lower,
            upper=upper,
        )
        if mask is None or mask.size == 0 or bool(np.any(mask)):
            return False

    trajectory_error_mask = compute_trajectory_error_failure_mask(
        data,
        env_name=env_name,
        profile=profile,
    )
    if trajectory_error_mask is not None:
        if trajectory_error_mask.size == 0 or bool(np.any(trajectory_error_mask)):
            return False

    if resolve_success_env_name(data, env_name) == "fixed_circle_iwd":
        return True

    final_x = float(data.get("final_x", data["x"][-1]))
    return final_x >= threshold_x


def is_statistic_successful_trajectory(
    data: Dict[str, np.ndarray],
    threshold_x: float,
    angle_threshold: float,
    env_name: Optional[str] = None,
    profile: Optional[RenderProfile] = None,
) -> bool:
    return is_successful_trajectory(
        data,
        threshold_x=threshold_x,
        angle_threshold=angle_threshold,
        env_name=env_name,
        profile=profile,
    )


def align_series(data_list: Sequence[Dict[str, np.ndarray]], key: str, max_len: int) -> np.ndarray:
    if not data_list:
        return np.full((0, max_len), np.nan)
    aligned = np.full((len(data_list), max_len), np.nan)
    for index, data in enumerate(data_list):
        series = data[key]
        aligned[index, : len(series)] = series
    return aligned


def align_series_by_x(
    data_list: Sequence[Dict[str, np.ndarray]],
    y_key: str,
    x_key: str,
    x_values: np.ndarray,
) -> np.ndarray:
    if not data_list:
        return np.full((0, len(x_values)), np.nan)

    aligned = np.full((len(data_list), len(x_values)), np.nan)
    for index, data in enumerate(data_list):
        if x_key not in data:
            continue

        x_series = np.asarray(data[x_key], dtype=float)
        y_series = np.asarray(data[y_key], dtype=float)
        finite_mask = np.isfinite(x_series) & np.isfinite(y_series)
        if np.sum(finite_mask) < 2:
            continue

        x_series = x_series[finite_mask]
        y_series = y_series[finite_mask]
        sort_order = np.argsort(x_series)
        x_series = x_series[sort_order]
        y_series = y_series[sort_order]
        x_series, unique_indices = np.unique(x_series, return_index=True)
        y_series = y_series[unique_indices]
        if len(x_series) < 2:
            continue

        in_range = (x_values >= x_series[0]) & (x_values <= x_series[-1])
        aligned[index, in_range] = np.interp(x_values[in_range], x_series, y_series)
    return aligned


def compute_mean_and_ci(aligned: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if aligned.size == 0:
        return np.zeros(1), np.zeros(1)

    counts = np.sum(~np.isnan(aligned), axis=0)
    valid_counts = counts > 0
    sums = np.nansum(aligned, axis=0)
    mean = np.divide(sums, counts, out=np.full(aligned.shape[1], np.nan, dtype=float), where=valid_counts)
    centered = aligned - mean
    squared = np.square(centered)
    variance = np.divide(
        np.nansum(squared, axis=0),
        counts,
        out=np.full(aligned.shape[1], np.nan, dtype=float),
        where=valid_counts,
    )
    std = np.sqrt(variance)
    sem = np.divide(std, np.sqrt(counts), out=np.zeros_like(std), where=counts > 0)
    ci = 1.96 * sem
    mean = np.nan_to_num(mean, nan=0.0)
    ci = np.nan_to_num(ci, nan=0.0)
    return mean, ci


def get_precomputed_angle_mean_ci(
    data_list: Sequence[Dict[str, np.ndarray]],
    key: str,
    panel_config: AnglePanelConfig,
    truncation_lower: Optional[float],
    truncation_upper: Optional[float],
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if not data_list:
        return None

    first = data_list[0]
    stats = first.get("_precomputed_angle_stats")
    if not isinstance(stats, dict):
        return None

    source = first.get("_npz_source")
    env_count = first.get("_npz_env_count")
    if source is None or env_count is None or len(data_list) != int(env_count):
        return None
    if any(data.get("_npz_source") != source for data in data_list):
        return None

    keys = list(stats.get("keys", []))
    if key not in keys:
        return None
    key_index = keys.index(key)

    mean_key = "mean"
    ci_key = "ci"
    if panel_config.x_key == "x":
        if not {"x", "x_mean", "x_ci"}.issubset(stats.keys()):
            return None
        x_values = np.asarray(stats.get("x"), dtype=float)
        mean_key = "x_mean"
        ci_key = "x_ci"
    elif panel_config.x_key == "time":
        x_values = np.asarray(stats.get("time"), dtype=float)
    elif panel_config.x_key is None:
        x_values = np.asarray(stats.get("step"), dtype=float)
    else:
        return None

    mean_values = np.asarray(stats.get(mean_key), dtype=float)
    ci_values = np.asarray(stats.get(ci_key), dtype=float)
    if mean_values.ndim != 2 or ci_values.ndim != 2:
        return None
    if key_index >= mean_values.shape[1] or key_index >= ci_values.shape[1]:
        return None
    if len(x_values) != mean_values.shape[0] or len(x_values) != ci_values.shape[0]:
        return None

    mean = mean_values[:, key_index]
    ci = ci_values[:, key_index]
    valid_mask = np.isfinite(x_values) & np.isfinite(mean) & np.isfinite(ci)
    return x_values, mean, ci, valid_mask


def select_plot_trajectories(
    data_list: Sequence[Dict[str, np.ndarray]],
    max_plot_trajectories: int,
) -> Sequence[Dict[str, np.ndarray]]:
    if max_plot_trajectories <= 0 or len(data_list) <= max_plot_trajectories:
        return data_list
    indices = np.linspace(0, len(data_list) - 1, max_plot_trajectories, dtype=int)
    indices = np.unique(indices)
    return [data_list[int(index)] for index in indices]


def set_common_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(True, which="major", alpha=0.85)
    ax.grid(True, which="minor", alpha=0.35)
    ax.tick_params(direction="out", length=3.5, width=0.8, colors=TEXT_MUTED)
    ax.tick_params(which="minor", length=2.0, width=0.5)
    for spine in ax.spines.values():
        spine.set_color(SPINE_COLOR)
        spine.set_linewidth(0.9)


def round_axis_limit(value: float, step: float = 2.0) -> float:
    return max(step, step * np.ceil(value / step))


def set_panel_titles_visible(visible: bool) -> None:
    global HIDE_PANEL_TITLES
    HIDE_PANEL_TITLES = not bool(visible)


def add_panel_title(ax: plt.Axes, panel_label: str, title: str) -> None:
    if HIDE_PANEL_TITLES:
        ax.set_title("")
        return
    label_text = f"{panel_label} {title}".strip()
    ax.set_title(label_text, loc="left", fontweight="bold", pad=4)


def format_metric_summary(mean: float, std: float) -> str:
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def compute_trajectory_deviation_values(
    data: Dict[str, np.ndarray],
    env_name: Optional[str],
) -> Optional[np.ndarray]:
    resolved_env_name = canonicalize_env_name(env_name) or infer_env_name_from_strings(data.get("file", ""))
    if "x" not in data or "y" not in data or len(data["x"]) == 0 or len(data["y"]) == 0:
        return None

    x = np.asarray(data["x"], dtype=float)
    y = np.asarray(data["y"], dtype=float)
    if resolved_env_name == "singlelane":
        deviation = y - get_singlelane_target_y(x)
    elif resolved_env_name == "moose":
        deviation = y - get_moose_target_y(x)
    elif resolved_env_name == "fixed_circle_iwd":
        radial_distance = np.hypot(x - FIXED_CIRCLE_CENTER_X, y - FIXED_CIRCLE_CENTER_Y)
        deviation = radial_distance - FIXED_CIRCLE_RADIUS
    else:
        return None

    return np.asarray(deviation, dtype=float)


def resolve_trajectory_error_panel_config(
    profile: Optional[RenderProfile] = None,
    panel_config: Optional[TrajectoryErrorPanelConfig] = None,
) -> Optional[TrajectoryErrorPanelConfig]:
    resolved_profile = resolve_render_profile(profile)
    resolved_panel_config = (
        resolved_profile.trajectory_error_panel if panel_config is None else panel_config
    )
    if resolved_panel_config is None:
        return None

    axis_source = resolved_profile.angle_panels[0] if resolved_profile.angle_panels else None
    inherited_xlabel = axis_source.xlabel if axis_source is not None else "Time step"
    inherited_x_key = axis_source.x_key if axis_source is not None else None
    inherited_xlim = axis_source.xlim if axis_source is not None else None
    return TrajectoryErrorPanelConfig(
        stem=resolved_panel_config.stem,
        panel_label=resolved_panel_config.panel_label,
        title=resolved_panel_config.title,
        xlabel=resolved_panel_config.xlabel or inherited_xlabel,
        ylabel=resolved_panel_config.ylabel,
        axis_postprocessor=resolved_panel_config.axis_postprocessor,
        x_key=resolved_panel_config.x_key if resolved_panel_config.x_key is not None else inherited_x_key,
        xlim=resolved_panel_config.xlim if resolved_panel_config.xlim is not None else inherited_xlim,
        ylim=resolved_panel_config.ylim,
        show_legend=resolved_panel_config.show_legend,
        use_absolute_value=resolved_panel_config.use_absolute_value,
        reference_line_values=resolved_panel_config.reference_line_values,
    )


def compute_trajectory_deviation_rmse(
    data: Dict[str, np.ndarray],
    env_name: Optional[str],
) -> Optional[float]:
    deviation = compute_trajectory_deviation_values(data, env_name=env_name)
    if deviation is None:
        return None
    return float(np.sqrt(np.mean(np.square(deviation))))


def compute_trajectory_deviation_max(
    data: Dict[str, np.ndarray],
    env_name: Optional[str],
) -> Optional[float]:
    deviation = compute_trajectory_deviation_values(data, env_name=env_name)
    if deviation is None:
        return None
    return float(np.max(np.abs(deviation)))


def compute_metric_summary(
    all_data: Sequence[Dict[str, np.ndarray]],
    env_name: Optional[str] = None,
    metric_definitions: Optional[Sequence[MetricDefinition]] = None,
) -> List[Dict[str, object]]:
    resolved_metric_definitions = metric_definitions or METRIC_DEFINITIONS
    summaries: List[Dict[str, object]] = []
    for metric_key, label, metric_fn in resolved_metric_definitions:
        values = np.asarray([metric_fn(data) for data in all_data], dtype=float)
        finite_values = values[np.isfinite(values)]
        if len(finite_values):
            mean: Optional[float] = float(np.mean(finite_values))
            std: Optional[float] = float(np.std(finite_values))
            formatted = format_metric_summary(mean, std)
        else:
            mean = None
            std = None
            formatted = "N/A"
        summaries.append(
            {
                "key": metric_key,
                "label": label,
                "mean": mean,
                "std": std,
                "count": int(len(finite_values)),
                "formatted": formatted,
            }
        )

    deviation_values = [
        value
        for value in (
            compute_trajectory_deviation_rmse(data, env_name=env_name) for data in all_data
        )
        if value is not None
    ]
    if deviation_values:
        values = np.asarray(deviation_values, dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        summaries.append(
            {
                "key": TRAJECTORY_DEVIATION_METRIC_KEY,
                "label": r"$\mathrm{RMSE}(e_{\mathrm{traj}})$",
                "mean": mean,
                "std": std,
                "formatted": format_metric_summary(mean, std),
            }
        )

    deviation_max_values = [
        value
        for value in (
            compute_trajectory_deviation_max(data, env_name=env_name) for data in all_data
        )
        if value is not None
    ]
    if deviation_max_values:
        values = np.asarray(deviation_max_values, dtype=float)
        mean = float(np.mean(values))
        std = float(np.std(values))
        summaries.append(
            {
                "key": TRAJECTORY_DEVIATION_MAX_METRIC_KEY,
                "label": r"$\max(|e_{\mathrm{traj}}|)$",
                "mean": mean,
                "std": std,
                "formatted": format_metric_summary(mean, std),
            }
        )
    return summaries


def summarize_angle_metrics(all_data: Sequence[Dict[str, np.ndarray]]) -> List[List[str]]:
    return [
        [str(summary["label"]), str(summary["formatted"])]
        for summary in compute_metric_summary(all_data)
    ]


def build_folder_summary(
    model_label: str,
    all_data: Sequence[Dict[str, np.ndarray]],
    valid_data: Sequence[Dict[str, np.ndarray]],
    threshold_x: float,
    angle_threshold: float,
    env_name: Optional[str] = None,
    display_name: Optional[str] = None,
    algorithm_name: Optional[str] = None,
    metric_definitions: Optional[Sequence[MetricDefinition]] = None,
    profile: Optional[RenderProfile] = None,
) -> Dict[str, object]:
    resolved_env_name = canonicalize_env_name(env_name)
    resolved_profile = resolve_render_profile(profile)
    success_flags = [
        is_successful_trajectory(
            data,
            threshold_x=threshold_x,
            angle_threshold=angle_threshold,
            env_name=resolved_env_name,
            profile=resolved_profile,
        )
        for data in all_data
    ]
    metric_data = select_statistic_data_for_profile(
        all_data,
        valid_data,
        resolved_profile,
        threshold_x=threshold_x,
        angle_threshold=angle_threshold,
        env_name=resolved_env_name,
    )
    metric_summaries = compute_metric_summary(
        metric_data,
        env_name=resolved_env_name,
        metric_definitions=metric_definitions,
    )
    summary: Dict[str, object] = {
        "folder_name": model_label,
        "display_name": display_name or infer_algorithm_display_name(model_label),
        "env_name": resolved_env_name,
        "trajectory_count": len(all_data),
        "success_count": int(np.sum(success_flags)),
        "success_rate": (float(np.sum(success_flags)) / len(all_data)) if all_data else 0.0,
    }
    for metric in metric_summaries:
        summary[str(metric["key"])] = metric
    summary["survival_mean"] = float(
        np.mean(
            [
                compute_survival_distance(
                    data,
                    angle_threshold=angle_threshold,
                    env_name=resolved_env_name,
                    profile=resolved_profile,
                )
                for data in all_data
            ]
        )
    ) if all_data else 0.0
    if algorithm_name is not None:
        summary["algorithm_name"] = algorithm_name
    summary["success_flags"] = success_flags
    return summary


def load_folder_data(
    model_dirs: Sequence[str],
    threshold_x: float,
    angle_threshold: float,
    env_name: Optional[str] = None,
    profile: Optional[RenderProfile] = None,
    verbose: bool = True,
) -> Optional[Tuple[List[Dict[str, np.ndarray]], List[Dict[str, np.ndarray]]]]:
    data_files: List[str] = []
    normalized_dirs = [os.path.abspath(os.path.normpath(path)) for path in model_dirs]
    for model_dir in normalized_dirs:
        dir_data_files = discover_rollout_data_files(model_dir)
        data_files.extend(dir_data_files)
        if verbose and len(normalized_dirs) > 1:
            print(f"  {os.path.basename(model_dir)}: {len(dir_data_files)} rollout data file(s)")

    if verbose:
        print(f"Found {len(data_files)} rollout data file(s) across {len(normalized_dirs)} folder(s)")

    all_data: List[Dict[str, np.ndarray]] = []
    for data_file in data_files:
        all_data.extend(load_rollout_data_file(data_file))

    if not all_data:
        if verbose:
            joined_dirs = ", ".join(normalized_dirs)
            print(f"Warning: no valid rollout data files in {joined_dirs}")
        return None

    valid_data = [
        data
        for data in all_data
        if is_successful_trajectory(
            data,
            threshold_x=threshold_x,
            angle_threshold=angle_threshold,
            env_name=env_name,
            profile=profile,
        )
    ]
    if verbose:
        failed_count = len(all_data) - len(valid_data)
        print(
            "Trajectory summary: "
            f"total={len(all_data)}, successful={len(valid_data)}, failed={failed_count}"
        )

    return all_data, valid_data


def plot_survival_panel(
    ax: plt.Axes,
    all_data: Sequence[Dict[str, np.ndarray]],
    success_flags: Sequence[bool],
    survival_distances: Sequence[float],
    threshold_x: float,
    success_rate: float,
    panel_config: Optional[SurvivalPanelConfig] = None,
    max_plot_trajectories: int = 600,
) -> None:
    resolved_panel = panel_config or SurvivalPanelConfig()
    set_common_axis_style(ax)
    add_panel_title(ax, resolved_panel.panel_label, resolved_panel.title)
    ax.grid(True, which="major", axis="x", alpha=0.85)
    ax.grid(True, which="minor", axis="x", alpha=0.35)
    ax.grid(False, which="both", axis="y")

    if not all_data:
        ax.text(0.5, 0.5, "No valid trajectories", ha="center", va="center", transform=ax.transAxes)
        return

    sorted_indices = np.argsort(survival_distances)
    sorted_distances = np.asarray(survival_distances)[sorted_indices]
    sorted_flags = np.asarray(success_flags)[sorted_indices]
    total_count = len(sorted_distances)
    if max_plot_trajectories > 0 and total_count > max_plot_trajectories:
        display_indices = np.unique(
            np.linspace(0, total_count - 1, max_plot_trajectories, dtype=int)
        )
    else:
        display_indices = np.arange(total_count, dtype=int)
    displayed_distances = sorted_distances[display_indices]
    displayed_flags = sorted_flags[display_indices]
    y_positions = display_indices + 1
    colors = [SUCCESS_COLOR if flag else FAILURE_COLOR for flag in displayed_flags]
    bar_height = max(0.72, total_count / max(len(display_indices), 1) * 0.72)

    ax.barh(
        y_positions,
        displayed_distances,
        height=bar_height,
        color=colors,
        edgecolor="white",
        linewidth=0.0 if len(display_indices) > 500 else 0.4,
        zorder=3,
    )

    if resolved_panel.xlim is not None:
        ax.set_xlim(*resolved_panel.xlim)
    else:
        axis_max = max(threshold_x * 1.12, float(np.max(sorted_distances)) * 1.05, 10.0)
        ax.set_xlim(0.0, axis_max)
    ax.set_ylim(0.4, total_count + 0.6)
    ax.set_xlabel(resolved_panel.xlabel)
    ax.set_ylabel(resolved_panel.ylabel)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))

    if total_count <= 15:
        full_y_positions = np.arange(1, total_count + 1)
        ax.set_yticks(full_y_positions)
        ax.set_yticklabels([str(index) for index in full_y_positions])
    else:
        tick_positions = np.linspace(1, total_count, num=6, dtype=int)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels([str(index) for index in tick_positions])

    ax.invert_yaxis()
    reference_line_x = (
        resolved_panel.reference_line_x
        if resolved_panel.reference_line_x is not None
        else threshold_x
    )
    if resolved_panel.show_reference_line:
        ax.axvline(reference_line_x, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.2, zorder=4)
        if resolved_panel.show_threshold_label:
            ax.text(
                reference_line_x,
                1.01,
                f" threshold = {reference_line_x:.0f} m",
                transform=ax.get_xaxis_transform(),
                color=THRESHOLD_COLOR,
                fontsize=8,
                ha="left",
                va="bottom",
            )

    median_distance = float(np.median(sorted_distances))
    mean_distance = float(np.mean(sorted_distances))
    max_distance = float(np.max(sorted_distances))
    success_count = int(np.sum(success_flags))
    if resolved_panel.show_stats_box:
        stats_text = (
            f"Success: {success_count}/{len(success_flags)}\n"
            f"Success rate: {success_rate * 100:.1f}%\n"
            f"Mean: {mean_distance:.1f} m\n"
            f"Median: {median_distance:.1f} m\n"
            f"Max: {max_distance:.1f} m"
        )
        ax.text(
            0.98,
            0.05,
            stats_text,
            transform=ax.transAxes,
            fontsize=8,
            ha="right",
            va="bottom",
            color=TEXT_MUTED,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white", "edgecolor": "#d0d0d0", "alpha": 0.98},
        )
    if resolved_panel.axis_postprocessor is not None:
        resolved_panel.axis_postprocessor(
            ax,
            {
                "all_data": all_data,
                "success_flags": success_flags,
                "survival_distances": survival_distances,
                "threshold_x": threshold_x,
                "success_rate": success_rate,
                "panel_config": resolved_panel,
            },
        )


def plot_angle_panel(
    ax: plt.Axes,
    valid_data: Sequence[Dict[str, np.ndarray]],
    key: str,
    ylabel: str,
    panel_label: str,
    title: str,
    angle_threshold: float,
    show_legend: bool,
    panel_config: Optional[AnglePanelConfig] = None,
    max_plot_trajectories: int = 600,
    statistic_data: Optional[Sequence[Dict[str, np.ndarray]]] = None,
) -> None:
    resolved_panel = panel_config or AnglePanelConfig(
        stem=sanitize_filename_stem(key),
        key=key,
        ylabel=ylabel,
        panel_label=panel_label,
        title=title,
        show_legend=show_legend,
    )
    set_common_axis_style(ax)
    add_panel_title(ax, resolved_panel.panel_label, resolved_panel.title)
    ax.set_xlabel(resolved_panel.xlabel)
    ax.set_ylabel(resolved_panel.ylabel)
    threshold_value = (
        resolved_panel.threshold_value
        if resolved_panel.threshold_value is not None
        else angle_threshold
    )
    truncation_lower = resolved_panel.failure_lower
    truncation_upper = resolved_panel.failure_upper
    if truncation_lower is None and truncation_upper is None and resolved_panel.show_threshold_lines:
        truncation_lower = -threshold_value
        truncation_upper = threshold_value

    def prepare_panel_data(source_data: Sequence[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        prepared: List[Dict[str, np.ndarray]] = []
        for data in source_data:
            if key not in data:
                continue
            y_series = np.asarray(data[key], dtype=float)
            if resolved_panel.x_key is not None and resolved_panel.x_key in data:
                x_series = np.asarray(data[resolved_panel.x_key], dtype=float)
            else:
                x_series = np.arange(len(y_series), dtype=float)
            common_len = min(len(x_series), len(y_series))
            if common_len <= 0:
                continue
            x_series = x_series[:common_len]
            y_series = y_series[:common_len]
            if truncation_lower is not None or truncation_upper is not None:
                x_series, y_series = truncate_xy_series_at_y_bounds(
                    x_series,
                    y_series,
                    lower=truncation_lower,
                    upper=truncation_upper,
                )
            if len(y_series) == 0:
                continue
            plot_item = dict(data)
            plot_item[key] = y_series
            if resolved_panel.x_key is not None:
                plot_item[resolved_panel.x_key] = x_series
            prepared.append(plot_item)
        return prepared

    panel_data = prepare_panel_data(valid_data)
    mean_panel_data = (
        prepare_panel_data(statistic_data)
        if statistic_data is not None
        else panel_data
    )

    if not panel_data:
        if resolved_panel.xlim is not None:
            ax.set_xlim(*resolved_panel.xlim)
        else:
            ax.set_xlim(0.0, 1.0)
        if resolved_panel.ylim is not None:
            ax.set_ylim(*resolved_panel.ylim)
        else:
            ax.set_ylim(-threshold_value * 1.1, threshold_value * 1.1)
        ax.text(0.5, 0.5, "No trajectories", ha="center", va="center", transform=ax.transAxes)
        return

    axis_data = panel_data
    mean_plot_x: Optional[np.ndarray] = None
    aligned = np.full((0, 0), np.nan)
    mean = np.asarray([], dtype=float)
    ci = np.asarray([], dtype=float)
    valid_mean_mask = np.asarray([], dtype=bool)

    max_len = max(len(data[key]) for data in axis_data)
    precomputed_mean_ci = get_precomputed_angle_mean_ci(
        mean_panel_data,
        key=key,
        panel_config=resolved_panel,
        truncation_lower=truncation_lower,
        truncation_upper=truncation_upper,
    )
    if resolved_panel.x_key is not None:
        if resolved_panel.xlim is not None:
            x_min, x_max = resolved_panel.xlim
        else:
            x_arrays = [
                np.asarray(data[resolved_panel.x_key], dtype=float)
                for data in panel_data
                if resolved_panel.x_key in data and len(data[resolved_panel.x_key]) > 0
            ]
            finite_x = np.concatenate([x_values[np.isfinite(x_values)] for x_values in x_arrays]) if x_arrays else np.array([])
            x_min = float(np.min(finite_x)) if finite_x.size else 0.0
            x_max = float(np.max(finite_x)) if finite_x.size else 1.0
        sample_count = max(max_len, 2)
        plot_x = np.linspace(x_min, x_max, sample_count)
        endpoint_x = np.asarray(
            [
                np.asarray(data[resolved_panel.x_key], dtype=float)[-1]
                for data in panel_data
                if resolved_panel.x_key in data and len(data[resolved_panel.x_key]) > 0
            ],
            dtype=float,
        )
        if endpoint_x.size:
            endpoint_x = endpoint_x[
                np.isfinite(endpoint_x)
                & (endpoint_x >= min(x_min, x_max))
                & (endpoint_x <= max(x_min, x_max))
            ]
            plot_x = np.unique(np.concatenate([plot_x, endpoint_x]))
        mean_plot_x = plot_x
        if mean_panel_data:
            mean_endpoint_x = np.asarray(
                [
                    np.asarray(data[resolved_panel.x_key], dtype=float)[-1]
                    for data in mean_panel_data
                    if resolved_panel.x_key in data and len(data[resolved_panel.x_key]) > 0
                ],
                dtype=float,
            )
            if mean_endpoint_x.size:
                mean_endpoint_x = mean_endpoint_x[
                    np.isfinite(mean_endpoint_x)
                    & (mean_endpoint_x >= min(x_min, x_max))
                    & (mean_endpoint_x <= max(x_min, x_max))
                ]
                mean_plot_x = np.unique(np.concatenate([mean_plot_x, mean_endpoint_x]))
            if precomputed_mean_ci is not None:
                mean_plot_x, mean, ci, valid_mean_mask = precomputed_mean_ci
            else:
                aligned = align_series_by_x(mean_panel_data, key, resolved_panel.x_key, mean_plot_x)
    else:
        plot_x = np.arange(max_len)
        if precomputed_mean_ci is not None:
            mean_plot_x, mean, ci, valid_mean_mask = precomputed_mean_ci
        elif mean_panel_data:
            mean_max_len = max(len(data[key]) for data in mean_panel_data)
            mean_plot_x = np.arange(mean_max_len)
            aligned = align_series(mean_panel_data, key, mean_max_len)
    if aligned.size and not valid_mean_mask.size:
        mean, ci = compute_mean_and_ci(aligned)
        valid_mean_mask = np.any(np.isfinite(aligned), axis=0)

    plot_data = select_plot_trajectories(panel_data, max_plot_trajectories)
    for data in plot_data:
        if resolved_panel.x_key is not None and resolved_panel.x_key in data:
            trajectory_x = np.asarray(data[resolved_panel.x_key], dtype=float)
        else:
            trajectory_x = np.arange(len(data[key]))
        ax.plot(
            trajectory_x,
            data[key],
            color=TRAJECTORY_COLOR,
            alpha=0.18,
            linewidth=0.8,
            zorder=1,
            rasterized=True,
        )

    if mean_plot_x is not None and valid_mean_mask.size and np.any(valid_mean_mask):
        ax.fill_between(
            mean_plot_x[valid_mean_mask],
            (mean - ci)[valid_mean_mask],
            (mean + ci)[valid_mean_mask],
            color=BAND_COLOR,
            alpha=0.28,
            linewidth=0.0,
            zorder=2,
            label="95% CI",
        )
        ax.plot(
            mean_plot_x[valid_mean_mask],
            mean[valid_mean_mask],
            color=MEAN_COLOR,
            linewidth=1.8,
            zorder=3,
            label="Mean",
        )
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=0.9, zorder=0)
    if resolved_panel.show_threshold_lines:
        ax.axhline(threshold_value, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.0, zorder=0)
        ax.axhline(-threshold_value, color=THRESHOLD_COLOR, linestyle="--", linewidth=1.0, zorder=0)

    observed_limit = np.nanmax(np.abs(aligned)) if aligned.size else 0.0
    envelope_limit = np.nanmax(np.abs(mean) + ci) if mean.size else 0.0
    y_limit = round_axis_limit(max(threshold_value * 1.08, observed_limit * 1.12, envelope_limit * 1.15, 2.0))

    if resolved_panel.xlim is not None:
        ax.set_xlim(*resolved_panel.xlim)
    else:
        ax.set_xlim(float(plot_x[0]), float(plot_x[-1]) if len(plot_x) > 1 else 1.0)
    if resolved_panel.ylim is not None:
        ax.set_ylim(*resolved_panel.ylim)
    else:
        ax.set_ylim(-y_limit, y_limit)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=resolved_panel.x_key is None))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    if resolved_panel.ylim is not None:
        y_low, y_high = resolved_panel.ylim
        ax.set_yticks(np.linspace(float(y_low), float(y_high), 5))

    if resolved_panel.show_legend:
        legend_handles = [
            Line2D([0], [0], color=TRAJECTORY_COLOR, alpha=0.45, linewidth=1.2, label="Trajectories"),
            Line2D([0], [0], color=MEAN_COLOR, linewidth=1.8, label="Mean"),
            Patch(facecolor=BAND_COLOR, edgecolor="none", alpha=0.28, label="95% CI"),
        ]
        if resolved_panel.show_threshold_lines:
            legend_handles.append(
                Line2D([0], [0], color=THRESHOLD_COLOR, linestyle="--", linewidth=1.0, label="Safety limit")
            )
        ax.legend(
            handles=legend_handles,
            loc="upper right",
            ncol=2,
            columnspacing=0.9,
            handlelength=1.6,
            borderpad=0.35,
    )
    if resolved_panel.axis_postprocessor is not None:
        resolved_panel.axis_postprocessor(
            ax,
            {
                "valid_data": valid_data,
                "key": key,
                "angle_threshold": angle_threshold,
                "panel_config": resolved_panel,
            },
        )


def plot_trajectory_error_panel(
    ax: plt.Axes,
    all_data: Sequence[Dict[str, np.ndarray]],
    env_name: Optional[str],
    panel_config: Optional[TrajectoryErrorPanelConfig] = None,
    max_plot_trajectories: int = 600,
    statistic_data: Optional[Sequence[Dict[str, np.ndarray]]] = None,
) -> None:
    resolved_panel = panel_config or TrajectoryErrorPanelConfig()
    set_common_axis_style(ax)
    add_panel_title(ax, resolved_panel.panel_label, resolved_panel.title)
    ax.set_xlabel(resolved_panel.xlabel or "Time step")
    ax.set_ylabel(resolved_panel.ylabel)

    y_min, y_max = resolved_panel.ylim
    if y_min > y_max:
        y_min, y_max = y_max, y_min
    if np.isclose(y_min, y_max):
        y_max = y_min + 1.0

    def prepare_error_data(source_data: Sequence[Dict[str, np.ndarray]]) -> List[Dict[str, np.ndarray]]:
        prepared: List[Dict[str, np.ndarray]] = []
        for data in source_data:
            error_values = compute_trajectory_deviation_values(data, env_name=env_name)
            if error_values is None or len(error_values) == 0:
                continue

            plot_item = dict(data)
            error_series = np.asarray(error_values, dtype=float)
            if resolved_panel.use_absolute_value:
                error_series = np.abs(error_series)
                lower_bound = None
                upper_bound = float(y_max)
            else:
                reference_values = tuple(float(value) for value in resolved_panel.reference_line_values)
                lower_candidates = [value for value in reference_values if value < 0.0]
                upper_candidates = [value for value in reference_values if value > 0.0]
                lower_bound = max(lower_candidates) if lower_candidates else None
                upper_bound = min(upper_candidates) if upper_candidates else None
            if resolved_panel.x_key is not None:
                if resolved_panel.x_key == "time" and resolved_panel.x_key not in plot_item:
                    plot_item[resolved_panel.x_key] = compute_time_series(data)
                if resolved_panel.x_key in plot_item:
                    x_series = np.asarray(plot_item[resolved_panel.x_key], dtype=float)
                    common_len = min(len(error_series), len(x_series))
                    if common_len <= 0:
                        continue
                    plot_item[resolved_panel.x_key] = x_series[:common_len]
                    error_series = error_series[:common_len]
            if resolved_panel.x_key is not None and resolved_panel.x_key in plot_item:
                x_series = np.asarray(plot_item[resolved_panel.x_key], dtype=float)
            else:
                x_series = np.arange(len(error_series), dtype=float)
            x_series, error_series = truncate_xy_series_at_y_bounds(
                x_series,
                error_series,
                lower=lower_bound,
                upper=upper_bound,
            )
            if len(error_series) == 0:
                continue
            if resolved_panel.x_key is not None:
                plot_item[resolved_panel.x_key] = x_series
            plot_item["trajectory_error"] = error_series
            prepared.append(plot_item)
        return prepared

    error_data = prepare_error_data(all_data)
    mean_error_data = (
        prepare_error_data(statistic_data)
        if statistic_data is not None
        else error_data
    )

    if not error_data:
        if resolved_panel.xlim is not None:
            ax.set_xlim(*resolved_panel.xlim)
        else:
            ax.set_xlim(0.0, 1.0)
        ax.set_ylim(y_min, y_max)
        ax.text(0.5, 0.5, "No trajectory error data", ha="center", va="center", transform=ax.transAxes)
        return

    mean_plot_x: Optional[np.ndarray] = None
    aligned = np.full((0, 0), np.nan)
    mean = np.asarray([], dtype=float)
    ci = np.asarray([], dtype=float)
    valid_mean_mask = np.asarray([], dtype=bool)

    max_len = max(len(data["trajectory_error"]) for data in error_data)
    if resolved_panel.x_key is not None:
        if resolved_panel.xlim is not None:
            x_min, x_max = resolved_panel.xlim
        else:
            x_arrays = [
                np.asarray(data[resolved_panel.x_key], dtype=float)
                for data in error_data
                if resolved_panel.x_key in data and len(data[resolved_panel.x_key]) > 0
            ]
            finite_x = np.concatenate([x_values[np.isfinite(x_values)] for x_values in x_arrays]) if x_arrays else np.array([])
            x_min = float(np.min(finite_x)) if finite_x.size else 0.0
            x_max = float(np.max(finite_x)) if finite_x.size else 1.0
        sample_count = max(max_len, 2)
        plot_x = np.linspace(x_min, x_max, sample_count)
        endpoint_x = np.asarray(
            [
                np.asarray(data[resolved_panel.x_key], dtype=float)[-1]
                for data in error_data
                if resolved_panel.x_key in data and len(data[resolved_panel.x_key]) > 0
            ],
            dtype=float,
        )
        if endpoint_x.size:
            endpoint_x = endpoint_x[
                np.isfinite(endpoint_x)
                & (endpoint_x >= min(x_min, x_max))
                & (endpoint_x <= max(x_min, x_max))
            ]
            plot_x = np.unique(np.concatenate([plot_x, endpoint_x]))
        mean_plot_x = plot_x
        if mean_error_data:
            mean_endpoint_x = np.asarray(
                [
                    np.asarray(data[resolved_panel.x_key], dtype=float)[-1]
                    for data in mean_error_data
                    if resolved_panel.x_key in data and len(data[resolved_panel.x_key]) > 0
                ],
                dtype=float,
            )
            if mean_endpoint_x.size:
                mean_endpoint_x = mean_endpoint_x[
                    np.isfinite(mean_endpoint_x)
                    & (mean_endpoint_x >= min(x_min, x_max))
                    & (mean_endpoint_x <= max(x_min, x_max))
                ]
                mean_plot_x = np.unique(np.concatenate([mean_plot_x, mean_endpoint_x]))
            aligned = align_series_by_x(mean_error_data, "trajectory_error", resolved_panel.x_key, mean_plot_x)
    else:
        plot_x = np.arange(max_len)
        if mean_error_data:
            mean_max_len = max(len(data["trajectory_error"]) for data in mean_error_data)
            mean_plot_x = np.arange(mean_max_len)
            aligned = align_series(mean_error_data, "trajectory_error", mean_max_len)
    if aligned.size:
        mean, ci = compute_mean_and_ci(aligned)
        valid_mean_mask = np.any(np.isfinite(aligned), axis=0)

    plot_data = select_plot_trajectories(error_data, max_plot_trajectories)
    for data in plot_data:
        if resolved_panel.x_key is not None and resolved_panel.x_key in data:
            trajectory_x = np.asarray(data[resolved_panel.x_key], dtype=float)
        else:
            trajectory_x = np.arange(len(data["trajectory_error"]))
        ax.plot(
            trajectory_x,
            data["trajectory_error"],
            color=TRAJECTORY_COLOR,
            alpha=0.18,
            linewidth=0.8,
            zorder=1,
            rasterized=True,
        )

    if mean_plot_x is not None and valid_mean_mask.size and np.any(valid_mean_mask):
        ci_lower = mean - ci
        if resolved_panel.use_absolute_value and y_min >= 0.0:
            ci_lower = np.maximum(ci_lower, y_min)
        ax.fill_between(
            mean_plot_x[valid_mean_mask],
            ci_lower[valid_mean_mask],
            (mean + ci)[valid_mean_mask],
            color=BAND_COLOR,
            alpha=0.28,
            linewidth=0.0,
            zorder=2,
            label="95% CI",
        )
        ax.plot(
            mean_plot_x[valid_mean_mask],
            mean[valid_mean_mask],
            color=MEAN_COLOR,
            linewidth=1.8,
            zorder=3,
            label="Mean",
        )
    ax.axhline(0.0, color="#888888", linestyle=":", linewidth=0.9, zorder=0)
    for reference_value in resolved_panel.reference_line_values:
        ax.axhline(
            float(reference_value),
            color=THRESHOLD_COLOR,
            linestyle="--",
            linewidth=1.0,
            zorder=4,
            clip_on=False,
        )

    if resolved_panel.xlim is not None:
        ax.set_xlim(*resolved_panel.xlim)
    else:
        ax.set_xlim(float(plot_x[0]), float(plot_x[-1]) if len(plot_x) > 1 else 1.0)
    ax.set_ylim(y_min, y_max)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=resolved_panel.x_key is None))
    ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_yticks(np.linspace(float(y_min), float(y_max), 5))

    if resolved_panel.show_legend:
        ax.legend(
            handles=[
                Line2D([0], [0], color=TRAJECTORY_COLOR, alpha=0.45, linewidth=1.2, label="Trajectories"),
                Line2D([0], [0], color=MEAN_COLOR, linewidth=1.8, label="Mean"),
                Patch(facecolor=BAND_COLOR, edgecolor="none", alpha=0.28, label="95% CI"),
            ],
            loc="upper right",
            ncol=2,
            columnspacing=0.9,
            handlelength=1.6,
            borderpad=0.35,
        )
    if resolved_panel.axis_postprocessor is not None:
        resolved_panel.axis_postprocessor(
            ax,
            {
                "all_data": all_data,
                "error_data": error_data,
                "statistic_error_data": mean_error_data,
                "env_name": env_name,
                "panel_config": resolved_panel,
            },
        )


def plot_metrics_table(
    ax: plt.Axes,
    all_data: Sequence[Dict[str, np.ndarray]],
    panel_config: Optional[MetricsTableConfig] = None,
) -> None:
    resolved_panel = panel_config or MetricsTableConfig()
    ax.set_axis_off()
    add_panel_title(ax, resolved_panel.panel_label, resolved_panel.title)

    if not all_data:
        ax.text(0.5, 0.5, "No valid trajectories", ha="center", va="center", transform=ax.transAxes)
        return

    ax.text(
        0.995,
        1.00,
        f"{resolved_panel.sample_label}, deg. Mean $\\pm$ SD; worst = maximum across N = {len(all_data)}.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color=TEXT_MUTED,
    )

    table = ax.table(
        cellText=summarize_angle_metrics(all_data),
        colLabels=["Metric", "Summary"],
        colLoc="left",
        cellLoc="left",
        colWidths=[0.24, 0.76],
        bbox=[0.0, 0.03, 1.0, 0.80],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.0)
    table.scale(1.0, 1.28)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d3d3d3")
        cell.set_linewidth(0.6)
        if row == 0:
            cell.set_facecolor("#eaf2f8")
            cell.get_text().set_weight("bold")
            cell.get_text().set_color("#2f2f2f")
        else:
            cell.set_facecolor("#fbfbfb" if row % 2 == 1 else "white")
            if col == 0:
                cell.get_text().set_weight("bold")
    if resolved_panel.axis_postprocessor is not None:
        resolved_panel.axis_postprocessor(
            ax,
            {
                "all_data": all_data,
                "panel_config": resolved_panel,
            },
        )


def sanitize_filename_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_") or "panel"


def wrap_folder_name(name: str, width: int = 18) -> str:
    return textwrap.fill(name, width=width, break_long_words=True, break_on_hyphens=False)


def get_summary_metric_label(
    metric_key: str,
    metric_definitions: Optional[Sequence[MetricDefinition]] = None,
) -> str:
    resolved_metric_definitions = metric_definitions or METRIC_DEFINITIONS
    for definition_key, label, _ in resolved_metric_definitions:
        if definition_key == metric_key:
            return label
    if metric_key == TRAJECTORY_DEVIATION_METRIC_KEY:
        return r"$\mathrm{RMSE}(e_{\mathrm{traj}})$"
    if metric_key == TRAJECTORY_DEVIATION_MAX_METRIC_KEY:
        return r"$\max(|e_{\mathrm{traj}}|)$"
    return metric_key


def get_available_summary_metric_keys(
    summary_records: Sequence[Dict[str, object]],
    metric_definitions: Optional[Sequence[MetricDefinition]] = None,
) -> List[str]:
    resolved_metric_definitions = metric_definitions or METRIC_DEFINITIONS
    metric_keys = [metric_key for metric_key, _, _ in resolved_metric_definitions]
    if any(record.get(TRAJECTORY_DEVIATION_METRIC_KEY) is not None for record in summary_records):
        metric_keys.append(TRAJECTORY_DEVIATION_METRIC_KEY)
    if any(record.get(TRAJECTORY_DEVIATION_MAX_METRIC_KEY) is not None for record in summary_records):
        metric_keys.append(TRAJECTORY_DEVIATION_MAX_METRIC_KEY)
    return [
        metric_key
        for metric_key in metric_keys
        if any(record.get(metric_key) is not None for record in summary_records)
    ]


def order_summary_metric_keys(
    metric_keys: Sequence[str],
    metric_order: Sequence[str] = (),
) -> List[str]:
    ordered_keys: List[str] = []
    available_keys = list(metric_keys)
    for metric_key in metric_order:
        if metric_key in available_keys and metric_key not in ordered_keys:
            ordered_keys.append(metric_key)
    ordered_keys.extend(metric_key for metric_key in available_keys if metric_key not in ordered_keys)
    return ordered_keys


def get_summary_metric_keys(
    summary_records: Sequence[Dict[str, object]],
    metric_order: Sequence[str] = (),
    metric_definitions: Optional[Sequence[MetricDefinition]] = None,
) -> List[str]:
    return order_summary_metric_keys(
        get_available_summary_metric_keys(
            summary_records,
            metric_definitions=metric_definitions,
        ),
        metric_order=metric_order,
    )


def create_summary_table_figure(
    summary_records: Sequence[Dict[str, object]],
    profile: Optional[RenderProfile] = None,
) -> plt.Figure:
    resolved_profile = resolve_render_profile(profile)
    if profile is not None and resolved_profile.create_summary_table_figure_fn is not None:
        return resolved_profile.create_summary_table_figure_fn(summary_records, resolved_profile)

    summary_config = resolved_profile.summary_table
    num_rows = max(1, len(summary_records))
    wrapped_model_names = [
        wrap_folder_name(
            str(record.get("display_name") or record["folder_name"]),
            width=summary_config.wrap_width,
        )
        for record in summary_records
    ] or [""]
    extra_model_lines = sum(name.count("\n") for name in wrapped_model_names)
    include_trajectory_deviation = any(
        record.get(TRAJECTORY_DEVIATION_METRIC_KEY) is not None for record in summary_records
    )
    include_trajectory_deviation_max = any(
        record.get(TRAJECTORY_DEVIATION_MAX_METRIC_KEY) is not None for record in summary_records
    )
    metric_keys = get_summary_metric_keys(
        summary_records,
        metric_order=summary_config.metric_order,
        metric_definitions=resolved_profile.metric_definitions,
    )

    fig_width = 15.8 + 1.9 * max(0, len(metric_keys) - len(METRIC_DEFINITIONS))
    fig_height = max(2.9, 1.45 + 0.60 * num_rows + 0.14 * extra_model_lines)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()
    add_panel_title(ax, "", summary_config.title)

    summary_note = (
        summary_config.note_with_trajectory_deviation
        if include_trajectory_deviation or include_trajectory_deviation_max
        else summary_config.note_without_trajectory_deviation
    )
    ax.text(
        0.995,
        1.005,
        summary_note,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color=TEXT_MUTED,
    )

    column_labels = ["Algorithm", "Traj.", "Success"] + [
        get_summary_metric_label(
            metric_key,
            metric_definitions=resolved_profile.metric_definitions,
        )
        for metric_key in metric_keys
    ]
    cell_rows: List[List[str]] = []
    for record in summary_records:
        success_count = int(record["success_count"])
        trajectory_count = int(record["trajectory_count"])
        success_rate = float(record["success_rate"]) * 100.0
        row = [
            wrap_folder_name(
                str(record.get("display_name") or record["folder_name"]),
                width=summary_config.wrap_width,
            ),
            str(trajectory_count),
            f"{success_count}/{trajectory_count} ({success_rate:.1f}%)",
        ]
        for metric_key in metric_keys:
            metric = record.get(metric_key)
            row.append("N/A" if metric is None else str(metric["formatted"]))
        cell_rows.append(row)

    base_col_widths = [0.13, 0.04, 0.09]
    metric_col_width = (0.995 - sum(base_col_widths)) / max(1, len(metric_keys))
    col_widths = base_col_widths + [metric_col_width] * len(metric_keys)

    table = ax.table(
        cellText=cell_rows,
        colLabels=column_labels,
        colLoc="center",
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0.0, 0.02, 1.0, 0.84],
    )
    table.auto_set_font_size(False)
    table_fontsize = (
        6.8
        if include_trajectory_deviation_max
        else 7.0 if include_trajectory_deviation else 7.2
    )
    table.set_fontsize(table_fontsize)
    table.scale(1.0, 1.30)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d0d0d0")
        cell.set_linewidth(0.55)
        if row == 0:
            cell.set_facecolor("#ddeaf3")
            cell.get_text().set_weight("bold")
            cell.get_text().set_color("#2f2f2f")
        else:
            cell.set_facecolor("#fcfcfc" if row % 2 == 1 else "white")
            if col == 0:
                cell.get_text().set_ha("left")
                cell.get_text().set_weight("bold")
    if summary_config.figure_postprocessor is not None:
        summary_config.figure_postprocessor(
            fig,
            {
                "summary_records": summary_records,
                "metric_keys": metric_keys,
                "axis": ax,
                "summary_config": summary_config,
            },
        )

    return fig


def export_summary_table_figure(
    summary_records: Sequence[Dict[str, object]],
    output_dir: str,
    output_name: str,
    dpi: int,
    save_pdf: bool,
    profile: Optional[RenderProfile] = None,
) -> List[str]:
    if not summary_records:
        return []

    fig = create_summary_table_figure(summary_records, profile=profile)
    exported_paths: List[str] = []

    png_path = os.path.join(output_dir, f"{output_name}_summary_table.png")
    fig.savefig(png_path, dpi=dpi)
    exported_paths.append(png_path)

    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{output_name}_summary_table.pdf")
        fig.savefig(pdf_path)
        exported_paths.append(pdf_path)

    plt.close(fig)
    return exported_paths


def get_summary_mean_radar_metric_keys(
    summary_records: Sequence[Dict[str, object]],
    metric_order: Sequence[str] = (),
    metric_definitions: Optional[Sequence[MetricDefinition]] = None,
) -> List[str]:
    return get_summary_metric_keys(
        summary_records,
        metric_order=metric_order,
        metric_definitions=metric_definitions,
    )


def get_summary_mean_radar_unit(
    metric_key: str,
    metric_units: Optional[Dict[str, str]] = None,
) -> str:
    if metric_units and metric_key in metric_units:
        return metric_units[metric_key]
    if metric_key in {
        "max_beta",
        "max_phi",
        "max_theta",
        "rms_beta",
        "rms_phi",
        "rms_theta",
    }:
        return "deg"
    if metric_key in {TRAJECTORY_DEVIATION_METRIC_KEY, TRAJECTORY_DEVIATION_MAX_METRIC_KEY}:
        return "m"
    return ""


def format_summary_mean_axis_bound(value: float) -> str:
    if abs(value) < 1.0:
        return f"{value:.2f}"
    if abs(value) < 10.0:
        return f"{value:.1f}"
    return f"{value:.0f}"


def get_summary_mean_radar_label(
    metric_key: str,
    axis_range: Optional[Tuple[float, float]] = None,
    metric_definitions: Optional[Sequence[MetricDefinition]] = None,
    metric_units: Optional[Dict[str, str]] = None,
) -> str:
    label_by_key = {
        "max_beta": r"$\max(|\beta|)$",
        "max_phi": r"$\max(|\phi|)$",
        "max_theta": r"$\max(|\theta|)$",
        "rms_beta": r"$\mathrm{RMS}(\beta)$",
        "rms_phi": r"$\mathrm{RMS}(\phi)$",
        "rms_theta": r"$\mathrm{RMS}(\theta)$",
        TRAJECTORY_DEVIATION_METRIC_KEY: r"$\mathrm{RMSE}(e_{\mathrm{traj}})$",
        TRAJECTORY_DEVIATION_MAX_METRIC_KEY: r"$\max(|e_{\mathrm{traj}}|)$",
    }
    unit = get_summary_mean_radar_unit(metric_key, metric_units=metric_units)
    base_label = label_by_key.get(
        metric_key,
        get_summary_metric_label(metric_key, metric_definitions=metric_definitions),
    )
    if metric_definitions is not None:
        base_label = get_summary_metric_label(metric_key, metric_definitions=metric_definitions)
    if axis_range is not None:
        low, high = axis_range
        range_label = (
            f"{format_summary_mean_axis_bound(low)}-"
            f"{format_summary_mean_axis_bound(high)}"
        )
        if unit:
            range_label = f"{range_label} {unit}"
        return f"{base_label}\n{range_label}"
    if unit:
        return f"{base_label}\n({unit})"
    return base_label


def extract_summary_mean_matrix(
    summary_records: Sequence[Dict[str, object]],
    metric_keys: Sequence[str],
) -> np.ndarray:
    rows: List[List[float]] = []
    for record in summary_records:
        row: List[float] = []
        for metric_key in metric_keys:
            metric = record.get(metric_key)
            if isinstance(metric, dict) and metric.get("mean") is not None:
                row.append(float(metric["mean"]))
            else:
                row.append(np.nan)
        rows.append(row)
    return np.asarray(rows, dtype=float)


def normalize_summary_algorithm_name(value: object) -> str:
    return re.sub(r"[\s_]+", "-", str(value).strip().upper()).strip("-")


def summary_record_matches_algorithm(
    record: Dict[str, object],
    algorithm_name: str,
) -> bool:
    target_name = normalize_summary_algorithm_name(algorithm_name)
    if not target_name:
        return False

    explicit_algorithm = record.get("algorithm_name")
    if explicit_algorithm is not None:
        return normalize_summary_algorithm_name(explicit_algorithm) == target_name

    identifier_names = [
        normalize_summary_algorithm_name(record.get(key))
        for key in ("folder_name", "label")
        if record.get(key) is not None
    ]
    if any(name == target_name for name in identifier_names):
        return True

    display_name = record.get("display_name")
    if display_name is None:
        return False

    normalized_display = normalize_summary_algorithm_name(display_name)
    if normalized_display != target_name:
        return False

    # Inferred display names can collapse PPO-family labels to "PPO"; avoid treating
    # explicit variants such as EPPO/PPPO/RPPO/PPO-DIST as the PPO baseline.
    return not any(
        name != target_name and target_name in name
        for name in identifier_names
    )


def find_summary_mean_radar_baseline_values(
    summary_records: Sequence[Dict[str, object]],
    metric_keys: Sequence[str],
    algorithm_name: Optional[str],
) -> Tuple[Optional[int], Optional[np.ndarray]]:
    if not algorithm_name:
        return None, None

    for record_index, record in enumerate(summary_records):
        if not summary_record_matches_algorithm(record, algorithm_name):
            continue
        values = extract_summary_mean_matrix([record], metric_keys)
        if values.size == 0:
            return record_index, None
        return record_index, values[0]

    return None, None


def compute_summary_mean_ratio_to_baseline(
    values: np.ndarray,
    baseline_values: np.ndarray,
) -> np.ndarray:
    ratio_values = np.full_like(values, np.nan, dtype=float)
    if values.size == 0 or baseline_values.size == 0:
        return ratio_values

    for col_index in range(values.shape[1]):
        if col_index >= len(baseline_values):
            continue
        baseline_value = float(baseline_values[col_index])
        if not np.isfinite(baseline_value) or np.isclose(baseline_value, 0.0):
            continue

        column = values[:, col_index]
        finite_mask = np.isfinite(column)
        ratio_values[finite_mask, col_index] = column[finite_mask] / baseline_value

    return ratio_values


def normalize_ratio_summary_mean_matrix(
    ratio_values: np.ndarray,
    ratio_range: Tuple[float, float],
) -> np.ndarray:
    if ratio_values.size == 0:
        return ratio_values

    low, high = map(float, ratio_range)
    if low > high:
        low, high = high, low
    if np.isclose(low, high):
        low, high = 0.0, 2.0

    normalized = np.full_like(ratio_values, np.nan, dtype=float)
    finite_mask = np.isfinite(ratio_values)
    normalized[finite_mask] = (ratio_values[finite_mask] - low) / (high - low)
    return np.clip(normalized, 0.0, 1.0)


def normalize_ratio_radar_tick(
    value: float,
    ratio_range: Tuple[float, float],
) -> float:
    low, high = map(float, ratio_range)
    if low > high:
        low, high = high, low
    if np.isclose(low, high):
        low, high = 0.0, 2.0
    return float(np.clip((float(value) - low) / (high - low), 0.0, 1.0))


def resolve_summary_mean_radar_ranges(
    values: np.ndarray,
    metric_keys: Sequence[str],
    range_overrides: Optional[Dict[str, Tuple[float, float]]] = None,
    expand_ranges_to_data: bool = True,
) -> List[Tuple[float, float]]:
    resolved_overrides = range_overrides or SUMMARY_MEAN_RADAR_RANGES
    axis_ranges: List[Tuple[float, float]] = []
    for col_index, metric_key in enumerate(metric_keys):
        column = values[:, col_index]
        finite_values = column[np.isfinite(column)]
        default_range = resolved_overrides.get(metric_key)

        if default_range is None:
            if finite_values.size:
                low = float(np.min(finite_values))
                high = float(np.max(finite_values))
                span = max(high - low, abs(high) * 0.12, 1e-6)
                low -= 0.18 * span
                high += 0.18 * span
                if low >= -1e-9 and np.min(finite_values) >= 0.0:
                    low = 0.0
            else:
                low, high = 0.0, 1.0
        else:
            low, high = map(float, default_range)

        if expand_ranges_to_data and finite_values.size:
            observed_low = float(np.min(finite_values))
            observed_high = float(np.max(finite_values))
            if observed_low < low or observed_high > high:
                low = min(low, observed_low)
                high = max(high, observed_high)
                span = max(high - low, abs(high) * 0.08, 1e-6)
                low -= 0.06 * span
                high += 0.06 * span
                if observed_low >= 0.0 and (default_range is None or default_range[0] >= 0.0):
                    low = max(0.0, low)

        if np.isclose(low, high):
            span = max(abs(high) * 0.1, 1.0)
            low -= span
            high += span
            if low < 0.0:
                low = 0.0
        axis_ranges.append((low, high))
    return axis_ranges


def normalize_summary_mean_matrix(
    values: np.ndarray,
    axis_ranges: Sequence[Tuple[float, float]],
    min_radius: float = 0.35,
) -> np.ndarray:
    if values.size == 0:
        return values

    rmin, rmax = float(min_radius), 1.0
    rmin = float(np.clip(rmin, 0.0, rmax))
    normalized = np.full_like(values, (rmin + rmax) / 2.0, dtype=float)
    for col_index, (low, high) in enumerate(axis_ranges):
        column = values[:, col_index]
        finite_mask = np.isfinite(column)
        if not np.any(finite_mask):
            continue

        if np.isclose(low, high):
            continue

        normalized[finite_mask, col_index] = rmin + (rmax - rmin) * (
            (column[finite_mask] - low) / (high - low)
        )
    return np.clip(normalized, rmin, rmax)


def close_polar_values(values: np.ndarray) -> np.ndarray:
    return np.concatenate([values, values[:1]])


def format_summary_mean_value(value: float) -> str:
    if not np.isfinite(value):
        return "N/A"
    if abs(value) >= 100.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def draw_summary_mean_radar_legend(
    fig: plt.Figure,
    legend_entries: Sequence[Tuple[str, str, float]],
    many_series: bool,
) -> None:
    if not legend_entries:
        return

    max_columns = 4 if many_series else min(4, len(legend_entries))
    columns = max(1, min(max_columns, len(legend_entries)))
    rows = int(np.ceil(len(legend_entries) / columns))
    col_width = 0.22
    start_x = 0.5 - 0.5 * columns * col_width
    row_step = 0.045
    base_y = 0.045
    fontsize = 12.5 if many_series else 13.5

    for index, (label, color, linewidth) in enumerate(legend_entries):
        row = index // columns
        col = index % columns
        y = base_y + (rows - 1 - row) * row_step
        line_x = start_x + col * col_width
        label_x = line_x + 0.045
        fig.add_artist(
            Line2D(
                [line_x, line_x + 0.032],
                [y, y],
                transform=fig.transFigure,
                color=color,
                linewidth=linewidth,
                solid_capstyle="round",
            )
        )
        fig.text(
            label_x,
            y,
            label,
            color=color,
            fontsize=fontsize,
            fontweight="bold" if linewidth >= 2.0 else "normal",
            va="center",
            ha="left",
        )
    return

    if many_series:
        start_x = 0.78
        start_y = 0.86
        row_step = 0.030
        line_x = start_x
        label_x = start_x + 0.075
        fontsize = 8.6
    else:
        start_x = 0.76
        start_y = 0.905
        row_step = 0.035
        line_x = start_x
        label_x = start_x + 0.065
        fontsize = 10.0

    for index, (label, color, linewidth) in enumerate(legend_entries):
        y = start_y - index * row_step
        fig.text(
            line_x,
            y,
            "────  ",
            color=color,
            fontsize=12,
            fontweight="bold" if linewidth >= 2.0 else "normal",
            va="center",
            ha="left",
        )
        fig.text(
            label_x,
            y,
            label,
            color="black",
            fontsize=fontsize,
            fontweight="bold" if linewidth >= 2.0 else "normal",
            va="center",
            ha="left",
        )


def create_summary_mean_radar_figure(
    summary_records: Sequence[Dict[str, object]],
    profile: Optional[RenderProfile] = None,
) -> Optional[plt.Figure]:
    resolved_profile = resolve_render_profile(profile)
    if profile is not None and resolved_profile.create_summary_mean_radar_figure_fn is not None:
        return resolved_profile.create_summary_mean_radar_figure_fn(summary_records, resolved_profile)

    summary_config = resolved_profile.summary_mean_radar
    metric_keys = get_summary_mean_radar_metric_keys(
        summary_records,
        metric_order=summary_config.metric_order,
        metric_definitions=resolved_profile.metric_definitions,
    )
    if not summary_records or not metric_keys:
        return None

    raw_values = extract_summary_mean_matrix(summary_records, metric_keys)
    baseline_record_index, baseline_values = find_summary_mean_radar_baseline_values(
        summary_records,
        metric_keys,
        summary_config.baseline_algorithm_name,
    )
    ratio_values: Optional[np.ndarray] = None
    has_ratio_values = bool(
        baseline_values is not None
        and np.any(np.isfinite(baseline_values) & ~np.isclose(baseline_values, 0.0))
    )
    if has_ratio_values and baseline_values is not None:
        ratio_values = compute_summary_mean_ratio_to_baseline(raw_values, baseline_values)
        has_ratio_values = bool(np.any(np.isfinite(ratio_values)))

    if has_ratio_values and ratio_values is not None:
        ratio_low, ratio_high = map(float, summary_config.ratio_range)
        if ratio_low > ratio_high:
            ratio_low, ratio_high = ratio_high, ratio_low
        if np.isclose(ratio_low, ratio_high):
            ratio_low, ratio_high = 0.0, 2.0
        ratio_range = (ratio_low, ratio_high)
        axis_ranges = [ratio_range for _ in metric_keys]
        radial_values = normalize_ratio_summary_mean_matrix(
            ratio_values,
            ratio_range,
        )
        category_labels = [
            get_summary_metric_label(
                metric_key,
                metric_definitions=resolved_profile.metric_definitions,
            )
            for metric_key in metric_keys
        ]
    else:
        ratio_range = None
        axis_ranges = resolve_summary_mean_radar_ranges(
            raw_values,
            metric_keys,
            range_overrides=summary_config.ranges,
            expand_ranges_to_data=summary_config.expand_ranges_to_data,
        )
        radial_values = normalize_summary_mean_matrix(
            raw_values,
            axis_ranges,
            min_radius=summary_config.normalized_min_radius,
        )
        category_labels = [
            get_summary_mean_radar_label(
                metric_key,
                axis_range,
                metric_definitions=resolved_profile.metric_definitions,
                metric_units=resolved_profile.summary_metric_units,
            )
            for metric_key, axis_range in zip(metric_keys, axis_ranges)
        ]
    num_categories = len(category_labels)
    angles = np.linspace(0, 2 * np.pi, num_categories, endpoint=False)
    closed_angles = close_polar_values(angles)

    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "mathtext.fontset": "dejavusans",
            "text.usetex": False,
        }
    ):
        many_series = len(summary_records) > 6
        fig, ax = plt.subplots(figsize=(13.5, 10.8), subplot_kw=dict(projection="polar"))

        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.set_ylim(0.0, 1.18)
        ax.set_frame_on(False)

        if has_ratio_values and ratio_range is not None:
            candidate_ticks = [
                ratio_range[0],
                0.6,
                1.0,
                1.2,
                ratio_range[1],
            ]
            grid_ticks = sorted(
                {
                    round(float(tick), 6)
                    for tick in candidate_ticks
                    if ratio_range[0] <= float(tick) <= ratio_range[1]
                }
            )
            grid_specs = [
                (normalize_ratio_radar_tick(tick, ratio_range), tick)
                for tick in grid_ticks
            ]
        else:
            grid_specs = [(radius, None) for radius in [0.4, 0.55, 0.7, 0.85, 1.0]]

        for radius, tick_value in grid_specs:
            if radius <= 0.0:
                continue
            is_baseline_grid = tick_value is not None and np.isclose(tick_value, 1.0)
            ax.plot(
                closed_angles,
                close_polar_values(np.full(num_categories, radius)),
                color="#aaaaaa" if is_baseline_grid else "#cccccc",
                lw=1.05 if is_baseline_grid else 0.8,
                linestyle="--",
                zorder=1,
            )
        for angle in angles:
            ax.plot([angle, angle], [0.0, 1.0], color="#cccccc", lw=0.8, linestyle="--", zorder=1)

        legend_entries: List[Tuple[str, str, float]] = []
        annotate_values = False
        annotation_offsets = np.linspace(0.08, -0.08, len(summary_records)) if len(summary_records) > 1 else [0.08]

        for record_index, record in enumerate(summary_records):
            color = summary_config.colors[record_index % len(summary_config.colors)]
            label = str(record.get("display_name") or record.get("folder_name") or f"Model {record_index + 1}")
            closed_radial = close_polar_values(radial_values[record_index])
            if baseline_record_index is None:
                linewidth = 2.8 if record_index == 0 else 1.3
            else:
                linewidth = 2.8 if record_index == baseline_record_index else 1.3

            ax.fill(closed_angles, closed_radial, color=color, alpha=0.18, zorder=3)
            ax.plot(
                closed_angles,
                closed_radial,
                color=color,
                lw=linewidth,
                solid_capstyle="round",
                zorder=4,
            )
            legend_entries.append((label, color, linewidth))

            if annotate_values:
                for angle, radius, value in zip(angles, radial_values[record_index], raw_values[record_index]):
                    if not np.isfinite(value):
                        continue
                    annotation_radius = float(radius + annotation_offsets[record_index])
                    if annotation_radius > 1.08:
                        annotation_radius = float(radius - abs(annotation_offsets[record_index]))
                    elif annotation_radius < 0.32:
                        annotation_radius = float(radius + abs(annotation_offsets[record_index]))
                    ax.text(
                        angle,
                        float(np.clip(annotation_radius, 0.32, 1.08)),
                        format_summary_mean_value(float(value)),
                        ha="center",
                        va="center",
                        fontsize=7.1,
                        color=color,
                        zorder=6,
                        bbox={
                            "boxstyle": "round,pad=0.12",
                            "facecolor": "white",
                            "edgecolor": "none",
                            "alpha": 0.86,
                        },
                    )

        label_radius = 1.11
        for angle, label in zip(angles, category_labels):
            if abs(np.sin(angle)) < 0.15:
                ha = "center"
            elif np.sin(angle) > 0:
                ha = "left"
            else:
                ha = "right"
            ax.text(
                angle,
                label_radius,
                label,
                ha=ha,
                va="center",
                fontsize=24.0,
                color="#333333",
                multialignment="center",
            )

        draw_summary_mean_radar_legend(fig, legend_entries, many_series=many_series)
        legend_columns = 4 if many_series else max(1, min(4, len(legend_entries)))
        legend_rows = int(np.ceil(len(legend_entries) / legend_columns)) if legend_entries else 0
        bottom_margin = 0.13 + max(0, legend_rows - 1) * 0.045
        fig.subplots_adjust(left=0.12, right=0.88, top=0.965, bottom=bottom_margin)
        if summary_config.figure_postprocessor is not None:
            summary_config.figure_postprocessor(
                fig,
                {
                    "summary_records": summary_records,
                    "metric_keys": metric_keys,
                    "raw_values": raw_values,
                    "ratio_values": ratio_values,
                    "radial_values": radial_values,
                    "axis_ranges": axis_ranges,
                    "baseline_algorithm_name": summary_config.baseline_algorithm_name,
                    "baseline_record_index": baseline_record_index,
                    "baseline_values": baseline_values,
                    "ratio_range": ratio_range,
                    "polar_axis": ax,
                    "summary_config": summary_config,
                },
            )

        return fig


def export_summary_mean_radar_figure(
    summary_records: Sequence[Dict[str, object]],
    output_dir: str,
    output_name: str,
    dpi: int,
    save_pdf: bool,
    profile: Optional[RenderProfile] = None,
) -> List[str]:
    fig = create_summary_mean_radar_figure(summary_records, profile=profile)
    if fig is None:
        return []

    exported_paths: List[str] = []
    png_path = os.path.join(output_dir, f"{output_name}_summary_mean_radar.png")
    fig.savefig(png_path, dpi=dpi, facecolor="white")
    exported_paths.append(png_path)

    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{output_name}_summary_mean_radar.pdf")
        fig.savefig(pdf_path, facecolor="white")
        exported_paths.append(pdf_path)

    plt.close(fig)
    return exported_paths


def summarize_model_source(
    model_source: Dict[str, object],
    args: argparse.Namespace,
    profile: Optional[RenderProfile] = None,
) -> Optional[Dict[str, object]]:
    resolved_profile = resolve_render_profile(profile)
    resolved_env_name = (
        canonicalize_env_name(model_source.get("env_name"))
        or canonicalize_env_name(getattr(args, "env_name", None))
        or canonicalize_env_name(resolved_profile.env_name)
        or infer_env_name_from_strings(
            model_source.get("label"),
            *(model_source.get("model_dirs") or []),
            model_source.get("primary_dir"),
        )
    )
    resolved_display_name = model_source.get("display_name") or infer_algorithm_display_name(
        model_source.get("label"),
        *(model_source.get("model_dirs") or []),
        model_source.get("primary_dir"),
    )
    loaded = load_folder_data(
        model_dirs=model_source["model_dirs"],
        threshold_x=args.threshold_x,
        angle_threshold=args.angle_threshold,
        env_name=resolved_env_name,
        profile=resolved_profile,
        verbose=False,
    )
    if loaded is None:
        return None
    all_data, valid_data = loaded
    summary_data, summary_valid_data = prepare_summary_data_for_profile(
        resolved_profile,
        all_data=all_data,
        threshold_x=args.threshold_x,
        angle_threshold=args.angle_threshold,
        env_name=resolved_env_name,
    )
    return build_folder_summary(
        model_label=str(model_source["label"]),
        all_data=summary_data,
        valid_data=summary_valid_data,
        threshold_x=args.threshold_x,
        angle_threshold=args.angle_threshold,
        env_name=resolved_env_name,
        display_name=resolved_display_name,
        algorithm_name=model_source.get("algorithm_name"),
        metric_definitions=resolved_profile.metric_definitions,
        profile=resolved_profile,
    )


def create_figure(
    model_label: str,
    all_data: Sequence[Dict[str, np.ndarray]],
    valid_data: Sequence[Dict[str, np.ndarray]],
    threshold_x: float,
    angle_threshold: float,
    figure_width: float,
    figure_height: float,
    show_run_label: bool,
    profile: Optional[RenderProfile] = None,
    max_plot_trajectories: int = 600,
    env_name: Optional[str] = None,
) -> plt.Figure:
    resolved_profile = resolve_render_profile(profile)
    resolved_env_name = canonicalize_env_name(env_name) or canonicalize_env_name(resolved_profile.env_name)
    if profile is not None and resolved_profile.create_figure_fn is not None:
        return resolved_profile.create_figure_fn(
            model_label,
            all_data,
            valid_data,
            threshold_x,
            angle_threshold,
            figure_width,
            figure_height,
            show_run_label,
            resolved_profile,
        )

    layout = resolved_profile.layout
    success_flags = [
        is_successful_trajectory(
            data,
            threshold_x=threshold_x,
            angle_threshold=angle_threshold,
            env_name=resolved_env_name,
            profile=resolved_profile,
        )
        for data in all_data
    ]
    survival_distances = [
        compute_survival_distance(
            data,
            angle_threshold=angle_threshold,
            env_name=resolved_env_name,
            profile=resolved_profile,
        )
        for data in all_data
    ]
    success_rate = (sum(success_flags) / len(success_flags)) if success_flags else 0.0
    statistic_data = select_statistic_data_for_profile(
        all_data,
        valid_data,
        resolved_profile,
        threshold_x=threshold_x,
        angle_threshold=angle_threshold,
        env_name=resolved_env_name,
    )

    fig = plt.figure(figsize=(figure_width, figure_height))
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=list(layout.gridspec_height_ratios),
        hspace=layout.gridspec_hspace,
        wspace=layout.gridspec_wspace,
    )

    ax_survival = fig.add_subplot(gs[0, 0])
    ax_roll = fig.add_subplot(gs[0, 1])
    ax_pitch = fig.add_subplot(gs[1, 0])
    ax_beta = fig.add_subplot(gs[1, 1])
    ax_table = fig.add_subplot(gs[2, :])

    plot_survival_panel(
        ax_survival,
        all_data=all_data,
        success_flags=success_flags,
        survival_distances=survival_distances,
        threshold_x=threshold_x,
        success_rate=success_rate,
        panel_config=resolved_profile.survival_panel,
        max_plot_trajectories=max_plot_trajectories,
    )
    for angle_ax, panel_config in zip(
        (ax_roll, ax_pitch, ax_beta),
        resolved_profile.angle_panels,
    ):
        plot_angle_panel(
            angle_ax,
            valid_data=all_data,
            key=panel_config.key,
            ylabel=panel_config.ylabel,
            panel_label=panel_config.panel_label,
            title=panel_config.title,
            angle_threshold=angle_threshold,
            show_legend=panel_config.show_legend,
            panel_config=panel_config,
            max_plot_trajectories=max_plot_trajectories,
            statistic_data=statistic_data,
        )
    plot_metrics_table(
        ax_table,
        all_data=statistic_data,
        panel_config=resolved_profile.metrics_table,
    )

    fig.align_ylabels([ax_survival, ax_roll, ax_pitch, ax_beta])
    fig.subplots_adjust(
        left=layout.combined_left,
        right=layout.combined_right,
        bottom=layout.combined_bottom,
        top=layout.combined_top,
    )

    if show_run_label:
        fig.text(
            0.5,
            0.985,
            model_label,
            ha="center",
            va="top",
            fontsize=8.5,
            color=TEXT_MUTED,
        )
    if layout.combined_figure_postprocessor is not None:
        layout.combined_figure_postprocessor(
            fig,
            {
                "model_label": model_label,
                "all_data": all_data,
                "valid_data": valid_data,
                "threshold_x": threshold_x,
                "angle_threshold": angle_threshold,
                "show_run_label": show_run_label,
                "figure_axes": {
                    "survival": ax_survival,
                    "angles": [ax_roll, ax_pitch, ax_beta],
                    "table": ax_table,
                },
                "profile": resolved_profile,
            },
        )

    return fig


def compute_panel_plot_inputs(
    all_data: Sequence[Dict[str, np.ndarray]],
    threshold_x: float,
    angle_threshold: float,
    env_name: Optional[str] = None,
    profile: Optional[RenderProfile] = None,
) -> Tuple[List[bool], List[float], float]:
    resolved_env_name = canonicalize_env_name(env_name)
    resolved_profile = resolve_render_profile(profile)
    success_flags = [
        is_successful_trajectory(
            data,
            threshold_x=threshold_x,
            angle_threshold=angle_threshold,
            env_name=resolved_env_name,
            profile=resolved_profile,
        )
        for data in all_data
    ]
    survival_distances = [
        compute_survival_distance(
            data,
            angle_threshold=angle_threshold,
            env_name=resolved_env_name,
            profile=resolved_profile,
        )
        for data in all_data
    ]
    success_rate = (sum(success_flags) / len(success_flags)) if success_flags else 0.0
    return success_flags, survival_distances, success_rate


def select_statistic_data_for_profile(
    all_data: Sequence[Dict[str, np.ndarray]],
    valid_data: Sequence[Dict[str, np.ndarray]],
    profile: Optional[RenderProfile] = None,
    threshold_x: Optional[float] = None,
    angle_threshold: Optional[float] = None,
    env_name: Optional[str] = None,
) -> Sequence[Dict[str, np.ndarray]]:
    resolved_profile = resolve_render_profile(profile)
    if not resolved_profile.metrics_use_successful_trajectories_only:
        return all_data
    if threshold_x is None or angle_threshold is None:
        return valid_data
    return [
        data
        for data in all_data
        if is_statistic_successful_trajectory(
            data,
            threshold_x=float(threshold_x),
            angle_threshold=float(angle_threshold),
            env_name=env_name,
            profile=resolved_profile,
        )
    ]


def create_panel_canvas(
    model_label: str,
    figure_width: float,
    figure_height: float,
    show_run_label: bool,
    profile: Optional[RenderProfile] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    layout = resolve_render_profile(profile).layout
    panel_width = max(layout.split_panel_min_width, figure_width * layout.split_panel_width_ratio)
    panel_height = max(layout.split_panel_min_height, figure_height * layout.split_panel_height_ratio)
    fig, ax = plt.subplots(figsize=(panel_width, panel_height))
    top_margin = layout.split_panel_top_with_label if show_run_label else layout.split_panel_top_without_label
    fig.subplots_adjust(
        left=layout.split_panel_left,
        right=layout.split_panel_right,
        bottom=layout.split_panel_bottom,
        top=top_margin,
    )
    if show_run_label:
        fig.text(
            0.5,
            0.985,
            model_label,
            ha="center",
            va="top",
            fontsize=8.5,
            color=TEXT_MUTED,
        )
    return fig, ax


def create_bcdf_stack_figure(
    model_label: str,
    all_data: Sequence[Dict[str, np.ndarray]],
    statistic_data: Sequence[Dict[str, np.ndarray]],
    angle_threshold: float,
    figure_width: float,
    figure_height: float,
    show_run_label: bool,
    profile: Optional[RenderProfile] = None,
    max_plot_trajectories: int = 600,
    env_name: Optional[str] = None,
) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
    resolved_profile = resolve_render_profile(profile)
    layout = resolved_profile.layout
    panel_width = max(layout.split_panel_min_width, figure_width * layout.split_panel_width_ratio)
    panel_side = panel_width
    panel_height = panel_side * 1.02
    fig, axes_array = plt.subplots(
        4,
        1,
        sharex=True,
        figsize=(panel_side, panel_height),
    )
    axes = list(np.ravel(axes_array))
    top_margin = 0.91 if show_run_label else 0.975
    hspace = 0.52 if not HIDE_PANEL_TITLES else 0.34
    fig.subplots_adjust(
        left=layout.split_panel_left,
        right=layout.split_panel_right,
        bottom=0.13,
        top=top_margin,
        hspace=hspace,
    )
    if show_run_label:
        fig.text(
            0.5,
            0.985,
            model_label,
            ha="center",
            va="top",
            fontsize=8.5,
            color=TEXT_MUTED,
        )

    for ax, panel_config in zip(axes[:3], resolved_profile.angle_panels[:3]):
        plot_angle_panel(
            ax,
            valid_data=all_data,
            key=panel_config.key,
            ylabel=panel_config.ylabel,
            panel_label=panel_config.panel_label,
            title=panel_config.title,
            angle_threshold=angle_threshold,
            show_legend=panel_config.show_legend,
            panel_config=panel_config,
            max_plot_trajectories=max_plot_trajectories,
            statistic_data=statistic_data,
        )
        ax.set_xlabel("")
        ax.tick_params(axis="x", which="both", labelbottom=False)

    signed_error_panel = resolve_trajectory_error_panel_config(
        resolved_profile,
        panel_config=resolved_profile.signed_trajectory_error_panel,
    )
    if signed_error_panel is not None:
        plot_trajectory_error_panel(
            axes[3],
            all_data=all_data,
            env_name=env_name,
            panel_config=signed_error_panel,
            max_plot_trajectories=max_plot_trajectories,
            statistic_data=statistic_data,
        )
    axes[3].tick_params(axis="x", which="both", labelbottom=True)
    fig.align_ylabels(axes)
    return fig, axes


def create_split_panel_figures(
    model_label: str,
    all_data: Sequence[Dict[str, np.ndarray]],
    valid_data: Sequence[Dict[str, np.ndarray]],
    threshold_x: float,
    angle_threshold: float,
    figure_width: float,
    figure_height: float,
    show_run_label: bool,
    profile: Optional[RenderProfile] = None,
    max_plot_trajectories: int = 600,
    env_name: Optional[str] = None,
) -> List[Tuple[str, plt.Figure]]:
    resolved_profile = resolve_render_profile(profile)
    resolved_env_name = canonicalize_env_name(env_name) or canonicalize_env_name(resolved_profile.env_name)
    if profile is not None and resolved_profile.create_split_panel_figures_fn is not None:
        return resolved_profile.create_split_panel_figures_fn(
            model_label,
            all_data,
            valid_data,
            threshold_x,
            angle_threshold,
            figure_width,
            figure_height,
            show_run_label,
            resolved_profile,
        )

    layout = resolved_profile.layout
    success_flags, survival_distances, success_rate = compute_panel_plot_inputs(
        all_data=all_data,
        threshold_x=threshold_x,
        angle_threshold=angle_threshold,
        env_name=resolved_env_name,
        profile=resolved_profile,
    )
    statistic_data = select_statistic_data_for_profile(
        all_data,
        valid_data,
        resolved_profile,
        threshold_x=threshold_x,
        angle_threshold=angle_threshold,
        env_name=resolved_env_name,
    )
    panel_figures: List[Tuple[str, plt.Figure]] = []

    fig_survival, ax_survival = create_panel_canvas(
        model_label=model_label,
        figure_width=figure_width,
        figure_height=figure_height,
        show_run_label=show_run_label,
        profile=resolved_profile,
    )
    plot_survival_panel(
        ax_survival,
        all_data=all_data,
        success_flags=success_flags,
        survival_distances=survival_distances,
        threshold_x=threshold_x,
        success_rate=success_rate,
        panel_config=resolved_profile.survival_panel,
        max_plot_trajectories=max_plot_trajectories,
    )
    if layout.split_panel_figure_postprocessor is not None:
        layout.split_panel_figure_postprocessor(
            fig_survival,
            {
                "panel_stem": resolved_profile.survival_panel.stem,
                "panel_key": "survival",
                "axis": ax_survival,
                "model_label": model_label,
                "profile": resolved_profile,
            },
        )
    panel_figures.append((resolved_profile.survival_panel.stem, fig_survival))

    for panel_config in resolved_profile.angle_panels:
        panel_fig, panel_ax = create_panel_canvas(
            model_label=model_label,
            figure_width=figure_width,
            figure_height=figure_height,
            show_run_label=show_run_label,
            profile=resolved_profile,
        )
        plot_angle_panel(
            panel_ax,
            valid_data=all_data,
            key=panel_config.key,
            ylabel=panel_config.ylabel,
            panel_label=panel_config.panel_label,
            title=panel_config.title,
            angle_threshold=angle_threshold,
            show_legend=panel_config.show_legend,
            panel_config=panel_config,
            max_plot_trajectories=max_plot_trajectories,
            statistic_data=statistic_data,
        )
        if layout.split_panel_figure_postprocessor is not None:
            layout.split_panel_figure_postprocessor(
                panel_fig,
                {
                    "panel_stem": panel_config.stem,
                    "panel_key": panel_config.key,
                    "axis": panel_ax,
                    "model_label": model_label,
                    "profile": resolved_profile,
                },
            )
        panel_figures.append((panel_config.stem, panel_fig))

    error_panel_configs = (
        resolved_profile.trajectory_error_panel,
        resolved_profile.signed_trajectory_error_panel,
    )
    for raw_error_panel in error_panel_configs:
        trajectory_error_panel = resolve_trajectory_error_panel_config(
            resolved_profile,
            panel_config=raw_error_panel,
        )
        if trajectory_error_panel is None:
            continue

        panel_fig, panel_ax = create_panel_canvas(
            model_label=model_label,
            figure_width=figure_width,
            figure_height=figure_height,
            show_run_label=show_run_label,
            profile=resolved_profile,
        )
        plot_trajectory_error_panel(
            panel_ax,
            all_data=all_data,
            env_name=resolved_env_name,
            panel_config=trajectory_error_panel,
            max_plot_trajectories=max_plot_trajectories,
            statistic_data=statistic_data,
        )
        if layout.split_panel_figure_postprocessor is not None:
            layout.split_panel_figure_postprocessor(
                panel_fig,
                {
                    "panel_stem": trajectory_error_panel.stem,
                    "panel_key": "trajectory_error",
                    "axis": panel_ax,
                    "model_label": model_label,
                    "profile": resolved_profile,
                },
        )
        panel_figures.append((trajectory_error_panel.stem, panel_fig))

    if should_create_bcdf_stack_panel(resolved_profile):
        stack_fig, stack_axes = create_bcdf_stack_figure(
            model_label=model_label,
            all_data=all_data,
            statistic_data=statistic_data,
            angle_threshold=angle_threshold,
            figure_width=figure_width,
            figure_height=figure_height,
            show_run_label=show_run_label,
            profile=resolved_profile,
            max_plot_trajectories=max_plot_trajectories,
            env_name=resolved_env_name,
        )
        if layout.split_panel_figure_postprocessor is not None:
            layout.split_panel_figure_postprocessor(
                stack_fig,
                {
                    "panel_stem": BCDF_STACK_PANEL_STEM,
                    "panel_key": "bcdf_stack",
                    "axis": stack_axes[-1],
                    "axes": stack_axes,
                    "model_label": model_label,
                    "profile": resolved_profile,
                },
            )
        panel_figures.append((BCDF_STACK_PANEL_STEM, stack_fig))

    return panel_figures


def export_split_panel_figures(
    model_label: str,
    all_data: Sequence[Dict[str, np.ndarray]],
    valid_data: Sequence[Dict[str, np.ndarray]],
    threshold_x: float,
    angle_threshold: float,
    output_dir: str,
    export_stem: str,
    dpi: int,
    save_pdf: bool,
    figure_width: float,
    figure_height: float,
    show_run_label: bool,
    profile: Optional[RenderProfile] = None,
    max_plot_trajectories: int = 600,
    env_name: Optional[str] = None,
) -> List[str]:
    panel_output_dir = os.path.join(output_dir, export_stem)
    os.makedirs(panel_output_dir, exist_ok=True)
    for stale_suffix in (".png", ".pdf"):
        stale_combined_path = os.path.join(output_dir, f"{export_stem}{stale_suffix}")
        if os.path.isfile(stale_combined_path):
            os.remove(stale_combined_path)
    exported_paths: List[str] = []
    panel_figures = create_split_panel_figures(
        model_label=model_label,
        all_data=all_data,
        valid_data=valid_data,
        threshold_x=threshold_x,
        angle_threshold=angle_threshold,
        figure_width=figure_width,
        figure_height=figure_height,
        show_run_label=show_run_label,
        profile=profile,
        max_plot_trajectories=max_plot_trajectories,
        env_name=env_name,
    )

    try:
        for panel_stem, panel_fig in panel_figures:
            png_path = os.path.join(panel_output_dir, f"{panel_stem}.png")
            panel_fig.savefig(png_path, dpi=dpi)
            exported_paths.append(png_path)

            if save_pdf:
                pdf_path = os.path.join(panel_output_dir, f"{panel_stem}.pdf")
                panel_fig.savefig(pdf_path)
                exported_paths.append(pdf_path)
    finally:
        for _, panel_fig in panel_figures:
            plt.close(panel_fig)

    return exported_paths


def export_figure(
    fig: plt.Figure,
    output_dir: str,
    export_stem: str,
    dpi: int,
    save_pdf: bool,
) -> List[str]:
    exported_paths: List[str] = []

    png_path = os.path.join(output_dir, f"{export_stem}.png")
    fig.savefig(png_path, dpi=dpi)
    exported_paths.append(png_path)

    if save_pdf:
        pdf_path = os.path.join(output_dir, f"{export_stem}.pdf")
        fig.savefig(pdf_path)
        exported_paths.append(pdf_path)

    return exported_paths


def process_folder(
    model_source: Dict[str, object],
    folder_index: int,
    total_folders: int,
    output_dir: str,
    args: argparse.Namespace,
    profile: Optional[RenderProfile] = None,
) -> Optional[Tuple[List[str], Optional[plt.Figure], Dict[str, object]]]:
    resolved_profile = resolve_render_profile(profile)
    model_label = str(model_source["label"])
    model_dirs = [str(path) for path in model_source["model_dirs"]]
    primary_dir = str(model_source["primary_dir"])
    resolved_env_name = (
        canonicalize_env_name(model_source.get("env_name"))
        or canonicalize_env_name(getattr(args, "env_name", None))
        or canonicalize_env_name(resolved_profile.env_name)
        or infer_env_name_from_strings(
            model_label,
            *model_dirs,
            primary_dir,
        )
    )
    resolved_display_name = model_source.get("display_name") or infer_algorithm_display_name(
        model_label,
        *model_dirs,
        primary_dir,
    )

    print(f"\n{'=' * 88}")
    print(f"[{folder_index}/{total_folders}] Processing model: {model_label}")
    if len(model_dirs) > 1:
        print(f"Merged rollout folders: {len(model_dirs)}")
        for path in model_dirs:
            print(f"  - {path}")
    print(f"{'=' * 88}")

    loaded = load_folder_data(
        model_dirs=model_dirs,
        threshold_x=args.threshold_x,
        angle_threshold=args.angle_threshold,
        env_name=resolved_env_name,
        profile=resolved_profile,
        verbose=True,
    )
    if loaded is None:
        return None
    all_data, valid_data = loaded
    summary_data, summary_valid_data = prepare_summary_data_for_profile(
        resolved_profile,
        all_data=all_data,
        threshold_x=args.threshold_x,
        angle_threshold=args.angle_threshold,
        env_name=resolved_env_name,
    )

    folder_summary = build_folder_summary(
        model_label=model_label,
        all_data=summary_data,
        valid_data=summary_valid_data,
        threshold_x=args.threshold_x,
        angle_threshold=args.angle_threshold,
        env_name=resolved_env_name,
        display_name=resolved_display_name,
        algorithm_name=model_source.get("algorithm_name"),
        metric_definitions=resolved_profile.metric_definitions,
        profile=resolved_profile,
    )

    exported_paths: List[str] = []
    fig: Optional[plt.Figure] = None
    summary_only_threshold = int(getattr(args, "summary_only_threshold", DEFAULT_SUMMARY_ONLY_TRAJECTORY_THRESHOLD) or 0)
    auto_summary_only = summary_only_threshold > 0 and len(all_data) > summary_only_threshold
    if auto_summary_only:
        print(
            "High-parallel rollout detected: "
            f"{len(all_data)} trajectories > {summary_only_threshold}; "
            "skipping per-trajectory figures and keeping summary outputs only."
        )
    if not args.summary_only and not auto_summary_only:
        export_stem = (
            args.output_name
            if os.path.abspath(output_dir) == os.path.abspath(primary_dir)
            else make_export_stem(model_label=model_label, output_name=args.output_name)
        )
        if args.split_panels:
            exported_paths = export_split_panel_figures(
                model_label=model_label,
                all_data=all_data,
                valid_data=valid_data,
                threshold_x=args.threshold_x,
                angle_threshold=args.angle_threshold,
                output_dir=output_dir,
                export_stem=export_stem,
                dpi=args.dpi,
                save_pdf=args.save_pdf,
                figure_width=args.figure_width,
                figure_height=args.figure_height,
                show_run_label=args.show_run_label,
                profile=resolved_profile,
                max_plot_trajectories=args.max_plot_trajectories,
                env_name=resolved_env_name,
            )
            if not args.no_display:
                fig = create_figure(
                    model_label=model_label,
                    all_data=all_data,
                    valid_data=valid_data,
                    threshold_x=args.threshold_x,
                    angle_threshold=args.angle_threshold,
                    figure_width=args.figure_width,
                    figure_height=args.figure_height,
                    show_run_label=args.show_run_label,
                    profile=resolved_profile,
                    max_plot_trajectories=args.max_plot_trajectories,
                    env_name=resolved_env_name,
                )
        else:
            fig = create_figure(
                model_label=model_label,
                all_data=all_data,
                valid_data=valid_data,
                threshold_x=args.threshold_x,
                angle_threshold=args.angle_threshold,
                figure_width=args.figure_width,
                figure_height=args.figure_height,
                show_run_label=args.show_run_label,
                profile=resolved_profile,
                max_plot_trajectories=args.max_plot_trajectories,
                env_name=resolved_env_name,
            )
            exported_paths = export_figure(
                fig=fig,
                output_dir=output_dir,
                export_stem=export_stem,
                dpi=args.dpi,
                save_pdf=args.save_pdf,
            )

    for path in exported_paths:
        print(f"Saved figure: {path}")

    return exported_paths, fig, folder_summary


def position_windows(figures: Sequence[plt.Figure], titles: Sequence[str]) -> None:
    if not figures:
        return

    screen_width, screen_height = get_screen_size()
    window_width = max(700, screen_width // 2)
    window_height = max(500, screen_height // 2)

    print(f"Detected screen size: {screen_width}x{screen_height}")
    print(f"Display window size: {window_width}x{window_height}")

    for index, (fig, title) in enumerate(zip(figures, titles)):
        manager = fig.canvas.manager
        try:
            manager.set_window_title(title)
        except Exception:
            pass

        try:
            col = index % 2
            row = index // 2
            x_pos = col * window_width
            y_pos = row * window_height
            manager.window.wm_geometry(f"{window_width}x{window_height}+{x_pos}+{y_pos}")
        except Exception:
            continue


def main(
    *,
    profile: Optional[RenderProfile] = None,
    default_env_name: Optional[str] = None,
    default_input_dir: str = DEFAULT_INPUT_DIR,
    description: str = DEFAULT_RENDER_DESCRIPTION,
) -> None:
    resolved_profile = resolve_render_profile(profile)
    apply_publication_style(resolved_profile)
    parser = build_parser(
        profile=resolved_profile,
        description=description,
        default_input_dir=default_input_dir,
        default_env_name=default_env_name,
    )
    args = parser.parse_args()
    set_panel_titles_visible(not bool(getattr(args, "hide_panel_titles", False)))

    print(f"Input directory: {args.input_dir}")
    if args.env_name:
        print(f"Environment hint: {args.env_name}")
    if args.all:
        print("Mode: process all rollout folders.")
    else:
        print(f"Mode: process latest {args.num_folders} rollout folders.")
    print(
        "Figure config: "
        f"{args.figure_width:.2f} in x {args.figure_height:.2f} in, "
        f"{args.dpi} dpi, threshold_x={args.threshold_x:.1f} m, "
        f"angle_threshold={args.angle_threshold:.1f} deg"
    )
    output_dir = resolve_output_dir(input_dir=args.input_dir, output_dir=args.output_dir)
    print(f"Output directory: {output_dir}")
    if args.summary_only:
        print("Export mode: summary-only (one cross-model figure).")
    elif args.split_panels:
        print("Per-model export mode: split panels into subdirectories.")
    else:
        print("Per-model export mode: single combined figure.")
    if args.skip_summary_table:
        print("Summary-table export disabled.")

    model_sources = resolve_model_sources(
        input_dir=args.input_dir,
        extra_input_dirs=args.extra_input_dirs,
        model_label=args.model_label,
        num_folders=args.num_folders,
        process_all=args.all,
    )

    generated_outputs: List[List[str]] = []
    figures: List[plt.Figure] = []
    summary_records: List[Dict[str, object]] = []
    window_titles: List[str] = []

    print(f"\n{'=' * 88}")
    print(f"Start processing {len(model_sources)} model source(s)")
    print(f"{'=' * 88}")

    for index, model_source in enumerate(model_sources, start=1):
        try:
            result = process_folder(
                model_source=model_source,
                folder_index=index,
                total_folders=len(model_sources),
                output_dir=output_dir,
                args=args,
                profile=resolved_profile,
            )
            if result is None:
                continue
            exported_paths, fig, folder_summary = result
            if exported_paths:
                generated_outputs.append(exported_paths)
            summary_records.append(folder_summary)
            if fig is not None:
                figures.append(fig)
                window_titles.append(
                    f"Figure {index}/{len(model_sources)} - {model_source['label']}"
                )
        except Exception as exc:
            print(f"Error while processing {model_source['label']}: {exc}")
            import traceback

            traceback.print_exc()

    summary_paths: List[str] = []
    summary_mean_plot_paths: List[str] = []
    if not args.skip_summary_table:
        summary_output_dir = resolve_summary_output_dir(args.input_dir, args.output_dir, output_dir)
        summary_paths = export_summary_table_figure(
            summary_records=summary_records,
            output_dir=summary_output_dir,
            output_name=args.output_name,
            dpi=args.dpi,
            save_pdf=args.save_pdf,
            profile=resolved_profile,
        )
        summary_mean_plot_paths = export_summary_mean_radar_figure(
            summary_records=summary_records,
            output_dir=summary_output_dir,
            output_name=args.output_name,
            dpi=args.dpi,
            save_pdf=args.save_pdf,
            profile=resolved_profile,
        )
    for path in summary_paths:
        print(f"Saved cross-figure summary table: {path}")
    for path in summary_mean_plot_paths:
        print(f"Saved cross-figure summary radar: {path}")

    print(f"\n{'=' * 88}")
    print("Processing complete")
    print(f"{'=' * 88}")
    total_generated = sum(len(paths) for paths in generated_outputs) + len(summary_paths) + len(summary_mean_plot_paths)
    print(f"Generated {total_generated} file(s):")
    for index, paths in enumerate(generated_outputs, start=1):
        joined_paths = ", ".join(paths)
        print(f"  {index}. {joined_paths}")
    if summary_paths:
        print(f"  summary. {', '.join(summary_paths)}")
    if summary_mean_plot_paths:
        print(f"  summary-radar. {', '.join(summary_mean_plot_paths)}")
    print(f"{'=' * 88}")

    if not args.no_display and figures:
        position_windows(figures, window_titles)
        plt.show()
        print("All figure windows closed.")
    else:
        for fig in figures:
            plt.close(fig)
        if args.no_display:
            print("Skipped interactive display because --no-display was used.")

    print("Program finished.")


if __name__ == "__main__":
    main()
