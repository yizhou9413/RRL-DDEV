import numpy as np

from render5 import (
    AnglePanelConfig,
    FigureLayoutConfig,
    MetricsTableConfig,
    RenderProfile,
    SUMMARY_MEAN_RADAR_RANGES,
    SummaryMeanRadarConfig,
    SummaryTableConfig,
    SurvivalPanelConfig,
    TrajectoryErrorPanelConfig,
    apply_publication_style_for_profile,
    build_parser_for_profile,
    compute_time_series,
    export_summary_mean_radar_figure_for_profile,
    export_summary_table_figure_for_profile,
    get_split_panel_filenames as _get_split_panel_filenames,
    main_for_profile,
    metric_from_data,
    summarize_model_source_for_profile,
)


FIXED_CIRCLE_TIME_XLIM = (0.0, 10.0)
FIXED_CIRCLE_BETA_TARGET_RANGE_RAD = (-1.0, -0.4)
FIXED_CIRCLE_TDS_TARGET_DEG = -20.0
FIXED_CIRCLE_SUMMARY_METRIC_ORDER = (
    "max_phi",
    "max_theta",
    "max_beta",
    "rms_phi",
    "rms_theta",
    "rms_beta",
    "traj_dev_rmse",
    "traj_dev_max",
)


def _first_beta_reach_time(data):
    beta = np.asarray(data["beta"], dtype=float)
    time = compute_time_series(data)
    finite = np.isfinite(beta) & np.isfinite(time)
    if not np.any(finite):
        return np.nan

    finite_indices = np.flatnonzero(finite)
    threshold = FIXED_CIRCLE_TDS_TARGET_DEG
    first_index = int(finite_indices[0])
    if beta[first_index] <= threshold:
        return float(time[first_index])

    previous_index = first_index
    for index in finite_indices[1:]:
        index = int(index)
        previous_beta = float(beta[previous_index])
        current_beta = float(beta[index])
        if current_beta <= threshold:
            previous_time = float(time[previous_index])
            current_time = float(time[index])
            denominator = current_beta - previous_beta
            if denominator == 0.0:
                return current_time
            fraction = (threshold - previous_beta) / denominator
            fraction = float(np.clip(fraction, 0.0, 1.0))
            return previous_time + fraction * (current_time - previous_time)
        previous_index = index

    return float(time[int(finite_indices[-1])])


def _beta_target_range_rms(data):
    beta = np.radians(np.asarray(data["beta"], dtype=float))
    lower_target, upper_target = FIXED_CIRCLE_BETA_TARGET_RANGE_RAD
    error = np.zeros_like(beta, dtype=float)
    error[beta > upper_target] = beta[beta > upper_target] - upper_target
    error[beta < lower_target] = beta[beta < lower_target] - lower_target
    finite_error = error[np.isfinite(error)]
    if finite_error.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(finite_error))))


FIXED_CIRCLE_METRIC_DEFINITIONS = (
    ("max_beta", "TDS", _first_beta_reach_time),
    ("max_phi", r"$\max(|\phi|)$", lambda data: metric_from_data(data, "max_phi", "phi", "max_abs")),
    ("max_theta", r"$\max(|\theta|)$", lambda data: metric_from_data(data, "max_theta", "theta", "max_abs")),
    ("rms_beta", r"$\mathrm{RMS}(e_\beta)$", _beta_target_range_rms),
    ("rms_phi", r"$\mathrm{RMS}(\phi)$", lambda data: metric_from_data(data, "rms_phi", "phi", "rms")),
    ("rms_theta", r"$\mathrm{RMS}(\theta)$", lambda data: metric_from_data(data, "rms_theta", "theta", "rms")),
)
FIXED_CIRCLE_SUMMARY_MEAN_RADAR_RANGES = dict(SUMMARY_MEAN_RADAR_RANGES)
FIXED_CIRCLE_SUMMARY_MEAN_RADAR_RANGES.update(
    {
        "max_beta": FIXED_CIRCLE_TIME_XLIM,
        "rms_beta": (0.0, 0.4),
    }
)


def _survival_y_axis_counts_from_one(ax, context):
    count = len(context.get("survival_distances", ()))
    if count <= 0:
        return

    ax.set_ylim(count + 0.5, 0.5)
    if count <= 15:
        ticks = list(range(1, count + 1))
    else:
        ticks = sorted(
            {
                max(1, min(count, round(1 + i * (count - 1) / 5)))
                for i in range(6)
            }
        )
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(tick) for tick in ticks])


def _fixed_circle_sideslip_axis_limits(ax, context):
    ax.set_ylim(-40.0, 20.0)


def _fixed_circle_signed_error_axis_ticks(ax, context):
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_yticklabels(["-0.5", "0.0", "0.5"])


PROFILE = RenderProfile(
    env_name="fixed_circle_iwd",
    default_input_dir="/home/zhouyi/lane-change/runs/a2c_continuous_fixed_circle_iwd_default",
    description="Render publication-quality trajectory analysis figures. (fixed_circle_iwd)",
    summary_time_range=FIXED_CIRCLE_TIME_XLIM,
    metric_definitions=FIXED_CIRCLE_METRIC_DEFINITIONS,
    summary_metric_units={
        "max_beta": "s",
        "rms_beta": "rad",
    },
    metrics_use_successful_trajectories_only=True,
    survival_panel=SurvivalPanelConfig(
        stem="a_survival_distance",
        panel_label="(a)",
        title="Survival time",
        xlabel="Time (s)",
        ylabel="Trajectory rank",
        axis_postprocessor=_survival_y_axis_counts_from_one,
        xlim=FIXED_CIRCLE_TIME_XLIM,
        reference_line_x=10.0,
        show_reference_line=False,
        show_threshold_label=False,
        show_stats_box=False,
    ),
    angle_panels=(
        AnglePanelConfig(
            "b_roll_angle",
            "phi",
            r"$\phi$ (deg)",
            "(b)",
            "Roll angle",
            False,
            xlabel="Time (s)",
            x_key="time",
            xlim=FIXED_CIRCLE_TIME_XLIM,
            ylim=(-10.0, 10.0),
            threshold_value=5.0,
        ),
        AnglePanelConfig(
            "c_pitch_angle",
            "theta",
            r"$\theta$ (deg)",
            "(c)",
            "Pitch angle",
            xlabel="Time (s)",
            x_key="time",
            xlim=FIXED_CIRCLE_TIME_XLIM,
            ylim=(-10.0, 10.0),
            threshold_value=5.0,
        ),
        AnglePanelConfig(
            "d_sideslip_angle",
            "beta",
            r"$\beta$ (deg)",
            "(d)",
            "Sideslip angle",
            xlabel="Time (s)",
            axis_postprocessor=_fixed_circle_sideslip_axis_limits,
            x_key="time",
            xlim=FIXED_CIRCLE_TIME_XLIM,
            show_threshold_lines=False,
            failure_lower=-45.0,
            failure_upper=15.0,
        ),
    ),
    metrics_table=MetricsTableConfig(
        panel_label="(e)",
        title="Trajectory-level angle summary",
        sample_label="Successful trajectories",
    ),
    signed_trajectory_error_panel=TrajectoryErrorPanelConfig(
        stem="f_signed_trajectory_error",
        panel_label="(f)",
        title="Trajectory error",
        xlabel="Time (s)",
        ylabel=r"$e_{\mathrm{traj}}$ (m)",
        axis_postprocessor=_fixed_circle_signed_error_axis_ticks,
        x_key="time",
        xlim=FIXED_CIRCLE_TIME_XLIM,
        ylim=(-0.6, 0.6),
        use_absolute_value=False,
        reference_line_values=(-0.5, 0.5),
    ),
    summary_table=SummaryTableConfig(
        title="Cross-model test summary",
        metric_order=FIXED_CIRCLE_SUMMARY_METRIC_ORDER,
        note_with_trajectory_deviation=(
            "Summary metrics use the 0-10 s window. "
            "Failed trajectories are excluded from metric means. "
            "TDS is seconds; RMS(e_beta) is radians; other angle metrics are degrees; "
            "trajectory deviation metrics are meters."
        ),
        note_without_trajectory_deviation=(
            "Summary metrics use the 0-10 s window. "
            "Failed trajectories are excluded from metric means. "
            "TDS is seconds; RMS(e_beta) is radians; other angle metrics are degrees."
        ),
    ),
    summary_mean_radar=SummaryMeanRadarConfig(
        title="Summary mean radar",
        subtitle=(
            "Mean values from the 0-10 s summary window normalized as algorithm / PPO."
        ),
        ranges=FIXED_CIRCLE_SUMMARY_MEAN_RADAR_RANGES,
        metric_order=FIXED_CIRCLE_SUMMARY_METRIC_ORDER,
    ),
    layout=FigureLayoutConfig(),
)


def build_parser():
    return build_parser_for_profile(PROFILE)


def summarize_model_source(model_source, args):
    return summarize_model_source_for_profile(PROFILE, model_source, args)


def export_summary_table_figure(summary_records, output_dir, output_name, dpi, save_pdf):
    return export_summary_table_figure_for_profile(
        PROFILE,
        summary_records,
        output_dir,
        output_name,
        dpi,
        save_pdf,
    )


def export_summary_mean_radar_figure(summary_records, output_dir, output_name, dpi, save_pdf):
    return export_summary_mean_radar_figure_for_profile(
        PROFILE,
        summary_records,
        output_dir,
        output_name,
        dpi,
        save_pdf,
    )


def apply_publication_style() -> None:
    apply_publication_style_for_profile(PROFILE)


def get_split_panel_filenames():
    return _get_split_panel_filenames(PROFILE)


def main() -> None:
    main_for_profile(PROFILE)


if __name__ == "__main__":
    main()
