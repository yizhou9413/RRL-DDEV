from render5 import (
    AnglePanelConfig,
    FigureLayoutConfig,
    MetricsTableConfig,
    RenderProfile,
    SummaryMeanRadarConfig,
    SummaryTableConfig,
    SurvivalPanelConfig,
    TrajectoryErrorPanelConfig,
    apply_publication_style_for_profile,
    build_parser_for_profile,
    export_summary_mean_radar_figure_for_profile,
    export_summary_table_figure_for_profile,
    get_split_panel_filenames as _get_split_panel_filenames,
    main_for_profile,
    summarize_model_source_for_profile,
)


DISTANCE_XLIM = (0.0, 90.0)
MOOSE_SUMMARY_METRIC_ORDER = (
    "max_phi",
    "max_theta",
    "max_beta",
    "rms_phi",
    "rms_theta",
    "rms_beta",
    "traj_dev_rmse",
    "traj_dev_max",
)
MOOSE_SUMMARY_MEAN_RADAR_RANGES = {
    "max_phi": (0.0, 5.0),
    "max_beta": (0.0, 10.0),
    "max_theta": (0.0, 5.0),
    "rms_phi": (0.0, 1.0),
    "rms_beta": (0.0, 5.0),
    "rms_theta": (0.0, 1.0),
    "traj_dev_rmse": (0.0, 0.2),
    "traj_dev_max": (0.0, 0.4),
}


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


def _signed_trajectory_error_axis_ticks(ax, context):
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_yticklabels(["-0.5", "0.0", "0.5"])


PROFILE = RenderProfile(
    env_name="moose",
    default_input_dir="/home/zhouyi/lane-change/runs/a2c_continuous_moose_default",
    description="Render publication-quality trajectory analysis figures. (moose)",
    summary_x_range=DISTANCE_XLIM,
    survival_panel=SurvivalPanelConfig(
        stem="a_survival_distance",
        panel_label="(a)",
        title="Survival distance",
        xlabel="Longitudinal distance (m)",
        ylabel="Trajectory rank",
        axis_postprocessor=_survival_y_axis_counts_from_one,
        xlim=DISTANCE_XLIM,
        reference_line_x=90.0,
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
            xlabel="Distance (m)",
            x_key="x",
            xlim=DISTANCE_XLIM,
            ylim=(-10.0, 10.0),
            threshold_value=5.0,
        ),
        AnglePanelConfig(
            "c_pitch_angle",
            "theta",
            r"$\theta$ (deg)",
            "(c)",
            "Pitch angle",
            xlabel="Distance (m)",
            x_key="x",
            xlim=DISTANCE_XLIM,
            ylim=(-10.0, 10.0),
            threshold_value=5.0,
        ),
        AnglePanelConfig(
            "d_sideslip_angle",
            "beta",
            r"$\beta$ (deg)",
            "(d)",
            "Sideslip angle",
            xlabel="Distance (m)",
            x_key="x",
            xlim=DISTANCE_XLIM,
        ),
    ),
    metrics_table=MetricsTableConfig(
        panel_label="(e)",
        title="Trajectory-level angle summary",
    ),
    signed_trajectory_error_panel=TrajectoryErrorPanelConfig(
        stem="f_signed_trajectory_error",
        panel_label="(f)",
        title="Trajectory error",
        ylabel=r"$e_{\mathrm{traj}}$ (m)",
        axis_postprocessor=_signed_trajectory_error_axis_ticks,
        ylim=(-0.6, 0.6),
        use_absolute_value=False,
        reference_line_values=(-0.5, 0.5),
    ),
    summary_table=SummaryTableConfig(
        title="Cross-model test summary",
        metric_order=MOOSE_SUMMARY_METRIC_ORDER,
    ),
    summary_mean_radar=SummaryMeanRadarConfig(
        title="Summary mean radar",
        subtitle="Mean values normalized as algorithm / PPO; radial range is 0 to 2.",
        ranges=MOOSE_SUMMARY_MEAN_RADAR_RANGES,
        metric_order=MOOSE_SUMMARY_METRIC_ORDER,
        normalized_min_radius=0.0,
        expand_ranges_to_data=False,
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
