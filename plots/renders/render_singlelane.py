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


SINGLELANE_DISTANCE_XLIM = (0.0, 75.0)
SINGLELANE_SUMMARY_METRIC_ORDER = (
    "max_phi",
    "max_theta",
    "max_beta",
    "rms_phi",
    "rms_theta",
    "rms_beta",
    "traj_dev_rmse",
    "traj_dev_max",
)


def _signed_trajectory_error_axis_ticks(ax, context):
    ax.set_yticks([-0.5, 0.0, 0.5])
    ax.set_yticklabels(["-0.5", "0.0", "0.5"])


PROFILE = RenderProfile(
    env_name="singlelane",
    default_input_dir="/home/zhouyi/lane-change/runs/a2c_continuous_singlelane_default",
    description="Render publication-quality trajectory analysis figures. (singlelane)",
    default_threshold_x=75.0,
    summary_x_range=SINGLELANE_DISTANCE_XLIM,
    survival_panel=SurvivalPanelConfig(
        stem="a_survival_distance",
        panel_label="(a)",
        title="Survival distance",
        xlabel="Longitudinal distance (m)",
        ylabel="Trajectory rank",
        xlim=SINGLELANE_DISTANCE_XLIM,
        reference_line_x=75.0,
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
            xlabel="Longitudinal distance (m)",
            x_key="x",
            xlim=SINGLELANE_DISTANCE_XLIM,
            ylim=(-10.0, 10.0),
            threshold_value=5.0,
        ),
        AnglePanelConfig(
            "c_pitch_angle",
            "theta",
            r"$\theta$ (deg)",
            "(c)",
            "Pitch angle",
            xlabel="Longitudinal distance (m)",
            x_key="x",
            xlim=SINGLELANE_DISTANCE_XLIM,
            ylim=(-10.0, 10.0),
            threshold_value=5.0,
        ),
        AnglePanelConfig(
            "d_sideslip_angle",
            "beta",
            r"$\beta$ (deg)",
            "(d)",
            "Sideslip angle",
            xlabel="Longitudinal distance (m)",
            x_key="x",
            xlim=SINGLELANE_DISTANCE_XLIM,
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
        metric_order=SINGLELANE_SUMMARY_METRIC_ORDER,
    ),
    summary_mean_radar=SummaryMeanRadarConfig(
        title="Summary mean radar",
        metric_order=SINGLELANE_SUMMARY_METRIC_ORDER,
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
