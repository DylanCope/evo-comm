from matplotlib.artist import _cm_set
from matplotlib.artist import _different_canvas
from matplotlib.artist import _fully_clipped_to_axes
from matplotlib.artist import _internal_update
from matplotlib.artist import _set_alpha_for_array
from matplotlib.artist import _set_gc_clip
from matplotlib.artist import _update_props
from matplotlib.artist import add_callback
from matplotlib.artist import convert_xunits
from matplotlib.artist import convert_yunits
from matplotlib.artist import findobj
from matplotlib.artist import format_cursor_data
from matplotlib.artist import get_agg_filter
from matplotlib.artist import get_alpha
from matplotlib.artist import get_animated
from matplotlib.artist import get_children
from matplotlib.artist import get_clip_box
from matplotlib.artist import get_clip_on
from matplotlib.artist import get_clip_path
from matplotlib.artist import get_cursor_data
from matplotlib.artist import get_figure
from matplotlib.artist import get_gid
from matplotlib.artist import get_in_layout
from matplotlib.artist import get_label
from matplotlib.artist import get_mouseover
from matplotlib.artist import get_path_effects
from matplotlib.artist import get_picker
from matplotlib.artist import get_rasterized
from matplotlib.artist import get_sketch_params
from matplotlib.artist import get_snap
from matplotlib.artist import get_tightbbox
from matplotlib.artist import get_transform
from matplotlib.artist import get_transformed_clip_path_and_affine
from matplotlib.artist import get_url
from matplotlib.artist import get_visible
from matplotlib.artist import get_zorder
from matplotlib.artist import have_units
from matplotlib.artist import is_transform_set
from matplotlib.artist import pchanged
from matplotlib.artist import pick
from matplotlib.artist import pickable
from matplotlib.artist import properties
from matplotlib.artist import remove
from matplotlib.artist import remove_callback
from matplotlib.artist import set
from matplotlib.artist import set_agg_filter
from matplotlib.artist import set_alpha
from matplotlib.artist import set_animated
from matplotlib.artist import set_clip_box
from matplotlib.artist import set_clip_on
from matplotlib.artist import set_clip_path
from matplotlib.artist import set_figure
from matplotlib.artist import set_gid
from matplotlib.artist import set_in_layout
from matplotlib.artist import set_label
from matplotlib.artist import set_mouseover
from matplotlib.artist import set_path_effects
from matplotlib.artist import set_rasterized
from matplotlib.artist import set_sketch_params
from matplotlib.artist import set_snap
from matplotlib.artist import set_url
from matplotlib.artist import set_visible
from matplotlib.artist import set_zorder
from matplotlib.artist import update
from matplotlib.lines import Line2D
from matplotlib.lines import _get_markerfacecolor
from matplotlib.lines import _get_transformed_path
from matplotlib.lines import _set_markercolor
from matplotlib.lines import _transform_path
from matplotlib.lines import contains
from matplotlib.lines import draw
from matplotlib.lines import get_aa
from matplotlib.lines import get_antialiased
from matplotlib.lines import get_bbox
from matplotlib.lines import get_c
from matplotlib.lines import get_color
from matplotlib.lines import get_dash_capstyle
from matplotlib.lines import get_dash_joinstyle
from matplotlib.lines import get_data
from matplotlib.lines import get_drawstyle
from matplotlib.lines import get_ds
from matplotlib.lines import get_fillstyle
from matplotlib.lines import get_gapcolor
from matplotlib.lines import get_linestyle
from matplotlib.lines import get_linewidth
from matplotlib.lines import get_ls
from matplotlib.lines import get_lw
from matplotlib.lines import get_marker
from matplotlib.lines import get_markeredgecolor
from matplotlib.lines import get_markeredgewidth
from matplotlib.lines import get_markerfacecolor
from matplotlib.lines import get_markerfacecoloralt
from matplotlib.lines import get_markersize
from matplotlib.lines import get_markevery
from matplotlib.lines import get_mec
from matplotlib.lines import get_mew
from matplotlib.lines import get_mfc
from matplotlib.lines import get_mfcalt
from matplotlib.lines import get_ms
from matplotlib.lines import get_path
from matplotlib.lines import get_pickradius
from matplotlib.lines import get_solid_capstyle
from matplotlib.lines import get_solid_joinstyle
from matplotlib.lines import get_window_extent
from matplotlib.lines import get_xdata
from matplotlib.lines import get_xydata
from matplotlib.lines import get_ydata
from matplotlib.lines import is_dashed
from matplotlib.lines import recache
from matplotlib.lines import recache_always
from matplotlib.lines import set_aa
from matplotlib.lines import set_antialiased
from matplotlib.lines import set_c
from matplotlib.lines import set_color
from matplotlib.lines import set_dash_capstyle
from matplotlib.lines import set_dash_joinstyle
from matplotlib.lines import set_dashes
from matplotlib.lines import set_data
from matplotlib.lines import set_drawstyle
from matplotlib.lines import set_ds
from matplotlib.lines import set_fillstyle
from matplotlib.lines import set_gapcolor
from matplotlib.lines import set_linestyle
from matplotlib.lines import set_linewidth
from matplotlib.lines import set_ls
from matplotlib.lines import set_lw
from matplotlib.lines import set_marker
from matplotlib.lines import set_markeredgecolor
from matplotlib.lines import set_markeredgewidth
from matplotlib.lines import set_markerfacecolor
from matplotlib.lines import set_markerfacecoloralt
from matplotlib.lines import set_markersize
from matplotlib.lines import set_markevery
from matplotlib.lines import set_mec
from matplotlib.lines import set_mew
from matplotlib.lines import set_mfc
from matplotlib.lines import set_mfcalt
from matplotlib.lines import set_ms
from matplotlib.lines import set_picker
from matplotlib.lines import set_pickradius
from matplotlib.lines import set_solid_capstyle
from matplotlib.lines import set_solid_joinstyle
from matplotlib.lines import set_transform
from matplotlib.lines import set_xdata
from matplotlib.lines import set_ydata
from matplotlib.lines import update_from
from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot_reward_and_mimicry_frequency(adjusted_df):
    set_plotting_style()

    _, ax = plt.subplots(figsize=(10, 6))
    twin_ax = ax.twinx()

    sns.lineplot(
        data=adjusted_df,
        x="adjusted_iteration",
        y="mce/use_overlap_freq",
        color=sns.color_palette()[1],
        ax=twin_ax,
    )

    sns.lineplot(
        data=adjusted_df,
        x="adjusted_iteration",
        y="performance/mean_total_reward",
        ax=ax,
    )

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean total reward")
    twin_ax.set_ylabel("Mimicry Frequency")

    ax.grid(False)
    twin_ax.grid(False)

    custom_lines = [
        Line2D([0], [0], color=sns.color_palette()[0], lw=4),
        Line2D([0], [0], color=sns.color_palette()[1], lw=4),
    ]

    twin_ax.legend(custom_lines, ["Reward", "Mimicry Frequency"], loc="lower right")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/reward_and_mimicry_frequency").glob("data_*.csv")
    ]
    plot_reward_and_mimicry_frequency(*data)
    plt.savefig(
        "figures/reward_and_mimicry_frequency/reward_and_mimicry_frequency.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()
