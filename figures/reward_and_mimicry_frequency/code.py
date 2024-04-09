from matplotlib.lines import Line2D
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

    ax.set_xlabel("Adjusted Iteration")
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
