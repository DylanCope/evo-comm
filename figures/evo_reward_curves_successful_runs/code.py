from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot(data):
    set_plotting_style(font_scale=2)

    plt.figure(figsize=(7, 5))

    sns.lineplot(
        data=data,
        x="iteration",
        y="max_reward",
        hue="overlapping_sounds",
        errorbar="se",
        palette=[sns.color_palette()[1], sns.color_palette()[2]],
    )
    plt.xlabel("Generation")
    plt.ylabel("Population best mean reward")
    plt.legend(title="Signal\nOverlap?", fontsize=16)


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/evo_reward_curves_successful_runs").glob("data_*.csv")
    ]
    plot(*data)
    plt.savefig(
        "figures/evo_reward_curves_successful_runs/evo_reward_curves.pdf", bbox_inches="tight", dpi=1000
    )


if __name__ == "__main__":
    reproduce_figure()
