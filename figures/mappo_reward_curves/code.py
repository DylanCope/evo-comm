from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot(data):
    set_plotting_style(font_scale=2)
    data = data[data.total_env_steps <= 5e6]
    sns.lineplot(
        data=data,
        x="total_env_steps",
        y="mean_total_reward",
        hue="overlapping_sounds",
        errorbar="se",
    )
    plt.xlabel("Steps")
    plt.ylabel("Mean total reward")
    plt.legend(title="Signal Overlap?", fontsize=16)


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/mappo_reward_curves").glob("data_*.csv")
    ]
    plot(*data)
    plt.savefig(
        "figures/mappo_reward_curves/mappo_reward_curves.pdf",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    reproduce_figure()
