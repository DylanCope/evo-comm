from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot(data):
    set_plotting_style()
    data = data[data.total_env_steps <= 5e6]
    sns.lineplot(
        data=data,
        x="total_env_steps",
        y="mean_total_reward",
        hue="overlap_mode",
        errorbar="se",
    )
    plt.xlabel("Steps")
    plt.ylabel("Mean total reward")
    plt.legend(title="Overlap Mode")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/mappo_reward_curves_with_partial").glob(
            "data_*.csv"
        )
    ]
    plot(*data)
    plt.savefig(
        "figures/mappo_reward_curves_with_partial/mappo_reward_curves_with_partial.pdf",
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    reproduce_figure()
