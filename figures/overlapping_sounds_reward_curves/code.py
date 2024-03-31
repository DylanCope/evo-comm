from pathlib import Path
from reproducible_figures.plotting import set_plotting_style
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matplotlib.use("pdf")


def plot(data):
    set_plotting_style()
    sns.lineplot(
        data=data,
        x="total_env_steps",
        y="mean_total_reward",
        hue="N_OVERLAPPING_SOUNDS",
        palette=sns.color_palette("mako_r", 3),
    )
    plt.xlabel("Steps")
    plt.ylabel("Mean total reward")
    plt.legend(title="Overlapping\nSounds")


def reproduce_figure():
    data = [
        pd.read_csv(csv_path)
        for csv_path in Path("figures/overlapping_sounds_reward_curves").glob(
            "data_*.csv"
        )
    ]
    plot(*data)
    plt.savefig(
        "figures/overlapping_sounds_reward_curves/overlapping_sounds_reward_curves.pdf",
        bbox_inches="tight",
        dpi=1000,
    )


if __name__ == "__main__":
    reproduce_figure()