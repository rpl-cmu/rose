import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rose.plot as plot
import seaborn as sns


def plot_bars(file):
    colors = plot.setup_plot()
    tags_to_names = plot.tags_to_names()
    df = pd.read_pickle(f"figures/data/{file}.pkl")

    df["WheelFactor"] = df["WheelFactor"].replace(tags_to_names)

    variations = [
        plot.WheelType.SVO.value,
        plot.WheelType.WHEEL.value,
        plot.WheelType.WHEEL_PLANAR.value,
        plot.WheelType.WHEEL_ROSE.value,
        plot.WheelType.WHEEL_ROSE_INTR_SLIP.value,
    ]
    df = df[df["WheelFactor"].isin(variations)]

    fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(4.5, 2), dpi=147)
    sns.boxplot(
        df,
        hue="WheelFactor",
        y="ATEt",
        orient="v",
        ax=ax[0],
        hue_order=variations,
        legend=False,
        palette=colors,
    )
    ax[0].set_title("ATEt $(m)$")
    ax[0].set_ylabel("")
    ax[0].tick_params(axis="y", pad=-2)
    ax[0].set_ylim(0, 10)

    sns.boxplot(
        df,
        hue="WheelFactor",
        y="ATEr",
        orient="v",
        ax=ax[1],
        hue_order=variations,
        legend="brief",
        palette=colors,
    )
    ax[1].set_title("ATEr $(deg)$")
    ax[1].get_legend().remove()
    ax[1].set_ylabel("")
    ax[1].tick_params(axis="y", pad=-2)
    ax[1].set_ylim(0, 5)

    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.23))

    # plt.savefig(f"figures/{file}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"figures/{file}.pdf", bbox_inches="tight", dpi=300)
    # plt.show()


if __name__ == "__main__":
    plot_bars("sim_monte_carlo_ideal")
    plot_bars("sim_monte_carlo_real")
