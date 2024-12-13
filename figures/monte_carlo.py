import matplotlib.pyplot as plt
from matplotlib.ticker import Locator
import pandas as pd
import rose.plot as plot
import seaborn as sns
import numpy as np


class MinorSymLogLocator(Locator):
    """
    Dynamically find minor tick positions based on the positions of
    major ticks for a symlog scaling.
    """

    def __init__(self, linthresh):
        """
        Ticks will be placed between the major ticks.
        The placement is linear for x between -linthresh and linthresh,
        otherwise its logarithmically
        """
        self.linthresh = linthresh

    def __call__(self):
        "Return the locations of the ticks"
        majorlocs = self.axis.get_majorticklocs()

        # iterate through minor locs
        minorlocs = []

        # handle the lowest part
        for i in range(1, len(majorlocs)):
            majorstep = majorlocs[i] - majorlocs[i - 1]
            if abs(majorlocs[i - 1] + majorstep / 2) < self.linthresh:
                ndivs = 10
            else:
                ndivs = 9
            minorstep = majorstep / ndivs
            locs = np.arange(majorlocs[i - 1], majorlocs[i], minorstep)[1:]
            minorlocs.extend(locs)

        return self.raise_if_exceeds(np.array(minorlocs))

    def tick_values(self, vmin, vmax):
        raise NotImplementedError(
            "Cannot get tick locations for a " "%s type." % type(self)
        )


def plot_bars(file, rot_max, t_max):
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
    ax[0].set_ylim(0, t_max)

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
    ax[1].set_ylim(0, rot_max)

    if "real" in file:
        ax[0].set_yscale("symlog", linthresh=10, linscale=2.0)
        ax[0].set_yticks([0, 2, 4, 6, 8, 10, 100], [0, 2, 4, 6, 8, 10, r"$10^2$"])
        ax[0].get_ygridlines()[5].set_linestyle("--")
        ax[1].set_yscale("symlog", linthresh=5, linscale=2.0)
        ax[1].set_yticks([0, 1, 2, 3, 4, 5, 10, 100], [0, 1, 2, 3, 4, 5, 10, r"$10^2$"])
        ax[1].get_ygridlines()[5].set_linestyle("--")

    fig.legend(loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.23))

    plt.savefig(f"figures/{file}.png", bbox_inches="tight", dpi=300)
    plt.savefig(f"figures/{file}.pdf", bbox_inches="tight", dpi=300)
    # plt.show()


if __name__ == "__main__":
    # plot_bars("sim_monte_carlo_ideal", 5, 10)
    plot_bars("sim_monte_carlo_real", 100, 200)
