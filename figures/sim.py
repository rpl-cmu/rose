from pathlib import Path

import matplotlib.pyplot as plt
import rose.jrl
import rose.plot as plot
from matplotlib.patches import ConnectionPatch, Rectangle


# https://stackoverflow.com/a/73308476
def indicate_inset(axin, axout, arrow_start=(0, 0), arrow_end=(0, 0)):
    (x0, x1), (y0, y1) = axin.get_xlim(), axin.get_ylim()
    width = x1 - x0
    height = y1 - y0

    rect = Rectangle(
        [x0, y0],
        width=width,
        height=height,
        transform=axout.transData,
        fc="none",
        ec="black",
    )
    axout.add_patch(rect)

    conn = ConnectionPatch(
        xyA=arrow_start,
        coordsA=rect.get_transform(),
        xyB=arrow_end,
        coordsB=axin.transAxes,
        arrowstyle="->",
        ec="black",
        lw=0.75,
    )
    fig.add_artist(conn)
    return rect, conn


if __name__ == "__main__":
    # ------------------------- Load all data ------------------------- #
    tags_to_names = plot.tags_to_names()
    result_files = [p for p in Path("figures/data").glob("*.jrr")]
    jrl_parser = rose.jrl.makeRoseParser()

    results = {}
    for p in result_files:
        values = jrl_parser.parseResults(str(p), False).robot_solutions["a"].values
        name = tags_to_names.get(p.stem, None)
        results[name] = plot.load_full_state(values)

    # ------------------------- Plot data ------------------------- #
    colors = plot.setup_plot()
    # c = sns.color_palette("colorblind")
    names_states = [
        r"$b$",
        r"$r_l$",
        r"$r_r$",
        r"$s_l$",
        r"$s_r$",
    ]
    unit_states = ["$m$", "$m$", "$m$", "$rad/s$", "$rad/s$"]
    names = [plot.WheelType.GT.value, plot.WheelType.WHEEL_ROSE_INTR_SLIP.value]

    fig = plt.figure(layout="constrained", figsize=(8, 1.5))
    ax = []
    ax.append(fig.add_subplot(161))
    ax.append(fig.add_subplot(162))
    ax.append(fig.add_subplot(163, sharey=ax[-1]))
    ax[-1].tick_params(labelleft=False)
    ax.append(fig.add_subplot(164))
    ax.append(fig.add_subplot(165, sharey=ax[-1]))
    ax[-1].tick_params(labelleft=False)
    ax.append(fig.add_subplot(166))

    # Plot all the main stuff
    for i, s in enumerate(names_states):
        ax[i].set_title(s + f" ({unit_states[i]})")
        ax[i].tick_params("y", pad=-2)
        ax[i].tick_params("x", pad=-2)
        ax[i].set_xticks([0, 250, 500])
        for n in names:
            ax[i].plot(results[n][s], c=colors[n], alpha=0.75)

    # Plot the zoomed in stuff
    for n in names:
        ax[-1].plot(results[n][names_states[-1]], label=n, c=colors[n], alpha=0.75)

    # Make zoomed in figure
    ax[-1].set_xlim(286, 296)
    ax[-1].set_ylim(-1, 8)
    ax[-1].set_xticks([])
    ax[-1].set_yticks([])
    indicate_inset(ax[-1], ax[-2], arrow_start=(0.5, 0.5), arrow_end=(0.0, 0.5))

    fig.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.2),
        fancybox=True,
        ncol=2,
    )

    # enlarge figure to include everything
    plt.savefig("figures/sim.png", bbox_inches="tight", dpi=300)
    plt.savefig("figures/sim.pdf", bbox_inches="tight", dpi=300)
    # plt.show()
