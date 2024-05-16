import gtsam
import jrl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import robust.plot
import seaborn as sns
import sympy as sy
from gtsam.symbol_shorthand import B, I, L, S
from robust.dataset import Dataset2JRL, Sensor, WheelNoise
from robust.sim import SimParameters, Simulation, symp
from tabulate import tabulate
from tqdm import tqdm, trange

np.set_printoptions(suppress=False, precision=3)


def compute_traj_length(values: gtsam.Values):
    poses = gtsam.utilities.allPose3s(values)
    keys = sorted(list(poses.keys()))
    dist = 0
    x_prev = poses.atPose3(keys[0])
    for k in keys[1:]:
        x_curr = poses.atPose3(k)
        dist += np.linalg.norm(x_curr.translation() - x_prev.translation())
        x_prev = x_curr

    return dist


if __name__ == "__main__":
    np.random.seed(0)
    params = SimParameters()

    params.time = 100
    params.num_feats = 50

    v = 2
    w = lambda t: np.cos(t * 1.5)
    sim = Simulation(params, np.pi / 6, v=2)
    sim.run_all(w, v)

    dataset = Dataset2JRL(sim)
    dataset.add_factors(Sensor.PRIOR, use_gt_orien=True)
    gt = dataset.get_ground_truth()

    # ------------------------- Plot 3D Grid ------------------------- #
    c = robust.plot.setup_plot()
    state = robust.plot.load_full_state(gt, dataset.N, pandas=False)

    # Load data
    x = np.linspace(-5, state[:, 3].max() + 5)
    y = np.linspace(-5, state[:, 4].max() + 5)
    X, Y = np.meshgrid(x, y)
    p = np.column_stack((X.flatten(), Y.flatten(), np.zeros(X.size)))
    Z = -sim.M(p.T).reshape(X.shape) + sim.M([0, 0, 0])

    minZ = Z.min()
    Z -= minZ

    print(state[:, 5].max() - state[:, 5].min())

    # Setup figure
    fig, ax = plt.subplots(
        subplot_kw={"projection": "3d"},
        figsize=(4, 2.7),
        dpi=300,
    )

    ax.plot_surface(
        X,
        Y,
        Z,
        alpha=0.5,
        lw=0.01,
        cmap=sns.color_palette("crest", as_cmap=True),
    )
    black = c[robust.plot.WheelType.GT.value]
    ax.plot(state[:, 3], state[:, 4], state[:, 5] - minZ, lw=3, c=black)

    # This changes the aspect ratio by changing the axis-box size (rather than  just the limits)
    # We set x/y equal, then manually change the z aspect
    ax.set_aspect("equal", adjustable="box")
    aspect = ax.get_box_aspect()
    aspect[-1] *= 20
    ax.set_box_aspect(aspect)

    ax.set_zticks([0.0, 1.0, 2.0])
    ax.set_yticks([0, 20, 40, 60, 80])

    ax.tick_params(axis="x", pad=-2)
    ax.tick_params(axis="y", pad=-2)
    ax.tick_params(axis="z", pad=-2)

    fig.subplots_adjust(left=0.0, right=0.92, top=1.3, bottom=-0.25)

    plt.savefig("figures/surface.pdf", dpi=300)
    plt.savefig("figures/surface.png", dpi=300)
    # plt.show()
