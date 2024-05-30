import argparse
import os

import gtsam
import numpy as np
import rose.jrl
from gtsam.symbol_shorthand import X
from rose.plot import (
    plot_3d_trajectory,
    plot_error_state,
    plot_state,
    plot_xy_trajectory,
)
from tabulate import tabulate


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


def get_subset(values: gtsam.Values, N: int):
    if N is None:
        return values

    values_pose3 = gtsam.Values()
    for i in range(N):
        values_pose3.insert(X(i), values.atPose3(X(i)))
    return values_pose3


def cbor(s):
    return "cbor" in s


if __name__ == "__main__":
    func = {
        "state": plot_state,
        "xy": plot_xy_trajectory,
        "3d": plot_3d_trajectory,
        "error": plot_error_state,
    }

    # For loading
    parser = argparse.ArgumentParser(description="Plot results of trajectory")
    parser.add_argument("--gt", type=str, help="Ground truth data to plot.")
    parser.add_argument("--traj", type=str, nargs="+", help="Result folders to plot")
    parser.add_argument(
        "--type",
        type=str,
        choices=list(func.keys()),
        default=list(func.keys())[0],
        help="Type of plot to generate",
    )
    parser.add_argument(
        "-n",
        "--iter",
        type=int,
        default=None,
        help="How many iterations to store. Defaults to all",
    )
    parser.add_argument(
        "--align",
        action="store_true",
        help="Whether to align trajectories when plotting",
    )

    # for plotting
    parser.add_argument(
        "--show", action="store_true", help="Whether to display figure at end"
    )
    parser.add_argument(
        "--filename", type=str, default=None, help="Where to save the resulting figure"
    )
    parser.add_argument(
        "-s",
        "--states",
        type=str,
        default="rpv",
        help="Which states to plot. Defaults to rpv",
    )

    # for computing
    parser.add_argument("--compute", action="store_true", help="Whether to compute ATE")
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Whether to normalized by length of trajectory",
    )

    args = parser.parse_args()

    # ------------------------- Load gt values ------------------------- #
    jrl_parser = rose.jrl.makeRoseParser()

    trajs = {}
    if args.gt:
        print("Loading GT...")
        dataset = jrl_parser.parseDataset(args.gt, cbor(args.gt))
        trajs["GT"] = dataset.groundTruth("a")
        name = dataset.name()

    # ------------------------- Load all trajectories ------------------------- #
    if args.traj:
        print("Loading Trajectories...", end="")
        for f in sorted(args.traj):
            results = jrl_parser.parseResults(f, cbor(f))

            result_name = os.path.splitext(os.path.basename(f))[0]

            if result_name == "_gt":
                result_name = "GT"
            elif result_name == "_wheel":
                result_name = "Wheel"
            print(f"\n\t{result_name}...", end="")

            values = results.robot_solutions["a"].values
            trajs[result_name] = values
            name = results.dataset_name

    print("\tDone!\n")

    # ------------------------- Compute ------------------------- #
    if args.compute:
        print("Computing GT length...", end="")
        km = compute_traj_length(get_subset(trajs["GT"], args.iter)) / 1000
        print(f"\t{km} km")

        km = km if args.norm else 1
        print("Computing ATE...")
        data = []
        for name, values in trajs.items():
            error = rose.jrl.computeATEPose3(
                trajs["GT"], get_subset(values, args.iter), align=args.align
            )
            data.append([name, error[0] / km, error[1] * 180 / np.pi])

        atet = "ATEt"
        if args.norm:
            atet += " / km"
        data = sorted(data, key=lambda x: x[1])
        data.insert(0, ["Run", atet, "ATEr"])

        print(tabulate(data, headers="firstrow", tablefmt="github"))
        print()

    # ------------------------- Plot ------------------------- #
    if args.show or args.filename:
        print("Plotting...", end="")
        func[args.type](
            figtitle=name,
            filename=args.filename,
            show=args.show,
            align=args.align,
            N=args.iter,
            states=args.states,
            **trajs,
        )
        print("\tDone!")
