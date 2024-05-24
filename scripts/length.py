import argparse
import os
from pathlib import Path

import gtsam
import jrl
import numpy as np
import rose.jrl
from gtsam.symbol_shorthand import B, X
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


cbor = lambda s: "cbor" in str(s)

# python scripts/length.py --traj $(find data/ -mindepth 1 -type d -mtime -2)

if __name__ == "__main__":
    # For loading
    parser = argparse.ArgumentParser(description="Plot results of trajectory")
    parser.add_argument("--traj", type=Path, nargs="+", help="Result folders to plot")

    args = parser.parse_args()

    # ------------------------- Load gt values ------------------------- #
    jrl_parser = rose.jrl.makeRoseParser()

    trajs = {}
    if args.traj:
        print("Loading Trajectories...", end="")
        for f in sorted(args.traj):
            if f.is_dir():
                f = f / "_gt.jrr"

            results = jrl_parser.parseResults(str(f), cbor(f))
            result_name = f.parent

            print(f"\n\t{result_name}...", end="")

            values = results.robot_solutions["a"].values
            trajs[result_name] = values
            name = results.dataset_name

    print("\tDone!\n")

    # ------------------------- Compute ------------------------- #
    print("Computing length...")
    data = []
    for name, values in trajs.items():
        km = compute_traj_length(values) / 1000
        data.append([name, km])

    data = sorted(data, key=lambda x: x[1])

    print(tabulate(data, headers=["Run", "km"], tablefmt="github"))
    print()
