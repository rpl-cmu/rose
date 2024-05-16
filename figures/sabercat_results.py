import argparse
import os
from pathlib import Path

import gtsam
import jrl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import robust.jrl
import robust.plot as plot
import seaborn as sns
from gtsam.symbol_shorthand import X
from tabulate import tabulate

cbor = lambda s: "cbor" in str(s)

np.set_printoptions(suppress=True, precision=4)


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
    parser = argparse.ArgumentParser(description="Plot results of trajectory")
    parser.add_argument(
        "--align",
        action="store_true",
        help="Whether to align trajectories when computing error",
    )
    parser.add_argument(
        "--show", action="store_true", help="Whether to display figure at end"
    )
    parser.add_argument(
        "--filename",
        type=str,
        default="figures/sabercat_results.pdf",
        help="Where to save the resulting figure",
    )
    args = parser.parse_args()

    # ------------------------- Load gt values ------------------------- #
    jrl_parser = robust.jrl.makeRobustParser()

    trajs = {}

    folders = [Path("./data") / p.stem for p in Path("/mnt/bags/best/").iterdir()]

    traj_files = []
    for f in folders:
        traj_files.extend([p for p in f.iterdir() if p.suffix == ".jrr"])

    # ------------------------- Load all trajectories ------------------------- #
    if traj_files:
        print("Loading Trajectories...", end="")
        for f in sorted(traj_files):
            try:
                if "mono" in str(f):
                    continue

                results = jrl_parser.parseResults(str(f), cbor(f))
                result_name = f.stem

                traj_name = f.parent.stem
                if traj_name not in trajs:
                    trajs[traj_name] = {}

                trajs[traj_name][result_name] = results.robot_solutions["a"].values
                # print(f"\n\t{traj_name} - {result_name}...", end="")
            except:
                print(f"\n\tFailed {f}", end="")

    for k, v in trajs.items():
        trajs[k]["length"] = compute_traj_length(v["_gt"]) / 1000
    print("\tDone!\n")

    # ------------------------- Parse through to get the ones we want ------------------------- #

    out = []
    for k, v in trajs.items():
        out.append([k, v["length"]])

    print("The following passed the minimum requirements:")
    out = sorted(out, key=lambda x: x[1])
    print(tabulate(out, headers=["Trajectory", "Length"], tablefmt="github"))

    # ------------------------- Plot ------------------------- #
    print("Plotting ATE...")
    colors = plot.setup_plot()
    data = []
    for traj_name, runs in trajs.items():
        for name, values in runs.items():
            if name in ["_gt", "length"]:
                continue

            if name == "_wheel":
                has_imu = False
                wheel_type = "wheel_only"
            else:
                sensors = name[2:].split(".")
                wheel_type = [n for n in sensors if "wheel" in n]
                wheel_type = "cam_only" if len(wheel_type) == 0 else wheel_type[0]
                if "planar_prior" in sensors:
                    wheel_type += "_planar"
                has_imu = "imu" in sensors

            error = jrl.computeATEPose3(runs["_gt"], values, align=args.align)

            data.append(
                [
                    traj_name,
                    has_imu,
                    wheel_type,
                    error[0] / runs["length"],
                    error[1] * 180 / np.pi,
                    runs["length"],
                ]
            )

    df = pd.DataFrame(
        data,
        columns=["Trajectory", "Uses IMU", "Wheel Type", "ATEt / km", "ATEr", "Length"],
    )

    # Get rid of bad runs
    df = df[df["Wheel Type"] != "wheel_manifold"]
    df = df[df["Trajectory"] != "gps-denied-higher-res_2024-01-23-12-34-32"]

    # ------------------------- Plot ------------------------- #
    fig, ax = plt.subplots(1, 2, layout="constrained", figsize=(8, 2.5))

    # W/O IMU
    alias = {
        "wheel_cov": plot.WheelType.WHEEL_COV.value,
        "wheel_intr_slip": plot.WheelType.WHEEL_COV_INTR_SLIP.value,
        "wheel_dang": plot.WheelType.WHEEL_UNDER.value,
        "wheel_dang_planar": plot.WheelType.WHEEL_PLANAR.value,
        "cam_only": plot.WheelType.SVO.value,
        "wheel_only": plot.WheelType.WHEEL.value,
    }
    df_no_imu = df[df["Uses IMU"] == False].copy()
    df_no_imu["Wheel Type"] = df_no_imu["Wheel Type"].replace(alias)
    df_no_imu = df_no_imu[df_no_imu["Wheel Type"] != plot.WheelType.WHEEL_UNDER.value]
    df_no_imu["Wheel Type"] = pd.Categorical(
        df_no_imu["Wheel Type"],
        categories=[
            plot.WheelType.SVO.value,
            plot.WheelType.WHEEL.value,
            plot.WheelType.WHEEL_PLANAR.value,
            plot.WheelType.WHEEL_COV.value,
            plot.WheelType.WHEEL_COV_INTR_SLIP.value,
        ],
        ordered=True,
    )

    sns.swarmplot(
        df_no_imu,
        ax=ax[0],
        x="Wheel Type",
        y="ATEt / km",
        hue="Wheel Type",
        legend=False,
        size=4,
        palette=colors,
    )

    sns.lineplot(
        df_no_imu,
        ax=ax[0],
        x="Wheel Type",
        y="ATEt / km",
        hue="Trajectory",
        legend=False,
        size=0.2,
        palette=["k"],
        alpha=0.2,
    )
    ax[0].set_xlabel("")
    ax[0].set_title("Visual Wheel Odometry")
    ax[0].tick_params(axis="x", rotation=12, bottom=True)

    # With IMU
    alias = {k: v.replace("SVO", "SVIO") for k, v in alias.items()}
    df_with_imu = df[df["Uses IMU"] == True].copy()
    df_with_imu["Wheel Type"] = df_with_imu["Wheel Type"].replace(alias)
    df_with_imu = df_with_imu[
        df_with_imu["Wheel Type"] != plot.WheelType.WHEEL.value.replace("SVO", "SVIO")
    ]
    df_with_imu["Wheel Type"] = pd.Categorical(
        df_with_imu["Wheel Type"],
        categories=[
            plot.WheelType.SVO.value.replace("SVO", "SVIO"),
            plot.WheelType.WHEEL_UNDER.value.replace("SVO", "SVIO"),
            plot.WheelType.WHEEL_PLANAR.value.replace("SVO", "SVIO"),
            plot.WheelType.WHEEL_COV.value.replace("SVO", "SVIO"),
            plot.WheelType.WHEEL_COV_INTR_SLIP.value.replace("SVO", "SVIO"),
        ],
        ordered=True,
    )

    sns.swarmplot(
        df_with_imu,
        ax=ax[1],
        x="Wheel Type",
        y="ATEt / km",
        hue="Wheel Type",
        legend=False,
        size=4,
        palette={k.replace("SVO", "SVIO"): v for k, v in colors.items()},
    )
    sns.lineplot(
        df_with_imu,
        ax=ax[1],
        x="Wheel Type",
        y="ATEt / km",
        hue="Trajectory",
        legend=False,
        size=0.2,
        palette=["k"],
        alpha=0.2,
    )
    ax[1].set_xlabel("")
    ax[1].set_title("Visual Intertial Wheel Odometry")
    ax[1].tick_params(axis="x", rotation=12, bottom=True)
    ax[1].set_ylim([0, 200])

    ax[0].set_ylim(ax[1].get_ylim())

    # pivot = df_with_imu.pivot(
    #     index="Trajectory", columns=["Wheel Type"], values="ATEt / km"
    # )
    # print(pivot.to_markdown(tablefmt="github"))
    # print("\n\n\n")
    # print(pivot)

    if args.filename is not None:
        plt.savefig(args.filename, bbox_inches="tight", dpi=300)
    if args.show:
        plt.show()
