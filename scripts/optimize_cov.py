import argparse
import multiprocessing
import os
from pathlib import Path
from typing import Optional

import gtsam
import jrl
import numpy as np
from rose.dataset import CamNoise, Dataset2JRL, Sensor, WheelNoise
from rose.flat import FlatDataset
from rose.kaist import KaistDataset
from rose.rose_python import (
    PlanarPriorTag,
    WheelRoseIntrSlipTag,
    WheelRoseIntrTag,
    WheelRoseSlipTag,
    WheelRoseTag,
    WheelBaselineTag,
    ZPriorTag,
    makeFrontend,
)
from rose.ros import GrizzlyBag
from scipy.optimize import basinhopping, brute, direct, dual_annealing, minimize, shgo
from tabulate import tabulate
from tqdm import tqdm

np.set_printoptions(suppress=False, precision=5, linewidth=400)


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

    # ------------------------- Parse Args ------------------------- #
    parser = argparse.ArgumentParser(description="Convert Kaist 2 jrl factor graph")
    parser.add_argument(
        "-k",
        "--kaist",
        type=Path,
        nargs="+",
        default=None,
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "-f",
        "--flat",
        type=Path,
        nargs="+",
        default=None,
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "-g",
        "--grizzly",
        type=Path,
        nargs="+",
        default=None,
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "--resave_grizzly",
        action="store_true",
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "-o",
        "--flat_out",
        type=Path,
        default=Path("/mnt/data/flat"),
        help="Where to convert datasets to.",
    )
    parser.add_argument(
        "-n",
        "--iter",
        type=int,
        default=None,
        help="How many iterations to store. Defaults to all",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use camera stash.",
    )
    args = parser.parse_args()

    # ------------------------- Set everything up ------------------------- #
    i = 0
    all_datasets = []
    if args.grizzly is not None:
        for g in args.grizzly:
            bag = GrizzlyBag(g)
            path = args.flat_out / bag.name

            if args.resave_grizzly or not path.exists():
                print(f"Saving grizzly bag to flat dataset at {path}...")
                bag.to_flat(args.flat_out)

            all_datasets.append(FlatDataset(path))

    if args.kaist is not None:
        for k in args.kaist:
            all_datasets.append(KaistDataset(k))

    elif args.flat is not None:
        for f in args.flat:
            all_datasets.append(FlatDataset(f))

    # ------------------------- Add camera factors ------------------------- #
    print("Adding stereo camera factors...")
    all_cams = []
    for dataset in all_datasets:
        cam = Dataset2JRL(dataset, args.iter)

        out_dir = f"data/{dataset.name}/"
        os.makedirs(out_dir, exist_ok=True)

        # First load up camera factors
        if not os.path.exists(out_dir + "cam.jrl") or args.no_cache:
            cam.add_factors(Sensor.CAM)
            cam.save_dataset(out_dir + "cam.jrl")
        else:
            cam.load_cache(out_dir + "cam.jrl")

        all_cams.append(cam)

    # ------------------------- Run everything ------------------------- #
    print("Beginning optimization...")

    def objective(x, arg=""):
        global i
        ates = [["Dataset", "IMU+Wheel", "Wheel"]]
        ate_total = 0
        for flat, cam in zip(all_datasets, all_cams):
            wheel_noise = flat.noise(Sensor.WHEEL)
            wheel_noise = vec2noise(x, wheel_noise)

            flat.add_noise(Sensor.WHEEL, wheel_noise)
            flat.clear_cache()

            data = Dataset2JRL(flat, args.iter)
            data.clear_factors()

            data.add_factors(Sensor.PRIOR)
            data.add_factors(Sensor.IMU)
            data.get_ground_truth()
            data.add_factors(Sensor.WHEEL)
            data += cam

            gt = data.traj[Sensor.GT]
            km = compute_traj_length(gt) / 1000

            # WheelRoseIntrSlipTag,  # wheel factor
            # jrl.PriorFactorPoint2Tag,  # slip prior throughout
            # PriorFactorIntrinsicsTag,  # intrinsics prior throughout
            # jrl.PriorFactorPoint3Tag,  # intrinsics prior

            frontend = makeFrontend()

            # try:
            #     # Run w/ IMU
            #     sol = frontend.run(
            #         data.to_dataset(),
            #         [
            #             jrl.PriorFactorPoint3Tag,
            #             jrl.PriorFactorIMUBiasTag,
            #             jrl.CombinedIMUTag,
            #             #
            #             WheelRoseIntrTag,
            #             PriorFactorIntrinsicsTag,
            #             jrl.PriorFactorPoint3Tag,
            #             #
            #             jrl.PriorFactorPose3Tag,  # pose prior
            #             jrl.StereoFactorPose3Point3Tag,
            #         ],
            #         f"opt.jrr",
            #         0,
            #         True,
            #     )
            #     ate_imu = jrl.computeATEPose3(gt, sol, False, False)[0]
            # except:
            #     ate_imu = 500
            ate_imu = 0

            try:
                # Run w/o IMU
                sol = frontend.run(
                    data.to_dataset(),
                    [
                        WheelRoseIntrSlipTag,
                        jrl.PriorFactorPoint3Tag,
                        jrl.PriorFactorPoint2Tag,
                        jrl.PriorFactorPoint3Tag,
                        #
                        jrl.PriorFactorPose3Tag,  # pose prior
                        jrl.StereoFactorPose3Point3Tag,
                    ],
                    f"opt.jrr",
                    0,
                    True,
                )
                ate_none = jrl.computeATEPose3(gt, sol, False, False)[0] / km
            except:
                ate_none = 500

            ate_total += ate_none
            ate_total += ate_imu
            ates.append([flat.name, ate_imu, ate_none])

            del frontend
            del data

        i += 1
        print(i, np.array(10**x), ate_total)
        print(tabulate(ates, headers="firstrow", tablefmt="github"))
        return ate_total

    def vec2noise(x, wheel_noise: Optional[WheelNoise] = None):
        if wheel_noise is None:
            wheel_noise = WheelNoise()

        x = 10**x
        # wheel_noise.sigma_rad_s = float(x[0])
        # wheel_noise.sigma_vy = float(x[1])
        # wheel_noise.sigma_vz = float(x[1])
        # wheel_noise.sigma_wx = float(x[2])
        # wheel_noise.sigma_wy = float(x[2])

        wheel_noise.sig_intr_baseline = float(x[0])
        wheel_noise.sig_intr_radius = float(x[1])

        # wheel_noise.sig_slip_prior = float(x[1])

        return wheel_noise

    bounds = np.array(
        [
            [1e-5, 1e-3],
            #
            # [0.01, 0.1],
            # [0.005, 0.05],
            #
            [1e-6, 1e-4],
            #
            # [-7, -1],
            # [-7, -1],
            # [-2, 0],
            # [1e-2, 1e0],
        ]
    )
    bounds = np.log10(bounds)

    out = shgo(objective, bounds, workers=10)
    print(out)

    # out = dual_annealing(
    #     objective,
    #     bounds,
    #     maxfun=100,
    #     no_local_search=False,
    #     x0=[0.05, 0.1, 0.01],
    # )

    # print(out)
    # noise = vec2noise(out.x, all_datasets[0].noise(Sensor.WHEEL))
    # noise.save("intr_wheel_rose.yaml")

    # print(
    #     direct(
    #         objective,
    #         bounds,
    #         locally_biased=False,
    #     )
    # )

    # print(
    #     brute(
    #         objective,
    #         ranges=bounds,
    #         args=("brute",),
    #         full_output=False,
    #         workers=5,
    #         Ns=10,
    #     )
    # )
