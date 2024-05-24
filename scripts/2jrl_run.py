import argparse
import os
from pathlib import Path

import cv2
import gtsam
import jrl
import numpy as np
from rose.dataset import CamNoise, Dataset2JRL, Sensor, WheelNoise
from rose import FlatDataset
from rose import KaistDataset
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
from rose import GrizzlyBag
from rose import SabercatBag
from tqdm import tqdm

np.set_printoptions(suppress=False, precision=4, linewidth=400)


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
    parser = argparse.ArgumentParser(description="Convert 2 jrl factor graph")
    parser.add_argument(
        "-k",
        "--kaist",
        type=Path,
        default=None,
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "-f",
        "--flat",
        type=Path,
        default=None,
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "-g",
        "--grizzly",
        type=Path,
        default=None,
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "-s",
        "--sabercat",
        type=Path,
        default=None,
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "--resave-data",
        action="store_true",
        help="Dataset folder to be read in.",
    )
    parser.add_argument(
        "--resave-params",
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

    d = WheelNoise.__dict__
    wheel_noise_params = {k: d[k] for k in d["__annotations__"].keys()}
    for k, v in wheel_noise_params.items():
        parser.add_argument(f"--{k}", type=float, default=v)

    d = CamNoise.__dict__
    cam_noise_params = {k: d[k] for k in d["__annotations__"].keys()}
    for k, v in cam_noise_params.items():
        parser.add_argument(f"--{k}", type=float, default=v)

    parser.add_argument("--meta", type=str, default="")

    parser.add_argument(
        "-n",
        "--iter",
        type=int,
        default=None,
        help="How many iterations to store. Defaults to all",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Whether to watch camera feed",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Don't use camera stash.",
    )
    args = parser.parse_args()
    args_dict = vars(args)

    # ------------------------- Run everything ------------------------- #
    wheel_noise_params = {k: args_dict[k] for k in wheel_noise_params.keys()}
    wheel_noise = WheelNoise(**wheel_noise_params)
    cam_noise_params = {k: args_dict[k] for k in cam_noise_params.keys()}
    cam_noise = CamNoise(**cam_noise_params)

    if args.grizzly is not None:
        bag = GrizzlyBag(args.grizzly)
        flat_path = args.flat_out / bag.name
        if args.resave_data or args.resave_params or not flat_path.exists():
            print(f"Saving grizzly bag to flat dataset at {flat_path}...")
            bag.to_flat(args.flat_out)
        dataset = FlatDataset(flat_path)

    elif args.sabercat is not None:
        bag = SabercatBag(args.sabercat)
        flat_path = args.flat_out / bag.name
        if not flat_path.exists():
            print(f"Saving sabercat bag to flat dataset at {flat_path}...")
            bag.to_flat(args.flat_out)
        if args.resave_data:
            print(f"Resaving data for {bag.name}...")
            bag.to_flat_data(args.flat_out)
        if args.resave_params:
            print(f"Resaving params for {bag.name}...")
            bag.to_flat_params(args.flat_out)
        dataset = FlatDataset(flat_path)

    elif args.kaist is not None:
        dataset = KaistDataset(args.kaist)

    elif args.flat is not None:
        dataset = FlatDataset(args.flat)

    # dataset.add_noise(Sensor.WHEEL, wheel_noise)
    # dataset.add_noise(Sensor.CAM, cam_noise)

    data = Dataset2JRL(dataset, args.iter)

    out_dir = Path(f"data/{dataset.name}/")
    out_dir.mkdir(exist_ok=True)

    # First check if we need to make & save camera factors
    if not (out_dir / "cam.jrl").exists() or args.no_cache:
        print("Adding stereo camera factors...")
        data.add_factors(Sensor.CAM, show=args.show)
        data.save_dataset(out_dir / "cam.jrl")
        data.clear_factors()

    print("Adding priors...")
    data.add_factors(Sensor.PRIOR)

    print("Adding IMU...")
    data.add_factors(Sensor.IMU)

    print("Adding Wheel...")
    data.add_factors(Sensor.WHEEL)

    print("Loading camera factors...")
    data.load_cache(out_dir / "cam.jrl")

    print("Getting ground truth...")
    data.get_ground_truth()

    data.save_traj(Sensor.GT, out_dir / "_gt.jrr")
    data.save_traj(Sensor.WHEEL, out_dir / "_wheel.jrr")
    gt = data.traj[Sensor.GT]
    km = compute_traj_length(gt) / 1000
    print("Ground truth length:", km)

    out_file = out_dir / f"data{args.meta}.jrl"

    def run(*kinds):
        factors = {
            "base": [jrl.PriorFactorPose3Tag, jrl.StereoFactorPose3Point3Tag],
            "imu": [
                jrl.PriorFactorPoint3Tag,
                jrl.PriorFactorIMUBiasTag,
                jrl.CombinedIMUTag,
            ],
            "wheel_rose": [WheelRoseTag],
            "wheel_intr": [WheelRoseIntrTag, jrl.PriorFactorPoint3Tag],
            "wheel_slip": [WheelRoseSlipTag, jrl.PriorFactorPoint2Tag],
            "wheel_intr_slip": [
                WheelRoseIntrSlipTag,
                jrl.PriorFactorPoint3Tag,
                jrl.PriorFactorPoint2Tag,
            ],
            "wheel_baseline": [WheelBaselineTag],
            "planar_prior": [ZPriorTag, PlanarPriorTag],
        }

        factors_these = sum([factors[k] for k in kinds], [])
        filename = out_dir / f"f-{'.'.join(kinds)}.jrr"

        try:
            frontend = makeFrontend()
            sol = frontend.run(
                data.to_dataset(),
                factors_these,
                str(filename),
                0,
                True,
            )
            ate = jrl.computeATEPose3(gt, sol, False, False)[0] / km
            print(f"{kinds} got {ate}")
        except:
            print(f"{kinds} Failed")

    # run("base")

    # run("base", "imu")
    # run("base", "imu", "wheel_rose")
    run("base", "imu", "wheel_intr_slip")
    # run("base", "imu", "wheel_baseline")
    # run("base", "imu", "wheel_baseline", "planar_prior")

    # run("base", "wheel_rose")
    run("base", "wheel_intr_slip")
    # run("base", "wheel_baseline")
    # run("base", "wheel_baseline", "planar_prior")
