import gtsam
import jrl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rose.plot
from rose.dataset import Dataset2JRL, Sensor
from rose.jrl import (
    CombinedIMUTag,
    PlanarPriorTag,
    PriorFactorIMUBiasTag,
    StereoFactorPose3Point3Tag,
    WheelBaselineTag,
    WheelRoseIntrSlipTag,
    WheelRoseIntrTag,
    WheelRoseSlipTag,
    WheelRoseTag,
    ZPriorTag,
)
from rose.sim import SimParameters, Simulation
from tqdm import trange

from rose import makeFrontend


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


def main(ideal: bool):
    kind = "ideal" if ideal else "real"
    print(f"Running Monte Carlo for {kind}...")

    np.random.seed(0)
    params = SimParameters()

    runs = 10

    params.slip_num = 0 if ideal else 10
    params.slip_length = 0.5
    params.slip_duration = 0.5

    params.w_intr_init_perturb = 1.0 if ideal else 1.1
    params.time = 100
    params.num_feats = 50
    params.sigma_pix = 1

    v = 2
    w = lambda t: np.cos(t * 1.5)

    traj = {}
    data = []
    for i in trange(runs, leave=False):
        sim = Simulation(params, yaw=np.pi / 6, v=v)
        sim.run_all(w, v)

        dataset = Dataset2JRL(sim)
        dataset.add_factors(Sensor.PRIOR, use_gt_orien=True)
        dataset.add_factors(Sensor.IMU)
        dataset.add_factors(Sensor.WHEEL)
        dataset.add_factors(Sensor.FEAT)

        gt = dataset.get_ground_truth()
        km = compute_traj_length(gt) / 1000

        wheel = dataset.traj[Sensor.WHEEL]
        et, er = rose.jrl.computeATEPose3(gt, wheel, False)
        data.append(["wheel", False, et, er * 180 / np.pi])

        for use_imu in [False]:
            for tag in [
                WheelBaselineTag,
                WheelRoseTag,
                WheelRoseIntrSlipTag,
                "stereo",
            ]:
                sensors = [
                    jrl.PriorFactorPose3Tag,
                    StereoFactorPose3Point3Tag,
                ]
                if tag != "stereo":
                    sensors.append(tag)
                if WheelBaselineTag in sensors:
                    sensors.append(PlanarPriorTag)
                    sensors.append(ZPriorTag)
                if WheelRoseSlipTag in sensors:
                    sensors.append(jrl.PriorFactorPoint2Tag)
                if WheelRoseIntrTag in sensors:
                    sensors.append(jrl.PriorFactorPoint3Tag)
                if WheelRoseIntrSlipTag in sensors:
                    sensors.append(jrl.PriorFactorPoint2Tag)
                    sensors.append(jrl.PriorFactorPoint3Tag)
                if use_imu:
                    sensors.extend(
                        [
                            jrl.PriorFactorPoint3Tag,
                            PriorFactorIMUBiasTag,
                            CombinedIMUTag,
                        ]
                    )

                frontend = makeFrontend(kf=5)
                sol = frontend.run(dataset.to_dataset(), sensors, "temp.jrr", 0, False)

                # Compute error
                name = f"{tag}{i}"
                et, er = rose.jrl.computeATEPose3(gt, sol, False)
                data.append([tag, use_imu, et, er * 180 / np.pi])
                traj[name] = sol
                # print(f"\tFinished {name}...")

            rose.plot.plot_state(
                GT=gt,
                states="rpis",
                show=False,
                filename=f"sim_{i}.png",
                baseline=traj[f"{WheelBaselineTag}{i}"],
                rose_minus=traj[f"{WheelRoseTag}{i}"],
                rose=traj[f"{WheelRoseIntrSlipTag}{i}"],
            )
            plt.close()

    print(f"\nDist\t", km, "\n")

    # ------------------------- Plot States ------------------------- #
    df = pd.DataFrame(
        data,
        columns=["WheelFactor", "IMU", "ATEt", "ATEr"],
    )
    df.to_pickle(f"figures/data/sim_monte_carlo_{kind}.pkl")


if __name__ == "__main__":
    # main(True)
    main(False)
