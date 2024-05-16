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
from robust.jrl import values2results
from robust.robust_python import (
    PlanarPriorTag,
    PriorFactorIntrinsicsTag,
    WheelCovIntrSlipTag,
    WheelCovIntrTag,
    WheelCovSlipTag,
    WheelCovTag,
    WheelDangTag,
    ZPriorTag,
    makeFrontend,
)
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
    np.random.seed(3)
    params = SimParameters()

    params.slip_num = 10
    params.slip_length = 0.5
    params.slip_duration = 0.5

    params.w_intr_init_perturb = 1.1
    # params.w_intr_perturb_time = [50, 80]
    params.time = 100
    params.num_feats = 50
    params.sigma_pix = 1

    v = 2
    w = lambda t: np.cos(t * 1.5)
    sim = Simulation(params, np.pi / 6, v=2)
    sim.run_all(w, v)

    dataset = Dataset2JRL(sim)
    dataset.add_factors(Sensor.PRIOR, use_gt_orien=True)
    dataset.add_factors(Sensor.IMU)
    dataset.add_factors(Sensor.WHEEL)
    dataset.add_factors(Sensor.FEAT)
    gt = dataset.get_ground_truth()

    writer = robust.robust_python.makeRobustWriter()
    writer.writeResults(values2results(gt), "figures/data/gt.jrr", False)

    traj = {"GT": dataset.traj[Sensor.GT], "Wheel": dataset.traj[Sensor.WHEEL]}
    data = [["Kind", "ATEt", "ATEr"]]
    names = {
        WheelCovTag: "6DoF Cov Prop",
        WheelDangTag: "Planar",
        WheelCovSlipTag: "Slip",
        WheelCovIntrTag: "Intrinsics",
        WheelCovIntrSlipTag: "Intrinsics + Slip",
        None: "No Wheel",
    }
    for tag in [
        WheelDangTag,
        WheelCovTag,
        WheelCovSlipTag,
        WheelCovIntrTag,
        WheelCovIntrSlipTag,
        None,
    ]:
        sensors = [
            jrl.StereoFactorPose3Point3Tag,
            jrl.PriorFactorPose3Tag,
            # jrl.PriorFactorPoint3Tag,
            # jrl.PriorFactorIMUBiasTag,
            # jrl.CombinedIMUTag,
        ]
        if tag is not None:
            sensors.append(tag)
        if tag == WheelDangTag:
            sensors.append(PlanarPriorTag)
            sensors.append(ZPriorTag)
        if WheelCovSlipTag in sensors:
            sensors.append(jrl.PriorFactorPoint2Tag)
        if WheelCovIntrTag in sensors:
            sensors.append(jrl.PriorFactorPoint3Tag)
        if WheelCovIntrSlipTag in sensors:
            sensors.append(jrl.PriorFactorPoint2Tag)
            sensors.append(jrl.PriorFactorPoint3Tag)
        frontend = makeFrontend(kf=5)
        sol = frontend.run(
            dataset.to_dataset(), sensors, f"figures/data/{tag}.jrr", 0, False
        )

        d = dataset.to_dataset()
        graph = gtsam.NonlinearFactorGraph()
        for mm in d.measurements("a"):
            graph.push_back(mm.filter(sensors).measurements)

        # Compute error
        km = compute_traj_length(gt) / 1000
        et, er = jrl.computeATEPose3(gt, sol, False)
        data.append([names[tag], et, er * 180 / np.pi])
        traj[names[tag]] = sol
        print(f"\tFinished {tag}, e: {graph.error(sol):.2E}...")

    print(f"\nDist\t", km, "\n")
    print(tabulate(data, headers="firstrow", tablefmt="github"))

    # ------------------------- Plot States ------------------------- #
    # fig, ax = robust.plot.plot_error_state(GT=gt, **traj)
    # plt.savefig("sim_error.png")
    # fig, ax = robust.plot.plot_xy_trajectory(states="rpis", show=True, **traj)
    fig, ax = robust.plot.plot_state(
        states="rpis", show=True, filename="sim.png", **traj
    )