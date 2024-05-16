import argparse
from functools import partial
from pathlib import Path
from typing import Any, Callable

import gtsam
import jrl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from robust.dataset import Dataset2JRL, Sensor, WheelData, WheelIntrinsics
from robust.flat import FlatDataset
from robust.kaist import KaistDataset
from robust.robust_python import PreintegratedWheelCov, PreintegratedWheelParams
from robust.ros import GrizzlyBag

np.set_printoptions(suppress=True, precision=4)


def add_eps(x, eps_vec):
    return x + eps_vec


def add_eps_pose(x, eps_vec):
    # Define our retraction for numerical derivatives
    e = gtsam.Pose3(gtsam.Rot3.Expmap(eps_vec[:3]), eps_vec[3:])
    return x * e


def nder(
    f: Callable, x: Any, eps: float = 1e-6, add_eps: Callable = add_eps
) -> np.ndarray:
    fx = f(x)
    if type(x) is np.ndarray:
        N = x.shape[0]
    elif type(x) is gtsam.Pose3:
        N = 6

    M = fx.shape[0]
    d = np.zeros((M, N))
    for i in range(N):
        eps_vec = np.zeros(N)
        eps_vec[i] = eps
        temp = add_eps(x, eps_vec)
        d[:, i] = (f(temp) - fx) / eps

    return d


def wheel_estimate(slip: np.ndarray, data: np.ndarray, dt: float = 0.1) -> np.ndarray:
    assert data.shape[1] == 2
    pwmParams = PreintegratedWheelParams()
    pwm = PreintegratedWheelCov(pwmParams)

    for wl, wr in data:
        pwm.integrateMeasurements(wl + slip[0], wr + slip[1], dt)

    return pwm.preint()


def H_function(slip: np.ndarray, data: np.ndarray, dt: float = 0.1) -> np.ndarray:
    pwmParams = PreintegratedWheelParams()
    pwm = PreintegratedWheelCov(pwmParams)

    E = np.zeros((6, 2))
    E[2:4, :] = np.eye(2)
    T = pwmParams.intrinsicsMat()
    out = np.zeros((6, 2))

    for wl, wr in data:
        preint = pwm.preint()
        Htinv = np.linalg.inv(gtsam.Pose3.ExpmapDerivative(preint))

        def temp(x):
            return (
                dt * np.linalg.inv(gtsam.Pose3.ExpmapDerivative(x)) @ E @ T @ [wl, wr]
            )

        A = np.eye(6) + nder(temp, preint)

        out = A @ out + dt * Htinv @ E @ T

        pwm.integrateMeasurements(wl, wr, dt)

    return out, pwm.preint_H_slip()


if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.rand(100, 2)
    slip = np.zeros(2)
    obj = partial(wheel_estimate, data=data)

    H_num = nder(obj, slip)
    H, H_cpp = H_function(slip, data)
    print(H_num)
    print(H)
    print(H_cpp)
