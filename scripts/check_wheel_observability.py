import gtsam
import numpy as np
from gtsam.symbol_shorthand import I, S, X
from rose.rose_python import (
    PreintegratedWheelCov,
    PreintegratedWheelParams,
    WheelFactor3,
    WheelFactor4Intrinsics,
    WheelFactor5,
)

np.set_printoptions(precision=3, suppress=True, linewidth=300)


def wheel_estimate(
    intrinsics: np.ndarray, data: np.ndarray, dt: float = 0.1
) -> np.ndarray:
    assert data.shape[1] == 2
    b, wl, wr = intrinsics
    pwmParams = PreintegratedWheelParams()
    pwmParams.intrinsics = [b, wl, wr]
    pwmParams = PreintegratedWheelParams()
    pwm = PreintegratedWheelCov(pwmParams)

    for wl, wr in data:
        pwm.integrateMeasurements(wl, wr, dt)

    return pwm


if __name__ == "__main__":
    np.random.seed(0)
    data = np.random.rand(100, 2)
    intrinsics = np.array([2, 0.5, 0.5])
    x0 = gtsam.Pose3.Identity()
    slip = np.zeros(2)

    # ------------------------- Make values & graph ------------------------- #
    values = gtsam.Values()
    graph = gtsam.NonlinearFactorGraph()
    pwm = wheel_estimate(intrinsics, data)

    x_prior = gtsam.PriorFactorPose3(
        X(0), x0, gtsam.noiseModel.Isotropic.Sigma(6, 1e-5)
    )
    int_prior = gtsam.PriorFactorPoint3(
        I(0), intrinsics, gtsam.noiseModel.Isotropic.Sigma(3, 1e-5)
    )
    graph.push_back(x_prior)
    graph.push_back(int_prior)
    values.insert(X(0), x0)
    values.insert(I(0), intrinsics)

    for i in range(5):
        wf5 = WheelFactor5(X(i), I(i), X(i + 1), I(i + 1), S(i), pwm)
        wf4 = WheelFactor4Intrinsics(X(i), I(i), X(i + 1), I(i + 1), pwm)
        wf3 = WheelFactor3(X(i), X(i + 1), S(i), pwm)

        graph.push_back(wf5)

        slip_prior = gtsam.PriorFactorPoint2(
            S(i), slip, gtsam.noiseModel.Isotropic.Sigma(2, 1e-5)
        )
        graph.push_back(slip_prior)
        values.insert(S(i), slip)

        x_next = pwm.predict(values.atPose3(X(i)))
        values.insert(X(i + 1), x_next)
        values.insert(I(i + 1), intrinsics)

    # ------------------------- Check out linearization ------------------------- #
    print("Checking out linearization...")
    gaussian_graph = graph.linearize(values)
    A, b = gaussian_graph.jacobian()
    # A = A[-9:]
    print(A.shape)
    print(np.linalg.matrix_rank(A))
    # print(A)
