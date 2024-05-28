import functools
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import ClassVar, Union

import gtsam
import numpy as np
import sympy as sy
from gtsam.symbol_shorthand import L, M, X
from gtsam.utils.plot import set_axes_equal
from rose.dataset import (
    CameraData,
    CamNoise,
    FeatData,
    GTData,
    IMUData,
    IMUNoise,
    PriorNoise,
    Sensor,
    WheelData,
    WheelIntrinsics,
    WheelNoise,
)
from tabulate import tabulate
from tqdm import tqdm, trange

np.set_printoptions(suppress=False, precision=3, linewidth=200)

symt = sy.Symbol("t")


def func_t(s):
    return sy.Function(s, real=True)(symt)


symp = sy.Matrix([func_t(f"p{i}") for i in range(3)])
E = np.array([[0, -1, 0], [1, 0, 0]])
e1, e2, e3 = np.eye(3)

DEFAULT_WHEEL = WheelIntrinsics(baseline=1.6, radius_l=0.15, radius_r=0.15)


@dataclass
class SimParameters:
    name: str = "Simulation"
    # Manifold params
    manifold: sy.Expr = sy.Matrix(
        [
            symp[2]
            - sy.cos(symp[0] / 8) / 4
            - sy.cos(symp[1] / 8) / 4
            + (symp[0] - symp[1]) / 100
        ]
    )
    # Noise Params
    sigma_wz: float = 7e-3
    sigma_vx: float = 1e-2
    sigma_pix: float = 1
    sigma_w: float = 1e-5
    sigma_bw: float = 1e-5
    sigma_a: float = 1e-5
    sigma_ba: float = 1e-5
    # Wheel params
    slip_num: int = 0
    slip_duration: float = 0.5  # seconds
    slip_length: float = 0.5  # meters
    w_intr: WheelIntrinsics = DEFAULT_WHEEL
    w_intr_init_perturb: float = 1.0
    w_intr_perturb_time: list[float] = field(default_factory=lambda: [])
    w_intr_other = DEFAULT_WHEEL * np.array([1.0, 0.97, 0.95])
    # Extrinsincs
    body_T_cam: gtsam.Pose3 = gtsam.Pose3(
        gtsam.Rot3(
            np.array(
                [
                    [0, 0, 1],
                    [-1, 0, 0],
                    [0, -1, 0],
                ]
            )
        ),
        [0, 0, 0],
    )
    # Camera params
    num_feats: int = 100
    K: gtsam.Cal3_S2Stereo = gtsam.Cal3_S2Stereo(450, 450, 0, 320, 240, 0.2)
    res: tuple[int, int] = field(default_factory=lambda: [640, 480])
    dist_min: float = 5
    dist_max: float = 10
    # Sim params
    time: float = 100
    freq_wheel: int = 100
    freq_cam: int = 5


@dataclass
class State:
    pose: gtsam.Pose3 = gtsam.Pose3.Identity()
    velocity: np.ndarray = np.zeros(3)
    bias_w: np.ndarray = np.zeros(3)
    bias_a: np.ndarray = np.zeros(3)
    slip: np.ndarray = np.zeros(2)
    w_intr: WheelIntrinsics = DEFAULT_WHEEL

    def into_values(self, values, idx):
        values.insert(X(idx), self.pose)


class Simulation:
    def __init__(
        self,
        params: SimParameters,
        yaw: float = 0,
        x: float = 0,
        y: float = 0,
        v: float = 0,
    ) -> None:
        self.params = params
        self.name = self.params.name
        self.ran = False

        self.t = 0
        self.dt = np.int64(1e9 / self.params.freq_wheel)
        self.dt_cam = np.int64(1e9 / self.params.freq_cam)

        x0 = self._setup_manifold(self.params.manifold, x, y, yaw)

        self.pts_active = set()
        self.pts = np.zeros((0, 3))
        self.gravity = np.array([0, 0, 9.81])

        self.vw_last = x0.rotation().matrix() @ np.array([v, 0, 0])
        self.state = State(x0, velocity=self.vw_last, w_intr=self.params.w_intr)

        self.measurements = [
            {
                "t": 0,
                "state": self.state,
                "cam": self.project_points(self.state),
                "wheel": np.zeros(2),
            }
        ]

        self._setup_wheels()
        self.intr_default = True

        self.noise_override = {}
        self.extrinsics_override = {}

    def clear_cache(self):
        self.extrinsics.cache_clear()
        self.noise.cache_clear()
        self.data.cache_clear()

    def add_noise(self, sensor: Sensor, noise):
        self.noise_override[sensor] = noise
        self.noise.cache_clear()

    def add_extrinsics(self, sensor: Sensor, extrinsics):
        self.extrinsics_override[sensor] = extrinsics
        self.extrinsics.cache_clear()

    # ------------------------- Manifold / Integration Utilities ------------------------- #
    def _f(self, state, w, v):
        dt = self.dt / 1e9
        Rgtsam = state.pose.rotation()
        R = Rgtsam.matrix()
        p = state.pose.translation()

        norm_gradM = np.linalg.norm(self.gradM(p))
        v_body = np.array([v, 0, 0])
        w_xy = E @ R.T @ self.gradMdot(R @ v_body, p).flatten() / norm_gradM
        w_body = np.append(w_xy, w)

        # put it together
        Rnew = Rgtsam * gtsam.Rot3.Expmap(w_body * dt)
        pnew = p + R @ v_body * dt
        pose = gtsam.Pose3(Rnew, pnew)

        # Calculate acceleration
        v_world = R @ v_body
        a = R.T @ (v_world - self.vw_last) / dt
        self.vw_last = v_world

        # Propagate bias
        bias_w = self.state.bias_w + np.random.normal(
            0, self.params.sigma_bw * np.sqrt(dt), 3
        )
        bias_a = self.state.bias_a + np.random.normal(
            0, self.params.sigma_ba * np.sqrt(dt), 3
        )

        # Change intrinsics
        # print(self.t, self.params.w_intr_perturb_time)
        if np.isclose(self.t / 1e9, self.params.w_intr_perturb_time).any():
            if self.intr_default:
                self.state.w_intr = self.params.w_intr_other
            else:
                self.state.w_intr = self.params.w_intr
            self.intr_default = not self.intr_default

        return (
            State(
                pose,
                R @ v_body,
                bias_w,
                bias_a,
                w_intr=self.state.w_intr,
            ),
            w_body,
            a,
        )

    def _setup_manifold(self, M, x, y, yaw):
        gradM_sym = M.jacobian(symp)
        gradM2_sym = gradM_sym.jacobian(symp)
        gradMdot_sym = (
            sy.zeros(3, 1) if len(gradM_sym.free_symbols) == 0 else gradM_sym.diff()
        )

        self.M = sy.lambdify([symp], M)
        self.gradM = sy.lambdify([symp], gradM_sym)
        self.gradMdot = sy.lambdify([symp.diff(), symp], gradMdot_sym)
        self.gradM2 = sy.lambdify([symp], gradM2_sym)

        # compute initial pose
        p0 = np.array([x, y, -self.M([x, y, 0])[0, 0]])

        grad = self.gradM(p0).flatten()
        x_basis = np.array([np.cos(yaw), np.sin(yaw), 0])
        x_basis[-1] = -x_basis @ grad
        x_basis /= np.linalg.norm(x_basis)
        z_basis = grad / np.linalg.norm(grad)
        y_basis = np.cross(z_basis, x_basis)

        R = gtsam.Rot3(np.column_stack((x_basis, y_basis, z_basis)))
        return gtsam.Pose3(R, p0)

    # ------------------------- Wheel Utilities ------------------------- #
    def _setup_wheels(self):
        # Setup Slippage
        N = self.params.time * self.params.freq_wheel
        self.slips = np.zeros((N, 2))

        slip_indices = np.random.choice(N, self.params.slip_num, replace=False)

        slip_meas = np.array(
            [
                self.params.slip_length
                / (self.params.slip_duration * self.params.w_intr.radius_l),
                self.params.slip_length
                / (self.params.slip_duration * self.params.w_intr.radius_r),
            ]
        )

        for i in slip_indices:
            side = np.random.choice(2)
            sign = np.random.choice([-1, 1])
            for j in range(int(self.params.slip_duration * self.params.freq_wheel)):
                if i + j >= N:
                    break
                self.slips[i + j, side] = sign * np.random.normal(
                    loc=slip_meas[side], scale=0.05
                )

    # ------------------------- Camera Utilities ------------------------- #
    def project_points(self, state: State) -> np.ndarray:
        cam = gtsam.StereoCamera(
            state.pose.compose(self.params.body_T_cam), self.params.K
        )
        all_measurements = np.zeros((self.params.num_feats, 4))
        idx_mm = 0
        dist = np.linalg.norm(self.pts - state.pose.translation(), axis=1)

        pts_done = set()
        for id_feat in self.pts_active:
            p = self.pts[id_feat]
            # First, see if it's close at all
            if (
                self.params.dist_min > dist[id_feat]
                or dist[id_feat] > self.params.dist_max
            ):
                pts_done.add(id_feat)
                continue

            # Project onto image
            try:
                mm = cam.project(p)
            except:
                pts_done.add(id_feat)
                continue

            # If it's in the image, keep it!
            if (
                0 < mm.uL()
                and mm.uL() < self.params.res[0]
                and 0 < mm.uR()
                and mm.uR() < self.params.res[0]
                and 0 < mm.v()
                and mm.v() < self.params.res[1]
            ):
                mm_noisy = mm.vector() + np.random.normal(
                    scale=self.params.sigma_pix, size=3
                )
                all_measurements[idx_mm, 0] = id_feat
                all_measurements[idx_mm, 1:] = mm_noisy
                idx_mm += 1
            else:
                pts_done.add(id_feat)
                continue

            # Stop early if we get enough
            if idx_mm >= self.params.num_feats:
                # TODO: Remove the rest?
                break

        self.pts_active -= pts_done

        # If we don't have enough points, make some more
        new_pts = []
        id_feat = self.pts.shape[0]
        while idx_mm < self.params.num_feats:
            mm = self.generate_cam_mm()
            mm_noisy = mm.vector() + np.random.normal(
                scale=self.params.sigma_pix, size=3
            )
            self.pts_active.add(id_feat)
            all_measurements[idx_mm, 0] = id_feat
            all_measurements[idx_mm, 1:] = mm_noisy
            idx_mm += 1
            id_feat += 1

            pt3d = cam.backproject(mm)
            new_pts.append(pt3d)

        if len(new_pts) > 0:
            self.pts = np.vstack((self.pts, new_pts))

        return all_measurements

    def generate_cam_mm(self) -> tuple[gtsam.StereoPoint2, np.ndarray]:
        # don't sample points too close to edge
        ul, v, dist = np.random.uniform(
            [5, 5, self.params.dist_min],
            [self.params.res[0] - 5, self.params.res[1] - 5, self.params.dist_max],
        )
        ur = ul - self.params.K.baseline() * self.params.K.fx() / dist
        return gtsam.StereoPoint2(ul, ur, v)

    # ------------------------- Run simulation ------------------------- #
    def step(self, w, v):
        self.t += self.dt
        out = {"t": self.t}
        sqrt_dt = np.sqrt(self.dt / 1e9)

        # Wheel Measurement
        if self.t % self.dt < 1e-2:
            self.state, w_body, a_body = self._f(self.state, w, v)

            self.state.slip = self.slips[self.t // self.dt - 1]
            u_noisy = np.array(
                [
                    w + np.random.normal(0, self.params.sigma_wz) / sqrt_dt,
                    v + np.random.normal(0, self.params.sigma_vx) / sqrt_dt,
                ]
            )
            out["state"] = self.state
            out["wheel"] = (
                np.linalg.inv(self.state.w_intr.as_mat()) @ u_noisy + self.state.slip
            )
            out["imu"] = np.append(
                w_body
                + self.state.bias_w
                + np.random.normal(0, self.params.sigma_w / sqrt_dt, 3),
                a_body
                + self.state.pose.rotation().matrix().T @ self.gravity
                + self.state.bias_a
                + np.random.normal(0, self.params.sigma_a / sqrt_dt, 3),
            )

        # Camera Measurements
        if self.t % self.dt_cam < 1e-2:
            out["cam"] = self.project_points(self.state)

        if self.t >= self.params.time * 1e9:
            self.ran = True

        return out

    def run_all(self, w, v):
        if self.ran:
            return

        if type(w) in [int, float]:
            w_before = w
            w = lambda t: w_before
        if type(v) in [int, float]:
            v_before = v
            v = lambda t: v_before

        # Collect all data
        loop = tqdm(total=self.params.time * 1e9 // self.dt, leave=False)
        while not self.ran:
            self.measurements.append(self.step(w(self.t / 1e9), v(self.t / 1e9)))
            loop.update()

    # ------------------------- Overrides ------------------------- #
    @functools.cached_property
    def stamps(self):
        if not self.ran:
            raise TypeError("Simulation hasn't finished running yet!")

        return np.array([i["t"] for i in self.measurements if "cam" in i])

    @functools.cache
    def extrinsics(self, sensor: Sensor) -> gtsam.Pose3:
        if sensor in self.extrinsics_override:
            return self.extrinsics_override[sensor]

        if sensor == Sensor.FEAT:
            return self.params.body_T_cam
        else:
            return gtsam.Pose3()

    @functools.cache
    def data(self, sensor: Sensor) -> Union[IMUData, CameraData, GTData, WheelData]:
        if not self.ran:
            raise TypeError("Simulation hasn't finished running yet!")

        if sensor == Sensor.FEAT:
            # Measurements
            ids = [
                i["cam"][:, 0].astype("int") for i in self.measurements if "cam" in i
            ]
            mm = [i["cam"][:, 1:] for i in self.measurements if "cam" in i]

            # Perturb intrinsics slightly for practicality
            intr_perturb = np.random.multivariate_normal(
                mean=np.zeros(6), cov=np.diag([2, 2, 2, 2, 0, 1e-4])
            )
            intr = self.params.K  # .retract(intr_perturb)

            return FeatData(
                t=self.stamps,
                ids=ids,
                stereo_pts=mm,
                extrinsics=self.extrinsics(Sensor.FEAT),
                intrinsics=intr,
                noise=self.noise(Sensor.FEAT),
            )

        elif sensor == Sensor.GT:
            all_mm = [i["state"] for i in self.measurements if "state" in i]
            x = np.array([i.pose.matrix() for i in all_mm])
            vel = np.vstack([i.velocity for i in all_mm])
            bias_w = np.array([i.bias_w for i in all_mm])
            bias_a = np.array([i.bias_a for i in all_mm])
            slip = np.array([i.slip for i in all_mm])
            intrinsics = np.array([i.w_intr for i in all_mm])
            # print("\nNumber of slips:", (slip != 0).any(axis=1).sum(), slip.shape[0])

            t = np.array([i["t"] for i in self.measurements if "state" in i])
            return GTData(
                t=t,
                x=x,
                vel=vel,
                bias_w=bias_w,
                bias_a=bias_a,
                slip=slip,
                intrinsics=intrinsics,
                extrinsics=self.extrinsics(Sensor.GT),
            )

        elif sensor == Sensor.WHEEL:
            wl, wr = np.array([i["wheel"] for i in self.measurements if "wheel" in i]).T
            t = np.array([i["t"] for i in self.measurements if "wheel" in i])
            return WheelData(
                t=t,
                wl=wl,
                wr=wr,
                extrinsics=self.extrinsics(Sensor.WHEEL),
                intrinsics=self.params.w_intr * self.params.w_intr_init_perturb,
                noise=self.noise(Sensor.WHEEL),
            )

        elif sensor == Sensor.IMU:
            mm = np.array([i["imu"] for i in self.measurements if "imu" in i])
            t = np.array([i["t"] for i in self.measurements if "imu" in i])
            w = mm[:, :3]
            a = mm[:, 3:]
            return IMUData(t=t, w=w, a=a, noise=self.noise(Sensor.IMU))

        else:
            raise NotImplementedError()

    @functools.cache
    def noise(self, sensor: Sensor):
        if sensor in self.noise_override:
            return self.noise_override[sensor]

        if sensor == Sensor.IMU:
            return IMUNoise(
                sigma_w=self.params.sigma_w,
                sigma_bw=self.params.sigma_bw,
                sigma_a=self.params.sigma_a,
                sigma_ba=self.params.sigma_ba,
                preint_cov=1e-7,
                preint_bias_cov=1e-7,
            )

        elif sensor == Sensor.FEAT:
            return CamNoise(sig_pix=self.params.sigma_pix)

        elif sensor == Sensor.WHEEL:
            M = self.params.w_intr.as_mat()
            Minv = np.linalg.inv(M)
            wz_vx = np.diag([self.params.sigma_wz, self.params.sigma_vx])

            sig_wl = Minv @ wz_vx @ Minv.T

            return WheelNoise(
                sig_man=1e-9,
                sig_maninit=1e-6,
                sig_man_pos=1e-2,
                sig_man_orien=1e-2,
                sigma_rad_s=sig_wl[0, 0],
                sigma_wx=5e-2,
                sigma_wy=5e-2,
                sigma_vy=1e-2,  # in the simulation case, this is 100% valid assumption
                sigma_vz=1e-2,  # => can arbitrarily shrink for better results. I've left it in the ballpark of the real version
                sig_intr_baseline=1e-2,
                sig_intr_radius=5e-4,
                sig_slip_prior=2,
                slip_prior_kernel=1,
            )

        elif sensor == Sensor.PRIOR:
            return PriorNoise()
