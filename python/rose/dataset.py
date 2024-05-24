from __future__ import annotations

import functools
from dataclasses import asdict, astuple, dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import cv2
import gtsam
import jrl
import numpy as np
import yaml
from gtsam.symbol_shorthand import B, I, L, M, S, V, W, X
from rose.jrl import (
    makeRoseParser,
    makeRoseWriter,
    values2results,
    values2typedvalues,
)
from rose.rose_python import (
    PlanarPriorFactor,
    PlanarPriorTag,
    PreintegratedWheelRose,
    PreintegratedWheelBaseline,
    PreintegratedWheelParams,
    PriorFactorIntrinsicsTag,
    WheelRoseIntrSlipTag,
    WheelRoseIntrTag,
    WheelRoseSlipTag,
    WheelRoseTag,
    WheelBaselineTag,
    WheelFactor2,
    WheelFactor3,
    WheelFactor4Intrinsics,
    WheelFactor5,
    ZPriorFactor,
    ZPriorTag,
)
from tqdm import tqdm

STEREO_MATCHER = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=128,
    blockSize=9,
    P1=100,
    P2=1000,
    disp12MaxDiff=1,
    preFilterCap=0,
    uniquenessRatio=10,
    speckleWindowSize=200,
    speckleRange=100,
    mode=cv2.StereoSGBM_MODE_SGBM,
)

KP_DETECTOR = cv2.FastFeatureDetector_create(threshold=40)

FEATURE_COL = 10
FEATURE_ROW = 10
FEATURE_PER_CELL = 2
FEATURE_BORDER = 5
MIN_MATCHES = 15


# ------------------------- Helpers to make sure we don't "stringly" type ------------------------- #
class Sensor(Enum):
    IMU = 1
    CAM = 2
    GT = 3
    WHEEL = 4
    PRIOR = 5
    FEAT = 6


# ------------------------- Noise Wrappers ------------------------- #
sigma_rad_s = 2e-1
SIGMA_WX = 0.02
SIGMA_WY = 0.02
SIGMA_VY = 0.18
SIGMA_VZ = 0.05
SIG_MAN = 0.09
SIG_MANINIT = 0.11
SIG_RP_PRIOR = 1e-3
SIG_Z_PRIOR = 0.003
SIG_MAN_POS = 1e-2
SIG_MAN_ORIEN = 1e-2


class BaseNoise:
    def __array__(self):
        return np.array(astuple(self))

    def vector(self):
        return self.__array__()

    def dict(self):
        return asdict(self)

    def save(self, file: Path):
        with open(file, "w") as outfile:
            yaml.dump(self.dict(), outfile, default_flow_style=False)

    def copy(self):
        return replace(self)


@dataclass(kw_only=True)
class WheelNoise(BaseNoise):
    sigma_rad_s: float = sigma_rad_s

    sigma_wx: float = SIGMA_WX
    sigma_wy: float = SIGMA_WY
    sigma_vy: float = SIGMA_VY
    sigma_vz: float = SIGMA_VZ

    sig_man: float = SIG_MAN
    sig_maninit: float = SIG_MANINIT
    sig_man_pos: float = SIG_MAN_POS
    sig_man_orien: float = SIG_MAN_ORIEN

    sig_rp_prior: float = SIG_RP_PRIOR
    sig_z_prior: float = SIG_Z_PRIOR

    sig_slip_prior: float = 0.4
    slip_prior_kernel: float = 2.7

    # sig_intr_baseline: float = 1e-4
    # sig_intr_radius: float = 5e-5
    sig_intr_baseline: float = 5e-3
    sig_intr_radius: float = 5e-3

    sig_intr_prior_baseline: float = 1e-2
    sig_intr_prior_radius: float = 1e-2

    def gtsam(self, intrinsics: WheelIntrinsics):
        pwmParams = PreintegratedWheelParams()

        pwmParams.intrinsics = intrinsics.vector()
        pwmParams.setWVCovFromWheel(
            self.sigma_rad_s**2,
            self.sigma_rad_s**2,
        )

        pwmParams.wxCov = self.sigma_wx**2
        pwmParams.wyCov = self.sigma_wy**2
        pwmParams.vyCov = self.sigma_vy**2
        pwmParams.vzCov = self.sigma_vz**2

        pwmParams.manCov = np.eye(3) * self.sig_man**2
        pwmParams.manInitCov = np.eye(3) * self.sig_maninit**2
        pwmParams.manPosCov = self.sig_man_pos**2
        pwmParams.manOrienCov = self.sig_man_pos**2

        pwmParams.intrinsicsBetweenCov = (
            np.diag(
                [
                    self.sig_intr_baseline,
                    self.sig_intr_radius,
                    self.sig_intr_radius,
                ]
            )
            ** 2
        )
        return pwmParams


@dataclass(kw_only=True)
class IMUNoise(BaseNoise):
    sigma_a: float
    sigma_w: float
    sigma_ba: float
    sigma_bw: float
    preint_cov: float
    preint_bias_cov: float

    def gtsam(self):
        pim_params = gtsam.PreintegrationCombinedParams.MakeSharedU()
        pim_params.setAccelerometerCovariance(np.eye(3) * self.sigma_a**2)
        pim_params.setGyroscopeCovariance(np.eye(3) * self.sigma_w**2)
        pim_params.setBiasAccCovariance(np.eye(3) * self.sigma_ba**2)
        pim_params.setBiasOmegaCovariance(np.eye(3) * self.sigma_bw**2)
        pim_params.setIntegrationCovariance(np.eye(3) * self.preint_cov)
        pim_params.setBiasAccOmegaInit(np.eye(6) * self.preint_bias_cov)
        return gtsam.PreintegratedCombinedMeasurements(pim_params)


@dataclass(kw_only=True)
class CamNoise(BaseNoise):
    sig_pix: float = 1.5

    def gtsam(self):
        return gtsam.noiseModel.Isotropic.Sigma(3, self.sig_pix)


@dataclass(kw_only=True)
class PriorNoise(BaseNoise):
    sig_pose: float = 1e-4
    sig_vel: float = 1e-4
    sig_bias: float = 1e-2
    sig_intr: float = 1e-5


# ------------------------- Intrinsics Wrappers ------------------------- #
@dataclass(kw_only=True)
class BaseIntrinsics:
    def dict(self):
        return asdict(self)

    def __array__(self):
        return np.array(astuple(self))

    def vector(self):
        return self.__array__()


@dataclass(kw_only=True)
class WheelIntrinsics(BaseIntrinsics):
    baseline: float
    radius_l: float
    radius_r: float

    def __mul__(self, other):
        if type(other) is float:
            return WheelIntrinsics(
                baseline=self.baseline * other,
                radius_l=self.radius_l * other,
                radius_r=self.radius_r * other,
            )
        elif type(other) is np.ndarray and other.size == 3:
            return WheelIntrinsics(
                baseline=self.baseline * other[0],
                radius_l=self.radius_l * other[1],
                radius_r=self.radius_r * other[2],
            )
        else:
            raise ValueError("Can only multiply by float or np.ndarray")

    def as_mat(self):
        return np.array(
            [
                [-self.radius_l / self.baseline, self.radius_r / self.baseline],
                [self.radius_l / 2, self.radius_r / 2],
            ]
        )


@dataclass(kw_only=True)
class CameraIntrinsics(BaseIntrinsics):
    # For use after rectification
    fx: float
    fy: float
    scale: float = 0
    cx: float
    cy: float
    baseline: float
    # To rectify
    mapx_l: np.ndarray = None
    mapy_l: np.ndarray = None
    mapx_r: np.ndarray = None
    mapy_r: np.ndarray = None
    is_rect: bool = True

    @property
    def stereo(self):
        return gtsam.Cal3_S2Stereo(
            self.fx, self.fy, self.scale, self.cx, self.cy, self.baseline
        )


# ------------------------- Data Wrappers ------------------------- #
@dataclass(kw_only=True)
class BaseData:
    t: np.ndarray  # in nsecs
    noise: BaseNoise
    extrinsics: gtsam.Pose3 = gtsam.Pose3.Identity()

    def __post_init__(self):
        # Make sure we're storing things as nanoseconds properly
        assert (
            self.t.dtype == np.int64
        ), "Time is not stored as np.int64, likely isn't nanoseconds!"

        # Make sure all data is in order
        assert np.all(
            np.diff(self.t) >= 0
        ), f"{self.__class__.__name__} Time is not in ascending order"

    @property
    def shape(self):
        return self.t.shape[0]

    def dt(self, idx, secs=True):
        return float(self.t[idx] - self.t[idx - 1]) / 1e9


@dataclass(kw_only=True)
class IMUData(BaseData):
    w: np.ndarray
    a: np.ndarray

    def __post_init__(self):
        # Delete any jumps greater than a second in the data
        dt = np.diff(self.t) / 1e9
        bad_idx = np.where(dt >= 1.0)[0] + 1

        self.t = np.delete(self.t, bad_idx)
        self.w = np.delete(self.w, bad_idx, axis=0)
        self.a = np.delete(self.a, bad_idx, axis=0)

        return super().__post_init__()


@dataclass(kw_only=True)
class GTData(BaseData):
    x: np.ndarray
    bias_w: np.ndarray = None
    bias_a: np.ndarray = None
    vel: np.ndarray = None
    slip: np.ndarray = None
    intrinsics: np.ndarray = None
    noise: BaseNoise = None

    def __post_init__(self):
        # Delete any jumps greater than a second in the data
        dt = np.diff(self.t) / 1e9
        bad_idx = np.where(dt >= 1.0)[0] + 1

        self.t = np.delete(self.t, bad_idx)
        self.x = np.delete(self.x, bad_idx, axis=0)
        if self.bias_w is not None:
            self.bias_w = np.delete(self.bias_w, bad_idx, axis=0)
        if self.bias_a is not None:
            self.bias_a = np.delete(self.bias_a, bad_idx, axis=0)
        if self.vel is not None:
            self.vel = np.delete(self.vel, bad_idx, axis=0)
        if self.slip is not None:
            self.slip = np.delete(self.slip, bad_idx, axis=0)
        if self.intrinsics is not None:
            self.intrinsics = np.delete(self.intrinsics, bad_idx, axis=0)

    def __getitem__(self, idx):
        return gtsam.Pose3(self.x[idx])


@dataclass(kw_only=True)
class StereoPic:
    l: np.ndarray
    r: np.ndarray
    disparity: np.ndarray
    intrinsics: gtsam.Cal3_S2Stereo
    kps: np.ndarray = np.zeros((0, 2), dtype=np.float32)
    ids: np.ndarray = np.zeros((0,), dtype=np.uint64)
    mask: Optional[np.ndarray] = None

    # These are the same across all StereoPic instances
    feat_idx = 1
    feat_border = FEATURE_BORDER
    feat_rows = FEATURE_ROW
    feat_cols = FEATURE_COL
    feat_per_cell = FEATURE_PER_CELL

    def __post_init__(self):
        self.camera = gtsam.StereoCamera(gtsam.Pose3(), self.intrinsics)
        self.h, self.w = self.l.shape[:2]

        if self.mask is None:
            self.mask = np.full((self.h, self.w), True)
        else:
            assert (
                self.mask.shape == self.l.shape
            ), "Mask shape doesn't match image shape"

    # TODO: Find better way to do this
    def _reset_vec(self):
        self.kpdisparity = np.array(
            [
                self.disparity[y, x] if x < self.w and y < self.h else -2
                for x, y in self.kps.astype(np.uint32)
            ]
        )

        self.stereopts = [
            gtsam.StereoPoint2(uL, uL - d, v)
            for (uL, v), d in zip(self.kps, self.kpdisparity)
        ]

        self.kpmask = np.array(
            [
                self.mask[y, x] if x < self.w and y < self.h else False
                for x, y in self.kps.astype(np.uint32)
            ]
        )

    def hstack(self):
        return np.hstack((self.l, self.r))

    def fill_out_kps(self):
        count, indices = self._count_kps()

        new_kps = []
        rem_kps = []
        for i, (row, col, subblock) in enumerate(self._blocks()):
            off_by = int(self.feat_per_cell - count[i])
            # If there's not enough
            if off_by > 0:
                # Get best ones with positive disparity & out of mask
                kps = KP_DETECTOR.detect(subblock, None)
                kps = [
                    k
                    for k in kps
                    if (
                        self.disparity[int(k.pt[1]) + row, int(k.pt[0]) + col] > 2
                        and self.mask[int(k.pt[1]) + row, int(k.pt[0]) + col]
                    )
                ]
                kps = sorted(kps, key=lambda kp: kp.response, reverse=True)[:off_by]
                if len(kps) == 0:
                    continue
                kps = cv2.KeyPoint_convert(kps) + [col, row]
                new_kps.append(kps)
            # If there's too many
            elif off_by < 0:
                rem_kps.extend(indices[i][: self.feat_per_cell])

        # # Clear out any boxes that are overly full
        # if len(rem_kps) != 0:
        #     self.kps = np.delete(self.kps, rem_kps, 0)
        #     self.ids = np.delete(self.ids, rem_kps, 0)

        # Add in new keypoints
        if len(new_kps) != 0:
            new_kps = np.vstack(new_kps).astype(np.float32)
            self.kps = np.vstack((self.kps, new_kps))
            self.ids = np.concatenate(
                (self.ids, np.zeros(new_kps.shape[0], dtype=np.uint64))
            )

        self._reset_vec()

    def _count_kps(self):
        h, w = self.l.shape[:2]
        w_block_size = (w - 2 * self.feat_border) / self.feat_cols
        h_block_size = (h - 2 * self.feat_border) / self.feat_rows
        count = np.zeros((self.feat_rows, self.feat_cols), dtype=np.uint64)
        indices = [[[] for _ in range(self.feat_cols)] for _ in range(self.feat_rows)]

        for i, (u, v) in enumerate(self.kps):
            w_idx = int((u - self.feat_border) // w_block_size)
            h_idx = int((v - self.feat_border) // h_block_size)
            count[h_idx, w_idx] += 1
            indices[h_idx][w_idx].append(i)

        indices = [i for row in indices for i in row]
        return count.flatten(), indices

    def _blocks(self):
        xs = np.uint32(
            np.rint(
                np.linspace(
                    0 + self.feat_border,
                    self.w - self.feat_border,
                    num=self.feat_cols + 1,
                )
            )
        )
        ys = np.uint32(
            np.rint(
                np.linspace(
                    0 + self.feat_border,
                    self.h - self.feat_border,
                    num=self.feat_rows + 1,
                )
            )
        )
        ystarts, yends = ys[:-1], ys[1:]
        xstarts, xends = xs[:-1], xs[1:]
        for y1, y2 in zip(ystarts, yends):
            for x1, x2 in zip(xstarts, xends):
                yield y1, x1, self.l[y1:y2, x1:x2]

    def _ransac(
        self, nextPic: StereoPic, matches: np.ndarray
    ) -> tuple[np.ndarray, float]:
        if matches.shape[0] < MIN_MATCHES:
            # return np.zeros(0), 0
            return matches, 0

        # convert all current points to 3d
        pts3d = [self.camera.backproject(p) for p in self.stereopts]
        pts2d = nextPic.kps

        pts3d_matched = np.array([pts3d[i] for i, j in matches])
        pts2d_matched = np.array([pts2d[j] for i, j in matches])

        success, R, t, inliers = cv2.solvePnPRansac(
            pts3d_matched,
            pts2d_matched,
            self.intrinsics.K(),
            distCoeffs=0,
            reprojectionError=4.0,
            flags=cv2.SOLVEPNP_P3P,
        )
        if inliers is None or inliers.size < MIN_MATCHES:
            return np.zeros(0), 0
        return matches[inliers.squeeze()], np.linalg.norm(t)

    def propagate(self, nextPic: StereoPic) -> tuple[np.ndarray, float]:
        # Move matches forward
        propped_kp, st, err = cv2.calcOpticalFlowPyrLK(
            self.l,
            nextPic.l,
            self.kps[:, np.newaxis],
            None,
            winSize=(11, 11),
            maxLevel=4,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )
        nextPic.kps = propped_kp.squeeze()
        nextPic._reset_vec()
        st = st.flatten().astype(bool)

        # Only select the good ones
        good_propped_feat = np.logical_and.reduce(
            (
                st,
                FEATURE_BORDER < nextPic.kps[:, 0],
                nextPic.kps[:, 0] < nextPic.w - FEATURE_BORDER,
                FEATURE_BORDER < nextPic.kps[:, 1],
                nextPic.kps[:, 1] < nextPic.h - FEATURE_BORDER,
                nextPic.kpdisparity > 2,
                self.kpmask,
            )
        )
        kp_prev_idx = np.where(good_propped_feat)[0]

        # Store everything
        nextPic.kps = nextPic.kps[good_propped_feat]
        nextPic.ids = np.zeros(nextPic.kps.shape[0], dtype=np.uint64)
        nextPic._reset_vec()

        # Clean up matches
        matches = np.column_stack(
            (
                kp_prev_idx,
                np.arange(nextPic.kps.shape[0], dtype=np.uint32),
            )
        )
        matches, dist = self._ransac(nextPic, matches)

        # Propagate feature indices on RANSAC matches
        for idx_prev, idx_curr in matches:
            id_prev = self.ids[idx_prev]
            if id_prev == 0:
                nextPic.ids[idx_curr] = StereoPic.feat_idx
                StereoPic.feat_idx += 1
            else:
                nextPic.ids[idx_curr] = id_prev

        return matches, dist


@dataclass(kw_only=True)
class CameraData(BaseData):
    left: list[Path]
    right: list[Path]
    disparity: list[Path]
    intrinsics: CameraIntrinsics
    mask: Optional[np.ndarray] = None

    total = 0

    def __getitem__(self, idx) -> StereoPic:
        l = cv2.imread(str(self.left[idx]), cv2.IMREAD_GRAYSCALE)
        r = cv2.imread(str(self.right[idx]), cv2.IMREAD_GRAYSCALE)

        # Rectify them
        if not self.intrinsics.is_rect:
            l = cv2.remap(
                l, self.intrinsics.mapx_l, self.intrinsics.mapy_l, cv2.INTER_LINEAR
            )
            r = cv2.remap(
                r, self.intrinsics.mapx_r, self.intrinsics.mapy_r, cv2.INTER_LINEAR
            )

        # Cache disparity results
        disp_file = self.disparity[idx]
        if disp_file.exists():
            disp = cv2.imread(str(disp_file), cv2.IMREAD_GRAYSCALE)
        else:
            disp = STEREO_MATCHER.compute(l, r).astype(np.float32) / 16
            disp_file.parent.mkdir(exist_ok=True)
            cv2.imwrite(str(disp_file), disp)

        return StereoPic(
            l=l, r=r, mask=self.mask, disparity=disp, intrinsics=self.intrinsics.stereo
        )


@dataclass(kw_only=True)
class WheelData(BaseData):
    wl: np.ndarray
    wr: np.ndarray
    intrinsics: WheelIntrinsics

    def _clear_cache(self):
        if "w" in self.__dict__:
            del self.__dict__["w"]
        if "v" in self.__dict__:
            del self.__dict__["v"]

    def interp(self, stamps):
        N = stamps.size + self.t.size
        new_wl = np.zeros(N)
        new_wr = np.zeros(N)
        new_t = np.zeros(N, dtype=np.int64)
        data_idx_og = 0
        data_idx_new = 0
        for s in stamps:
            while self.t[data_idx_og] < s:
                new_wl[data_idx_new] = self.wl[data_idx_og]
                new_wr[data_idx_new] = self.wr[data_idx_og]
                new_t[data_idx_new] = self.t[data_idx_og]
                data_idx_og += 1
                data_idx_new += 1

            if self.t[data_idx_og] == s:
                continue

            # Interpolate
            new_wl[data_idx_new] = np.interp(
                s,
                self.t[data_idx_og - 1 : data_idx_og + 1],
                self.wl[data_idx_og - 1 : data_idx_og + 1],
            )
            new_wr[data_idx_new] = np.interp(
                s,
                self.t[data_idx_og - 1 : data_idx_og + 1],
                self.wr[data_idx_og - 1 : data_idx_og + 1],
            )
            new_t[data_idx_new] = s
            data_idx_new += 1

        self.t = new_t
        self.wl = new_wl
        self.wr = new_wr
        self._clear_cache()

    @functools.cached_property
    def w(self):
        return (
            self.intrinsics.radius_r * self.wr - self.intrinsics.radius_l * self.wl
        ) / self.intrinsics.baseline

    @functools.cached_property
    def v(self):
        return (
            self.intrinsics.radius_r * self.wr + self.intrinsics.radius_l * self.wl
        ) / 2


@dataclass(kw_only=True)
class FeatData(BaseData):
    ids: list[np.ndarray]
    stereo_pts: list[np.ndarray]
    extrinsics: gtsam.Pose3
    intrinsics: gtsam.Cal3_S2Stereo


class Dataset:
    def extrinsics(self, sensor: Sensor) -> gtsam.Pose3:
        raise NotImplementedError()

    def intrinsics(self, sensor: Sensor) -> Union[WheelIntrinsics, gtsam.Cal3_S2Stereo]:
        raise NotImplementedError()

    def stamps(self):
        raise NotImplementedError()

    def data(self, sensor: Sensor) -> Union[IMUData, CameraData, GTData, WheelData]:
        raise NotImplementedError()

    def noise(self, sensor: Sensor):
        raise NotImplementedError()


class Dataset2JRL:
    def __init__(self, data: Dataset, N: int = None) -> None:
        self.data = data

        self.stamps = self.data.stamps[:N]
        self.N = self.stamps.shape[0]
        self.clear_factors()

        self.funcs = {
            Sensor.PRIOR: self._prior_factor,
            Sensor.IMU: self._imu_factor,
            Sensor.CAM: self._cam_factor,
            Sensor.WHEEL: self._wheel_factor,
            Sensor.FEAT: self._feat_factor,
        }

        self.writer = makeRoseWriter()
        self.parser = makeRoseParser()

        # Things that should be updated as we go
        self.traj = {}
        self.vio_T_x0 = None

    def __add__(self, other: Dataset2JRL):
        N = min(len(self.factor_graphs), len(other.factor_graphs))
        for i in range(N):
            self.factor_graphs[i].push_back(other.factor_graphs[i])
            self.factor_tags[i].extend(other.factor_tags[i])

        self.traj.update(other.traj)
        if self.vio_T_x0 is None:
            self.vio_T_x0 = other.vio_T_x0

        return self

    def load_cache(self, filename: str):
        previous = self.parser.parseDataset(str(filename), False)
        for i, mm in enumerate(previous.measurements("a")):
            if i >= self.N:
                break
            self.factor_graphs[i].push_back(mm.measurements)
            self.factor_tags[i].extend(mm.measurement_types)

        if previous.containsGroundTruth():
            self.traj[Sensor.GT] = previous.groundTruth("a")

    def clear_factors(self):
        self.factor_graphs = [gtsam.NonlinearFactorGraph() for _ in range(self.N)]
        self.factor_tags = [[] for _ in range(self.N)]

    def to_dataset(self) -> jrl.Dataset:
        builder = jrl.DatasetBuilder(self.data.name, ["a"])
        for i, (graph, tag) in enumerate(zip(self.factor_graphs, self.factor_tags)):
            builder.addEntry("a", i, graph, tag)

        if Sensor.GT in self.traj:
            builder.addGroundTruth("a", values2typedvalues(self.traj[Sensor.GT]))

        return builder.build()

    def save_dataset(self, filename: Path, extra: str = ""):
        builder = jrl.DatasetBuilder(self.data.name + extra, ["a"])
        for i, (graph, tag) in enumerate(zip(self.factor_graphs, self.factor_tags)):
            builder.addEntry("a", i, graph, tag)

        if Sensor.GT in self.traj:
            builder.addGroundTruth("a", values2typedvalues(self.traj[Sensor.GT]))

        dataset = builder.build()
        self.writer.writeDataset(dataset, str(filename), False)

    def get_ground_truth(self):
        gt = self.data.data(Sensor.GT)
        values = gtsam.Values()

        wheel_extrinsics = self.data.extrinsics(Sensor.WHEEL)
        values.insert(W(0), wheel_extrinsics)

        # g-frame = GPS frame on body
        # i-frame = IMU frame on body
        # UTM = default origin of GPS data
        # VIO = origin of state estimate data
        i_T_g = gt.extrinsics
        g_T_i = i_T_g.inverse()
        vio_T_xi0 = self.vio_T_x0

        gt_idx_prev = 0
        gt_idx = 1
        for i, stamp in enumerate(self.stamps):
            # Increment till we find the right stamp
            while gt_idx < gt.shape and gt.t[gt_idx] <= stamp:
                gt_idx += 1

            if i == 0:
                # Find transform from UTM origin to VIO origin
                xg0_T_utm = gt[gt_idx - 1].inverse()
                vio_T_utm = vio_T_xi0 * i_T_g * xg0_T_utm

            utm_T_xgi = gt[gt_idx - 1]

            if not np.isnan(utm_T_xgi.matrix()).any():
                vio_T_xii = vio_T_utm * utm_T_xgi * g_T_i

            values.insert(X(i), vio_T_xii)

            if gt.bias_w is not None and gt.bias_a is not None:
                values.insert(
                    B(i),
                    gtsam.imuBias.ConstantBias(
                        gt.bias_a[gt_idx - 1], gt.bias_w[gt_idx - 1]
                    ),
                )
            if gt.vel is not None:
                values.insert(V(i), gt.vel[gt_idx - 1])

            if gt.slip is not None:
                values.insert(S(i), gt.slip[gt_idx_prev:gt_idx].mean(axis=0))
            if gt.intrinsics is not None:
                values.insert(I(i), gt.intrinsics[gt_idx - 1])

            gt_idx_prev = gt_idx

            if gt_idx >= gt.shape:
                break

        self.utm_T_vio = vio_T_utm.inverse()
        self.traj[Sensor.GT] = values
        return values

    def save_traj(self, sensor: Sensor, filename: Path):
        if sensor not in self.traj:
            print(f"{sensor} hasn't been loaded")

        self.writer.writeResults(
            values2results(self.traj[sensor], sensor.name, self.data.name),
            str(filename),
            False,
        )

    def add_factors(self, sensor: Sensor, **kwargs):
        self.funcs[sensor](**kwargs)

    def _prior_factor(self, use_gt_orien=False):
        prior_noise: PriorNoise = self.data.noise(Sensor.PRIOR)

        # GT Orientation Initialization
        if use_gt_orien:
            gt_data = self.data.data(Sensor.GT)
            gt_idx = 1
            while gt_data.t[gt_idx] < self.stamps[0]:
                gt_idx += 1
            # Align initialization with GT init
            vio_R_x0 = gt_data[gt_idx - 1].rotation()
            vio_T_x0 = gtsam.Pose3(vio_R_x0, [0, 0, 0])

            self.bias0 = gtsam.imuBias.ConstantBias()

            vel0 = gt_data.vel[gt_idx - 1]
        else:
            imu_data = self.data.data(Sensor.IMU)
            # Find IMU indices to initialize
            end = 1
            while imu_data.t[end] < self.stamps[0]:
                end += 1
            start = max(0, end - 200)

            # IMU Orientation Initialization
            ez_W = np.array([0, 0, 1])
            e_acc = np.mean(imu_data.a[start:end], axis=0)
            e_acc /= np.linalg.norm(e_acc)
            angle = np.arccos(ez_W @ e_acc)
            poseDelta = np.zeros(6)
            poseDelta[:3] = np.cross(ez_W, e_acc)
            poseDelta[:3] *= angle / np.linalg.norm(poseDelta[:3])
            vio_T_x0 = gtsam.Pose3.Expmap(-poseDelta)

            # Figure out bias estimates
            avg_w = np.mean(imu_data.w[start:end], axis=0)
            vio_R_x0 = vio_T_x0.rotation().matrix()
            avg_a = np.mean(imu_data.a[start:end], axis=0)
            # TODO: Pull gravity from pim params
            # Remove gravity from VIO frame, then back to body frame
            avg_a = vio_R_x0.T @ (vio_R_x0 @ avg_a + np.array([0, 0, -9.81]))
            # If any of them are too large, zero 'em out cuz they're probably wrong
            avg_a[avg_a > 0.1] = 0

            self.bias0 = gtsam.imuBias.ConstantBias(avg_a, avg_w)

            # Add all priors!
            vel0 = np.zeros(3)

        # TODO: Do some sort of sanity check on initial bias values
        # print("Initial Bias: ", self.bias0)

        intr0 = self.data.data(Sensor.WHEEL).intrinsics.vector()
        self.factor_graphs[0].addPriorPose3(
            X(0), vio_T_x0, gtsam.noiseModel.Isotropic.Sigma(6, prior_noise.sig_pose)
        )
        # TODO: Loosen this in case we're moving at the start?
        self.factor_graphs[0].addPriorPoint3(
            V(0), vel0, gtsam.noiseModel.Isotropic.Sigma(3, prior_noise.sig_vel)
        )
        self.factor_graphs[0].addPriorConstantBias(
            B(0), self.bias0, gtsam.noiseModel.Isotropic.Sigma(6, prior_noise.sig_bias)
        )

        self.factor_graphs[0].addPriorPoint3(
            I(0), intr0, gtsam.noiseModel.Isotropic.Sigma(3, prior_noise.sig_intr)
        )

        self.factor_tags[0].append(jrl.PriorFactorPose3Tag)
        self.factor_tags[0].append(jrl.PriorFactorPoint3Tag)
        self.factor_tags[0].append(jrl.PriorFactorIMUBiasTag)
        self.factor_tags[0].append(jrl.PriorFactorPoint3Tag)

        self.vio_T_x0 = vio_T_x0

    def _imu_factor(self):
        imu_data = self.data.data(Sensor.IMU)
        pim = imu_data.noise.gtsam()

        pim.resetIntegrationAndSetBias(self.bias0)

        # Find the start
        imu_data_idx = 1
        while imu_data.t[imu_data_idx] < self.stamps[0]:
            imu_data_idx += 1

        # Iterate through making the graphs
        for i, stamp in enumerate(self.stamps[1:], start=1):
            while imu_data.t[imu_data_idx] < stamp:
                wm = imu_data.w[imu_data_idx]
                am = imu_data.a[imu_data_idx]
                dt = imu_data.dt(imu_data_idx)
                pim.integrateMeasurement(am, wm, dt)
                imu_data_idx += 1
                if imu_data_idx >= imu_data.shape:
                    break

            imu_factor = gtsam.CombinedImuFactor(
                X(i - 1), V(i - 1), X(i), V(i), B(i - 1), B(i), pim
            )
            self.factor_graphs[i].push_back(imu_factor)
            self.factor_tags[i].append(jrl.CombinedIMUTag)
            pim.resetIntegrationAndSetBias(self.bias0)
            if imu_data_idx >= imu_data.shape:
                break

    def _cam_factor(self, show: bool = False):
        cam_data = self.data.data(Sensor.CAM)
        noise = cam_data.noise.gtsam()

        prev = cam_data[0]
        prev.fill_out_kps()

        bar = tqdm(total=self.N, leave=False)
        bar.update()

        for i, t in enumerate(self.stamps[1:], start=1):
            # Propagate keypoints
            curr = cam_data[i]
            matches, dist = prev.propagate(curr)
            # TODO: If skip not matches, should try matching with previous frame?

            if show:  # and i > 360:
                img = cv2.cvtColor(curr.l, cv2.COLOR_GRAY2RGB)
                for kp_idx_prev, kp_idx_curr in matches:
                    a, b = curr.kps[kp_idx_curr]
                    c, d = prev.kps[kp_idx_prev]

                    img = cv2.circle(img, (int(a), int(b)), 3, (0, 255, 0), 1)
                    img = cv2.line(
                        img, (int(a), int(b)), (int(c), int(d)), (0, 0, 255), 1
                    )

                cv2.imshow("features", img)
                cv2.waitKey(1)

            for kp_idx_prev, kp_idx_curr in matches:
                id = curr.ids[kp_idx_curr]
                stereo_prev = prev.stereopts
                stereo_curr = curr.stereopts

                # If it wasn't seen previously, move id back
                if prev.ids[kp_idx_prev] == 0:
                    prev.ids[kp_idx_prev] = id
                    factor_prev = gtsam.GenericStereoFactor3D(
                        stereo_prev[kp_idx_prev],
                        noise,
                        X(i - 1),
                        L(id),
                        cam_data.intrinsics.stereo,
                        cam_data.extrinsics,
                    )
                    self.factor_graphs[i].push_back(factor_prev)
                    self.factor_tags[i].append(jrl.StereoFactorPose3Point3Tag)

                factor_curr = gtsam.GenericStereoFactor3D(
                    stereo_curr[kp_idx_curr],
                    noise,
                    X(i),
                    L(id),
                    cam_data.intrinsics.stereo,
                    cam_data.extrinsics,
                )
                self.factor_graphs[i].push_back(factor_curr)
                self.factor_tags[i].append(jrl.StereoFactorPose3Point3Tag)

            prev = curr
            prev.fill_out_kps()

            bar.update()
            bar.set_description(f"Matches: {matches.shape[0]}, ||p||: {dist:.04f}")

        if show:
            cv2.destroyWindow("features")

    def _wheel_factor(self):
        data = self.data.data(Sensor.WHEEL)
        pwmParams = data.noise.gtsam(data.intrinsics)

        # The baseline struggles with the tighter noise. Loosen it some
        # 8 was empirically chosen as a good value for loosening
        dang_noise = data.noise.copy()
        dang_noise.sigma_rad_s *= 8
        pwmParamsDang = dang_noise.gtsam(data.intrinsics)

        data.interp(self.stamps)

        allPWMs = [
            PreintegratedWheelBaseline(pwmParamsDang),
            PreintegratedWheelRose(pwmParams),
            PreintegratedWheelRose(pwmParams),
            PreintegratedWheelRose(pwmParams),
            PreintegratedWheelRose(pwmParams),
        ]
        pwmFactors = [
            lambda i, pwm: WheelFactor2(X(i - 1), X(i), pwm.copy(), data.extrinsics),
            lambda i, pwm: WheelFactor2(X(i - 1), X(i), pwm.copy(), data.extrinsics),
            lambda i, pwm: WheelFactor3(
                X(i - 1), X(i), S(i), pwm.copy(), data.extrinsics
            ),
            lambda i, pwm: WheelFactor4Intrinsics(
                X(i - 1), I(i - 1), X(i), I(i), pwm.copy(), data.extrinsics
            ),
            lambda i, pwm: WheelFactor5(
                X(i - 1), I(i - 1), X(i), I(i), S(i), pwm.copy(), data.extrinsics
            ),
        ]
        pwmTags = [
            WheelBaselineTag,
            WheelRoseTag,
            WheelRoseSlipTag,
            WheelRoseIntrTag,
            WheelRoseIntrSlipTag,
        ]

        wheel_traj = gtsam.Values()
        Xprev = self.vio_T_x0
        wheel_traj.insert(X(0), Xprev)

        # Find the start
        wheel_data_idx = 1
        while data.t[wheel_data_idx] < self.stamps[0]:
            wheel_data_idx += 1

        # Iterate through making the graphs
        for i, stamp in tqdm(
            enumerate(self.stamps[1:], start=1),
            total=self.stamps.size - 1,
            leave=False,
        ):
            while data.t[wheel_data_idx] <= stamp:
                wl = data.wl[wheel_data_idx]
                wr = data.wr[wheel_data_idx]
                dt = data.dt(wheel_data_idx)

                wheel_data_idx += 1
                if wheel_data_idx >= data.shape:
                    break

                if dt < 1e-4:
                    continue

                for pwm in allPWMs:
                    pwm.integrateMeasurements(wl, wr, dt)

            # Store an integrated trajectory for plotting
            pwmIdx = 1
            Xcurr = pwmFactors[pwmIdx](i, allPWMs[pwmIdx]).predict(Xprev)
            wheel_traj.insert(X(i), Xcurr)
            Xprev = Xcurr

            if allPWMs[0].deltaTij() == 0:
                print(
                    f"\tWarning, interval {i} had no wheel messages, assuming 0 movement.."
                )
                dt = (self.stamps[i] - self.stamps[i - 1]) / 1e9
                for pwm in allPWMs:
                    pwm.integrateMeasurements(0, 0, dt)

            for pwm, factor, tag in zip(allPWMs, pwmFactors, pwmTags):
                f = factor(i, pwm)
                self.factor_graphs[i].push_back(f)
                self.factor_tags[i].append(tag)

                pwm.resetIntegration()

            if wheel_data_idx >= data.shape:
                break

            # RP Prior Factor
            planar = PlanarPriorFactor(X(i), np.eye(2) * data.noise.sig_rp_prior**2)
            self.factor_graphs[i].push_back(planar)
            self.factor_tags[i].append(PlanarPriorTag)

            # Z Prior Factor
            z = ZPriorFactor(X(i), np.eye(1) * data.noise.sig_z_prior**2)
            self.factor_graphs[i].push_back(z)
            self.factor_tags[i].append(ZPriorTag)

            # Intrinsics Prior
            intrinsics_prior = gtsam.PriorFactorPoint3(
                I(i),
                data.intrinsics,
                gtsam.noiseModel.Diagonal.Sigmas(
                    [
                        data.noise.sig_intr_prior_baseline,
                        data.noise.sig_intr_prior_radius,
                        data.noise.sig_intr_prior_radius,
                    ]
                ),
            )
            self.factor_graphs[i].push_back(intrinsics_prior)
            self.factor_tags[i].append(PriorFactorIntrinsicsTag)

            # Slip prior
            slip_robust = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Tukey.Create(data.noise.slip_prior_kernel),
                gtsam.noiseModel.Isotropic.Sigma(2, data.noise.sig_slip_prior),
            )
            slip = gtsam.PriorFactorPoint2(
                S(i),
                np.zeros(2),
                slip_robust,
            )
            self.factor_graphs[i].push_back(slip)
            self.factor_tags[i].append(jrl.PriorFactorPoint2Tag)

        self.traj[Sensor.WHEEL] = wheel_traj

    def _feat_factor(self):
        data = self.data.data(Sensor.FEAT)
        noise = gtsam.noiseModel.Isotropic.Sigma(3, data.noise.sig_pix)

        id_seen = set()
        feat_waiting = {}
        for i, stamp in enumerate(self.stamps):
            for id, stereo_pt in zip(data.ids[i], data.stereo_pts[i]):
                factor_cam = gtsam.GenericStereoFactor3D(
                    gtsam.StereoPoint2(stereo_pt),
                    noise,
                    X(i),
                    L(id),
                    data.intrinsics,
                    data.extrinsics,
                )

                # If it's been seen before
                if id in id_seen:
                    self.factor_graphs[i].push_back(factor_cam)
                    self.factor_tags[i].append(jrl.StereoFactorPose3Point3Tag)
                # If this is the second time it's been seen
                elif id in feat_waiting.keys():
                    factor_cam_og = feat_waiting.pop(id)
                    # Check to make sure the state hasn't been marginalized out
                    i_before = gtsam.Symbol(factor_cam_og.keys()[0]).index()
                    if i - i_before > 5:
                        feat_waiting[id] = factor_cam
                    else:
                        self.factor_graphs[i].push_back(factor_cam)
                        self.factor_graphs[i].push_back(factor_cam_og)
                        self.factor_tags[i].append(jrl.StereoFactorPose3Point3Tag)
                        self.factor_tags[i].append(jrl.StereoFactorPose3Point3Tag)
                        id_seen.add(id)
                else:
                    feat_waiting[id] = factor_cam
