import functools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import cv2
import gtsam
import numpy as np
from rose.dataset import (
    BaseIntrinsics,
    BaseNoise,
    CameraData,
    CameraIntrinsics,
    CamNoise,
    Dataset,
    GTData,
    IMUData,
    IMUNoise,
    Sensor,
    WheelData,
    WheelIntrinsics,
    WheelNoise,
)
from rose.rose_python import PreintegratedWheelParams

# https://www.xsens.com/hubfs/Downloads/Leaflets/MTi-300.pdf
SIGMA_A = 60 * (1 / 10**6) * (1 / 9.81)
SIGMA_G = 0.01 * np.pi / 180
# this is not correct parameter, but should be close
SIGMA_BA = 15 * (1 / 10**6) * (1 / 9.81)
# again not the correct parameter, but should be close
SIGMA_BG = 10 * (np.pi / 180) * (1 / 3600)
PREINT_COV = 1.0e-8
PREINT_BIAS_COV = 1.0e-5

SIGMA_PIXEL = 1


class KaistDataset(Dataset):
    def __init__(self, dir: Path) -> None:
        self.dir = dir
        self.name = dir.stem

        self.noise_override = {}

        self.calibration_files = {
            Sensor.IMU: dir / "calibration/Vehicle2IMU.txt",
            Sensor.CAM: dir / "calibration/Vehicle2Stereo.txt",
            # Wheel & GT data is at origin of vehicle (See openVINS)
            Sensor.GT: None,
            Sensor.WHEEL: None,
        }

        self.intrinsics_file = {
            Sensor.CAM: {
                "l": dir / "calibration/left.yaml",
                "r": dir / "calibration/right.yaml",
            },
            Sensor.WHEEL: dir / "calibration/EncoderParameter.txt",
        }

        self.stamp_file = dir / "sensor_data/stereo_stamp.csv"
        self.data_files = {
            Sensor.IMU: dir / "sensor_data/xsens_imu.csv",
            Sensor.CAM: {
                "l": dir / "image/stereo_left/",
                "r": dir / "image/stereo_right/",
                "d": dir / "image/stereo_disp/",
            },
            Sensor.GT: dir / "global_pose.csv",
            Sensor.WHEEL: dir / "sensor_data/encoder.csv",
        }

    def add_noise(self, sensor: Sensor, noise):
        self.noise_override[sensor] = noise
        self.noise.cache_clear()

    @functools.cache
    def _calfile2pose(self, sensor: Sensor) -> gtsam.Pose3:
        """
        Load various vehicle sensor extrinsics
        """
        cal_file = self.calibration_files[sensor]
        if cal_file is None:
            return gtsam.Pose3.Identity()

        with open(cal_file, "r") as f:
            lines = f.readlines()

        R = np.array([float(i) for i in lines[3].split()[1:]]).reshape((3, 3))
        t = np.array([float(i) for i in lines[4].split()[1:]])
        return gtsam.Pose3(gtsam.Rot3(R), t)

    @functools.cache
    def extrinsics(self, sensor: Sensor) -> gtsam.Pose3:
        vc_T_imu = self._calfile2pose(Sensor.IMU)
        vc_T_sens = self._calfile2pose(sensor)

        imu_T_sens = vc_T_imu.inverse() * vc_T_sens

        # If it's a camera pose, we also want to add rectification rotation on
        if sensor == Sensor.CAM:
            left = cv2.FileStorage(
                str(self.intrinsics_file[sensor]["l"]), cv2.FILE_STORAGE_READ
            )
            # after_R_before rectification
            a_R_b = left.getNode("rectification_matrix").mat()
            b_T_a = gtsam.Pose3(gtsam.Rot3(a_R_b).inverse(), [0, 0, 0])
            imu_T_sens = imu_T_sens * b_T_a

        return imu_T_sens

    @functools.cache
    def intrinsics(self, sensor: Sensor) -> BaseIntrinsics:
        """
        Load various vehicle sensor intrinsics
        """
        if sensor not in self.intrinsics_file:
            raise ValueError("Intrinsics not defined for this sensor")

        if sensor == Sensor.CAM:
            left = cv2.FileStorage(
                str(self.intrinsics_file[sensor]["l"]), cv2.FILE_STORAGE_READ
            )
            right = cv2.FileStorage(
                str(self.intrinsics_file[sensor]["r"]), cv2.FILE_STORAGE_READ
            )
            w = int(left.getNode("image_width").real())
            h = int(left.getNode("image_height").real())
            K_l = left.getNode("camera_matrix").mat()
            K_r = right.getNode("camera_matrix").mat()
            R_l = left.getNode("rectification_matrix").mat()
            R_r = right.getNode("rectification_matrix").mat()
            P_l = left.getNode("projection_matrix").mat()
            P_r = right.getNode("projection_matrix").mat()
            dist_l = left.getNode("distortion_coefficients").mat()
            dist_r = right.getNode("distortion_coefficients").mat()

            mapx_l, mapy_l = cv2.initUndistortRectifyMap(
                K_l, dist_l, R_l, P_l, (w, h), 5
            )
            mapx_r, mapy_r = cv2.initUndistortRectifyMap(
                K_r, dist_r, R_r, P_r, (w, h), 5
            )

            baseline = -P_r[0, 3] / P_r[1, 1]
            return CameraIntrinsics(
                fx=float(P_r[0, 0]),
                fy=float(P_r[1, 1]),
                cx=float(P_r[0, 2]),
                cy=float(P_r[1, 2]),
                baseline=float(baseline),
                mapx_l=mapx_l,
                mapy_l=mapy_l,
                mapx_r=mapx_r,
                mapy_r=mapy_r,
            )

        elif sensor == Sensor.WHEEL:
            with open(self.intrinsics_file[sensor], "r") as f:
                lines = f.readlines()
            self.wheel_resolution = int(lines[1].split()[-1])
            radius_l = float(lines[2].split()[-1]) / 2
            radius_r = float(lines[3].split()[-1]) / 2
            baseline = float(lines[4].split()[-1])
            return WheelIntrinsics(
                baseline=baseline,
                radius_l=radius_l,
                radius_r=radius_r,
            )

    @functools.cached_property
    def stamps(self):
        stamps = np.loadtxt(self.stamp_file, dtype=np.int64)
        dirs = self.data_files[Sensor.CAM]
        img_exists = np.logical_and(
            [(dirs["l"] / f"{s}.png").exists() for s in stamps],
            [(dirs["r"] / f"{s}.png").exists() for s in stamps],
        )
        return stamps[img_exists]

    @functools.cache
    def data(self, sensor: Sensor) -> Union[IMUData, CameraData, GTData, WheelData]:
        if sensor == Sensor.IMU:
            t = np.loadtxt(
                self.data_files[sensor], delimiter=",", usecols=0, dtype=np.int64
            )
            data = np.loadtxt(
                self.data_files[sensor], delimiter=",", usecols=range(8, 14)
            )
            w = data[:, 0:3]
            a = data[:, 3:6]
            return IMUData(t=t, w=w, a=a, noise=self.noise(Sensor.IMU))

        elif sensor == Sensor.CAM:
            dirs = self.data_files[sensor]
            stamps = self.stamps
            left = [dirs["l"] / f"{s}.png" for s in stamps]
            right = [dirs["r"] / f"{s}.png" for s in stamps]
            disp = [dirs["d"] / f"{s}.png" for s in stamps]
            return CameraData(
                t=stamps,
                left=left,
                right=right,
                disparity=disp,
                extrinsics=self.extrinsics(Sensor.CAM),
                intrinsics=self.intrinsics(Sensor.CAM),
                noise=self.noise(Sensor.CAM),
            )

        elif sensor == Sensor.GT:
            t = np.loadtxt(
                self.data_files[sensor], delimiter=",", usecols=0, dtype=np.int64
            )
            x = np.loadtxt(
                self.data_files[sensor], delimiter=",", usecols=range(1, 13)
            ).reshape((-1, 3, 4))

            return GTData(t=t, x=x, extrinsics=self.extrinsics(Sensor.GT))

        elif sensor == Sensor.WHEEL:
            # Load wheel resolution
            self.intrinsics(Sensor.WHEEL)

            t = np.loadtxt(
                self.data_files[sensor], delimiter=",", usecols=0, dtype=np.int64
            )
            data = np.loadtxt(
                self.data_files[sensor],
                delimiter=",",
                usecols=range(1, 3),
                dtype=np.int64,
            )

            # convert encoders into velocities
            dt = np.diff(t).astype(np.float32) / 1e9
            rps_r = np.diff(data[:, 1]).astype(np.float32) / self.wheel_resolution / dt
            rps_l = np.diff(data[:, 0]).astype(np.float32) / self.wheel_resolution / dt

            w_l = 2 * np.pi * rps_l
            w_r = 2 * np.pi * rps_r

            return WheelData(
                t=t[1:],
                wl=w_l,
                wr=w_r,
                extrinsics=self.extrinsics(Sensor.WHEEL),
                intrinsics=self.intrinsics(Sensor.WHEEL),
                noise=self.noise(Sensor.WHEEL),
            )

    @functools.cache
    def noise(self, sensor: Sensor) -> Union[
        gtsam.PreintegratedCombinedMeasurements,
        gtsam.noiseModel.Base,
        PreintegratedWheelParams,
    ]:
        if sensor == Sensor.IMU:
            return IMUNoise(
                sigma_a=SIGMA_A,
                sigma_w=SIGMA_G,
                sigma_ba=SIGMA_BA,
                sigma_bw=SIGMA_BG,
                preint_cov=PREINT_COV,
                preint_bias_cov=PREINT_BIAS_COV,
            )

        elif sensor == Sensor.CAM:
            return CamNoise(sig_pix=SIGMA_PIXEL)

        elif sensor == Sensor.WHEEL:
            wheel_noise = self.noise_override.get(
                Sensor.WHEEL, WheelNoise(sigma_rad_s=25)
            )
            # Convert raw encoder sigma to continuous time sigma
            # enc -> wheel % -> radians -> radians / second -> (radians / second) (1 / \sqrt{Hz})
            wheel_noise.sigma_rad_s = (
                (wheel_noise.sigma_rad_s / self.wheel_resolution)
                * (2 * np.pi)
                / (1 / 100)
                / np.sqrt(100)
            )
            return wheel_noise
