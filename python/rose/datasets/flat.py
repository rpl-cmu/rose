import functools
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Union

import cv2
import gtsam
import numpy as np
import yaml
from rose.dataset import (
    BaseData,
    BaseIntrinsics,
    BaseNoise,
    CameraData,
    CameraIntrinsics,
    CamNoise,
    Dataset,
    GTData,
    IMUData,
    IMUNoise,
    PriorNoise,
    Sensor,
    WheelData,
    WheelIntrinsics,
    WheelNoise,
)


class FlatDataset(Dataset):
    def __init__(self, dir: Path) -> None:
        self.dir = dir
        self.name = dir.stem

        self.intrinsics_override = {}
        self.noise_override = {}
        self.extrinsics_override = {}

    def clear_cache(self):
        self.extrinsics.cache_clear()
        self.intrinsics.cache_clear()
        self.noise.cache_clear()
        self.data.cache_clear()

    def add_intrinsics(self, sensor: Sensor, intrinsics):
        self.intrinsics_override[sensor] = intrinsics
        self.intrinsics.cache_clear()

    def add_noise(self, sensor: Sensor, noise):
        self.noise_override[sensor] = noise
        self.noise.cache_clear()

    def add_extrinsics(self, sensor: Sensor, extrinsics):
        self.extrinsics_override[sensor] = extrinsics
        self.extrinsics.cache_clear()

    def loadyaml(self, file: Path) -> dict:
        with open(file, "r") as f:
            y = yaml.safe_load(f)
            return y

    @functools.cache
    def extrinsics(self, sensor: Sensor) -> gtsam.Pose3:
        if sensor in self.extrinsics_override:
            return self.extrinsics_override[sensor]

        d = self.dir / f"calibration/ext_{sensor.name.lower()}.txt"
        return gtsam.Pose3(np.loadtxt(d))

    @functools.cache
    def intrinsics(self, sensor: Sensor) -> BaseIntrinsics:
        """
        Load various vehicle sensor intrinsics
        """
        if sensor in self.intrinsics_override:
            return self.intrinsics_override[sensor]

        d = self.dir / f"calibration/int_{sensor.name.lower()}.yaml"
        intrinsics = self.loadyaml(d)

        if sensor == Sensor.CAM:
            return CameraIntrinsics(**intrinsics)
        elif sensor == Sensor.WHEEL:
            return WheelIntrinsics(**intrinsics)

        return intrinsics

    @functools.cached_property
    def stamps(self):
        stamps = np.loadtxt(self.dir / "stamps.txt", dtype=np.int64)
        img_exists = np.logical_and(
            [(self.dir / f"left/{s}.png").exists() for s in stamps],
            [(self.dir / f"right/{s}.png").exists() for s in stamps],
        )
        return stamps[img_exists]

    @functools.cache
    def data(self, sensor: Sensor) -> BaseData:
        d = self.dir / f"{sensor.name.lower()}.txt"

        if sensor == Sensor.IMU:
            t = np.loadtxt(d, usecols=0, dtype=np.int64)
            data = np.loadtxt(d, usecols=range(1, 7))
            w = data[:, 0:3]
            a = data[:, 3:6]
            return IMUData(t=t, w=w, a=a, noise=self.noise(Sensor.IMU))

        elif sensor == Sensor.CAM:
            stamps = self.stamps
            left = [self.dir / f"left/{str(s)}.png" for s in stamps]
            right = [self.dir / f"right/{str(s)}.png" for s in stamps]
            disp = [self.dir / f"disp/{str(s)}.png" for s in stamps]
            mask_file = self.dir / "mask.txt"
            if mask_file.exists():
                mask = np.loadtxt(self.dir / "mask.txt", dtype=bool)
            else:
                mask = None
            return CameraData(
                t=stamps,
                left=left,
                right=right,
                disparity=disp,
                mask=mask,
                extrinsics=self.extrinsics(Sensor.CAM),
                intrinsics=self.intrinsics(Sensor.CAM),
                noise=self.noise(Sensor.CAM),
            )

        elif sensor == Sensor.GT:
            t = np.loadtxt(d, usecols=0, dtype=np.int64)
            x = np.loadtxt(d, usecols=range(1, 13)).reshape((-1, 3, 4))
            return GTData(t=t, x=x, extrinsics=self.extrinsics(Sensor.GT))

        elif sensor == Sensor.WHEEL:
            t = np.loadtxt(d, usecols=0, dtype=np.int64)
            wl, wr = np.loadtxt(d, usecols=range(1, 3)).T

            return WheelData(
                t=t,
                wl=wl,
                wr=wr,
                extrinsics=self.extrinsics(Sensor.WHEEL),
                intrinsics=self.intrinsics(Sensor.WHEEL),
                noise=self.noise(Sensor.WHEEL),
            )

    @functools.cache
    def noise(self, sensor: Sensor) -> BaseNoise:
        if sensor in self.noise_override:
            noise = self.noise_override[sensor].dict()
        else:
            d = self.dir / f"noise/{sensor.name.lower()}.yaml"
            noise = self.loadyaml(d)
        if sensor == Sensor.IMU:
            return IMUNoise(**noise)

        elif sensor == Sensor.CAM:
            return CamNoise(**noise)

        elif sensor == Sensor.WHEEL:
            return WheelNoise(**noise)

        elif sensor == Sensor.PRIOR:
            return PriorNoise(**noise)
