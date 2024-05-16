import os
import re
from enum import Enum
from pathlib import Path

import cv2
import gtsam
import numpy as np
import rosbag
import yaml
import yourdfpy
from cv_bridge import CvBridge

from .dataset import CameraIntrinsics, CamNoise, IMUNoise, WheelIntrinsics, WheelNoise

URDF_DIR = os.environ["URDF_DIR"]
SCALE = 672 / 5376

# TODO: Save noise??


# ------------------------- Helpers for grizzlyROS bags ------------------------- #
class Sensor(Enum):
    ADV = 1
    KVH = 2
    VIS = 3
    GPS = 4
    QUAD0 = 5
    QUAD1 = 6
    WHEELL = 7
    WHEELR = 8


class GrizzlyBag:
    def __init__(self, bagname: Path, modules: list[int] = [0, 1, 2]) -> None:
        self.name = bagname.stem
        self.bag = rosbag.Bag(bagname, skip_index=True)
        self.modules = modules

        types_topics = self.bag.get_type_and_topic_info()[1]
        for k, v in types_topics.items():
            if v.msg_type == "unicorn_sensor_msgs/ImageSet":
                self.robot_name = k.split("/")[1]
                break

        self.bridge = CvBridge()

        self.sensor2topic = {
            Sensor.ADV: f"/{self.robot_name}/ins",
            Sensor.KVH: f"/{self.robot_name}/kvh/raw",
            Sensor.VIS: f"/{self.robot_name}/visodo",
            Sensor.GPS: f"/{self.robot_name}/ins",
            Sensor.QUAD0: f"/{self.robot_name}/quadcam/visodo/0",
            Sensor.QUAD1: f"/{self.robot_name}/quadcam/visodo/1",
            Sensor.WHEELL: f"/{self.robot_name}/left_wheel/current_speed",
            Sensor.WHEELR: f"/{self.robot_name}/right_wheel/current_speed",
        }

        self.sensor2parser = {
            Sensor.ADV: self._get_imu,
            Sensor.KVH: None,  # TODO
            Sensor.VIS: self._get_image,
            Sensor.GPS: self._get_gps,
            Sensor.QUAD0: self._get_quad,
            Sensor.QUAD1: self._get_quad,
            Sensor.WHEELL: self._get_wheel,
            Sensor.WHEELR: self._get_wheel,
        }

    # ------------------------- Getting data from bag ------------------------- #
    def get_msgs(
        self, sensor: Sensor, just_stamp=False, f=None
    ) -> tuple[np.ndarray, list]:
        topic = self.sensor2topic[sensor]
        if f is None:
            if just_stamp:
                f = lambda msg, t, stamp, data: stamp.append(msg.header.stamp.to_nsec())
            else:
                f = self.sensor2parser[sensor]

        stamp = []
        data = []
        for topic, msg, t in self.bag.read_messages(topic):
            if f is not None:
                f(msg, t, stamp, data)

        if len(data) > 0 and type(data[0]) in [list, float, int]:
            data = np.array(data)

        return np.array(stamp, dtype=np.int64), data

    def _get_image(self, msg, t, stamp, modules):
        """
        This can be used if all the images are saved under a single topic for unicorn type.

        For each timestep, we have 3 modules that each have left, right, and rgb images.
        """
        data = {i: {} for i in self.modules}
        for i in msg.images:
            # Find the module because it is not consistent via indexing
            name = i.header.frame_id
            module_num = int(re.match(r"camera_module[0-9]", name).group()[-1]) - 1

            if module_num in self.modules:
                # Find which camera in each module it is
                if "nir0" in name:
                    cam_type = "left"
                elif "nir1" in name:
                    cam_type = "right"
                elif "rgb" in name:
                    cam_type = "rgb"
                else:
                    raise ValueError("Unknown image type!")

                # Read the image and put it in the data structure
                img = self.bridge.imgmsg_to_cv2(i, desired_encoding="passthrough")
                data[module_num][cam_type] = img

        stamp.append(msg.header.stamp.to_nsec())
        modules.append(data)

    def _get_imu(self, msg, t, stamp, data):
        data.append(
            [
                msg.gyroscope_x_raw,
                msg.gyroscope_y_raw,
                msg.gyroscope_z_raw,
                msg.accelerometer_x_raw,
                msg.accelerometer_y_raw,
                msg.accelerometer_z_raw,
            ]
        )
        stamp.append(msg.header.stamp.to_nsec())

    def _get_gps(self, msg, t, stamp, data):
        """Save in END format"""
        rot = gtsam.Rot3.RzRyRx(msg.roll, msg.pitch, msg.heading)
        t = [msg.utm_northing, msg.utm_easting, -msg.utm_height]
        pose = gtsam.Pose3(rot, t)

        data.append(pose.matrix()[:3].flatten().tolist())
        stamp.append(msg.header.stamp.to_nsec())

    def _get_quad(self, msg, t, stamp, data):
        pose = msg.pose.pose
        data.append(
            [
                pose.orientation.w,
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.position.x,
                pose.position.y,
                pose.position.z,
            ]
        )
        # TODO!! Using ROS timestamp here instead, not as accurate
        # Needs to be swapped if CAM times are ever synced
        stamp.append(t.to_nsec())

    def _get_wheel(self, msg, t, stamp, data):
        # TODO!! No timestamp on wheel odometry?
        # Convert to rad/s
        data.append(msg.data * 2 * np.pi)
        stamp.append(t.to_nsec())

    # ------------------------- Getting data from YAML / URDF ------------------------- #
    def intrinsics(self, sensor: Sensor, module=None):
        if sensor == Sensor.VIS:
            if module is None:
                raise ValueError("You have to choose a module for camera intrinsics")

            dir = f"{URDF_DIR}/{self.robot_name}/current/calibration/"
            file = os.path.join(dir, f"camera_module{module+1}_nir1.yaml")
            with open(file, "r") as stream:
                parameters = yaml.safe_load(stream)

            # Being lazy here, be careful 1 is scaled as well
            P = (
                np.array(parameters["projection_matrix"]["data"]).reshape((3, 4))
                * SCALE
            )

            baseline = -P[0, 3] / P[1, 1]
            return CameraIntrinsics(
                fx=float(P[0, 0]),
                fy=float(P[1, 1]),
                cx=float(P[0, 2]),
                cy=float(P[1, 2]),
                baseline=float(baseline),
            )

        elif sensor == Sensor.WHEELL or sensor == Sensor.WHEELR:
            # TODO: Pull this from aidtr-urdf somehow
            r = 0.15605
            b = 1.6226
            return WheelIntrinsics(baseline=b, radius_l=r, radius_r=r)

    def extrinsics(self, sensor: Sensor, module=None):
        urdf = yourdfpy.URDF.load(
            f"{URDF_DIR}/{self.robot_name}/current/urdf/sensors.urdf"
        )
        imu_frame = "imu_NED"

        if sensor == Sensor.ADV:
            frame = imu_frame
        elif sensor == Sensor.VIS:
            if module is None:
                raise ValueError("You have to choose a module for camera intrinsics")
            frame = f"camera_module{module+1}_nir0_rect"
        elif sensor == Sensor.WHEELL or sensor == Sensor.WHEELR:
            # TODO: Pull this from URDF?
            x, y, z, w = 1, 0, 0, 0
            p = [0.5878, 0, 1.5611]
            return gtsam.Pose3(gtsam.Rot3(w, x, y, z), p)
        elif sensor == Sensor.GPS:
            frame = imu_frame

        return gtsam.Pose3(urdf.get_transform(frame, imu_frame))

    def noise(self, sensor: Sensor):
        if sensor == Sensor.ADV:
            return IMUNoise(
                sigma_a=9.8100e-04,
                sigma_w=6.9813e-05,
                sigma_ba=1.9620e-04,
                sigma_bw=1.4544e-05,
                preint_cov=1.0e-8,
                preint_bias_cov=1.0e-5,
            )
        elif sensor == Sensor.WHEELL or sensor == Sensor.WHEELR:
            return WheelNoise(sigma_rad_s=0.2)
        elif sensor == Sensor.VIS:
            return CamNoise()

    def to_flat(self, dir: Path, module=1):
        outdir = dir / self.name
        dir_cal = outdir / "calibration"
        dir_noise = outdir / "noise"
        dir_left = outdir / "left"
        dir_right = outdir / "right"
        dir_disp = outdir / "disp"
        outdir.mkdir(exist_ok=True)
        dir_cal.mkdir(exist_ok=True)
        dir_left.mkdir(exist_ok=True)
        dir_right.mkdir(exist_ok=True)
        dir_disp.mkdir(exist_ok=True)
        dir_noise.mkdir(exist_ok=True)

        # ------------------------- save extrinsics ------------------------- #
        np.savetxt(
            dir_cal / "ext_wheel.txt", self.extrinsics(Sensor.WHEELL).matrix()[:3]
        )
        np.savetxt(
            dir_cal / "ext_cam.txt",
            self.extrinsics(Sensor.VIS, module).matrix()[:3],
        )
        np.savetxt(dir_cal / "ext_gt.txt", self.extrinsics(Sensor.GPS).matrix()[:3])

        # ------------------------- save intrinsics & noise------------------------- #
        # intrinsics
        self.saveyaml(
            dir_cal / "int_cam.yaml", self.intrinsics(Sensor.VIS, module).dict()
        )
        self.saveyaml(dir_cal / "int_wheel.yaml", self.intrinsics(Sensor.WHEELL).dict())

        # noise
        self.saveyaml(dir_noise / "cam.yaml", self.noise(Sensor.VIS).dict())
        self.saveyaml(dir_noise / "imu.yaml", self.noise(Sensor.ADV).dict())
        self.saveyaml(dir_noise / "wheel.yaml", self.noise(Sensor.WHEELL).dict())

        # ------------------------- save data ------------------------- #
        stamps_cam, _ = self.get_msgs(Sensor.VIS, just_stamp=True)
        stamps_imu, imu = self.get_msgs(Sensor.ADV)
        stamps_wl, wl = self.get_msgs(Sensor.WHEELL)
        stamps_wr, wr = self.get_msgs(Sensor.WHEELR)

        # Check if any images need to be skipped over till we get wheel & imu data
        first_cam_idx = 0
        while (
            stamps_cam[first_cam_idx] < stamps_imu[0]
            or stamps_cam[first_cam_idx] < stamps_wl[0]
        ):
            first_cam_idx += 1

        skip_cam = np.arange(0, first_cam_idx).tolist()
        if len(skip_cam) != 0:
            print(
                f"\tSkipping the first {first_cam_idx} images while waiting for IMU/Wheel data"
            )

        # Camera data
        # Remove any out of order data
        dt = np.diff(stamps_cam) / 1e9
        bad_idx = np.where(np.logical_or(dt >= 1.0, dt <= 0))[0] + 1
        bad_idx = bad_idx.tolist()
        bad_idx.extend(skip_cam)
        global idx
        idx = 0

        # Save camera data image by image so we don't overflow memory
        def save_cam(msg, t_dummy, stamp_dummy, data_dummy):
            global idx
            if idx in bad_idx:
                idx += 1
                return

            saved = {}
            for i in msg.images:
                # Find the module because it is not consistent via indexing
                name = i.header.frame_id
                module_num = int(re.match(r"camera_module[0-9]", name).group()[-1]) - 1

                # Find which camera in each module it is
                if "nir0" in name:
                    cam_type = "left"
                elif "nir1" in name:
                    cam_type = "right"
                elif "rgb" in name:
                    cam_type = "rgb"
                else:
                    raise ValueError("Unknown image type!")

                # Read the image and put it in the data structure
                if cam_type != "rgb" and module_num == module:
                    saved[cam_type] = self.bridge.imgmsg_to_cv2(
                        i, desired_encoding="passthrough"
                    )

            # If missing an image skip!
            if "left" not in saved or "right" not in saved:
                bad_idx.append(idx)
                idx += 1
                return

            t = msg.header.stamp.to_nsec()
            file_l = dir_left / f"{t}.png"
            file_r = dir_right / f"{t}.png"
            if not file_l.exists():
                cv2.imwrite(str(file_l), saved["left"])
            if not file_r.exists():
                cv2.imwrite(str(file_r), saved["right"])

            idx += 1

        self.get_msgs(Sensor.VIS, f=save_cam)

        if len(bad_idx) != 0:
            print(
                f"\tRemoving {len(bad_idx)} / {stamps_cam.size} images due to bad timestamps..."
            )
        stamps_cam = np.delete(stamps_cam, bad_idx, axis=0)
        np.savetxt(outdir / "stamps.txt", stamps_cam, fmt="%i")

        # IMU data
        self.savetxt(outdir / "imu.txt", stamps_imu, imu)

        # Wheel data
        # They appear to be within about 1e-4 seconds of eachother
        size = min(stamps_wr.size, stamps_wl.size)
        wheel = np.column_stack((wl[:size], wr[:size]))
        # Drop any extra messages that came through
        self.savetxt(outdir / "wheel.txt", stamps_wl[:size], wheel)

        stamps, gps = self.get_msgs(Sensor.GPS)
        self.savetxt(outdir / "gt.txt", stamps, gps)

        return outdir

    def savetxt(self, file: Path, stamps: np.ndarray, data: np.ndarray):
        assert stamps.shape[0] == data.shape[0]

        dtype = [(str(i), float) for i in range(data.shape[1])]
        dtype.insert(0, ("stamps", np.int64))

        out = np.zeros(stamps.size, dtype=dtype)
        out["stamps"] = stamps
        for i in range(data.shape[1]):
            out[str(i)] = data[:, i]

        fmt = "%i " + " ".join(["%.18e" for _ in range(data.shape[1])])

        np.savetxt(file, out, fmt=fmt)

    def saveyaml(self, file: Path, data: dict):
        with open(file, "w") as outfile:
            yaml.dump(data, outfile, default_flow_style=False)
