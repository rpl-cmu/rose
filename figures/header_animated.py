from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rose.jrl
import rose.plot as plot
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rose.dataset import Dataset2JRL, Sensor
from rose import FlatDataset
from tqdm import tqdm


def animate(traj):
    map_file = "/home/contagon/catkin_new/src/multicam_frontend_ros/config/gascola_imagery_august_2020/gascola_august_2020.vrt"

    # ------------------------- Load in topographical map ------------------------- #
    dst_crs = "EPSG:32617"
    src = rasterio.open(map_file)

    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds
    )
    kwargs = src.meta.copy()
    kwargs.update(
        {"crs": dst_crs, "transform": transform, "width": width, "height": height}
    )

    destination = np.zeros((height, width, 3), dtype=src.dtypes[0])

    for i in range(1, 4):
        reproject(
            source=rasterio.band(src, i),
            destination=destination[:, :, i - 1],
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
    destination[destination < 0] = 200
    destination = destination.squeeze()
    print("----Loaded topographical map")

    # ------------------------- Load all data ------------------------- #
    RESULTS = Path(f"data/{traj}")

    # Have to load this to get transform utm_T_vio
    dataset = FlatDataset(Path(f"/mnt/data/flat/{traj}"))
    data = Dataset2JRL(dataset)
    data.add_factors(Sensor.PRIOR)
    data.get_ground_truth()

    types = {
        plot.WheelType.GT.value: "_gt.jrr",
        plot.WheelType.WHEEL.value: "_wheel.jrr",
        plot.WheelType.WHEEL_PLANAR.value: "f-base.wheel_baseline.planar_prior.jrr",
        plot.WheelType.WHEEL_ROSE_INTR_SLIP.value: "f-base.wheel_intr_slip.jrr",
    }

    results = {}
    for name, filename in types.items():
        values = (
            rose.jrl.makeRoseParser()
            .parseResults(str(RESULTS / filename), False)
            .robot_solutions["a"]
            .values
        )
        results[name] = plot.load_full_state(values, offset=data.utm_T_vio)

    print("----Data Loaded")

    # ------------------------- Plot data ------------------------- #
    ratio = 16 / 9
    width = 8

    colors = plot.setup_plot()
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(width, width / ratio))

    min_x, max_x, min_y, max_y = np.inf, -np.inf, np.inf, -np.inf
    for name, values in results.items():
        if min_x > np.min(values["$p_y$"]):
            min_x = np.min(values["$p_y$"])
        if max_x < np.max(values["$p_y$"]):
            max_x = np.max(values["$p_y$"])
        if min_y > np.min(values["$p_x$"]):
            min_y = np.min(values["$p_x$"])
        if max_y < np.max(values["$p_x$"]):
            max_y = np.max(values["$p_x$"])

    # Give a little padding
    diff = 25
    min_x -= diff
    max_x += diff
    min_y -= diff
    max_y += diff

    # Get the aspect ratio of the map correct
    height = max_y - min_y
    width = max_x - min_x

    if width / height > ratio:
        diff = (width / ratio - height) / 2
        min_y -= diff
        max_y += diff
    else:
        diff = (height * ratio - width) / 2
        min_x -= diff
        max_x += diff

    x1, y1 = ~transform * (min_x, min_y)
    x2, y2 = ~transform * (max_x, max_y)
    final_map = destination[int(y2) : int(y1), int(x1) : int(x2)]
    extent = [0, max_x - min_x, 0, max_y - min_y]

    plt.imshow(final_map, extent=extent, cmap="grey", alpha=0.55)

    # dark_ones = [plot.WheelType.WHEEL_ROSE_INTR_SLIP.value, plot.WheelType.GT.value]
    lines = {}
    for name, values in results.items():
        alpha = 1  # if name in dark_ones else 0.9
        lw = 1.25  # if name in dark_ones else 0.8
        (lines[name],) = ax.plot(
            [],
            [],
            label=name,
            c=colors[name],
            lw=lw,
            alpha=alpha,
        )

    ax.set_aspect("equal")
    # ax.set_xlabel("East (m)")
    # ax.set_ylabel("North (m)")
    # ax.tick_params(axis="both", pad=-1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    ob = plot.AnchoredHScaleBar(
        size=50,
        extent=0.02,
        label="50m",
        loc=1,
        frameon=True,
        ppad=1.75,
        sep=1,
        linekw=dict(color=colors["GT"], lw=0.6),
        framecolor=(0.6, 0.6, 0.6),
        # label_outline=0.75,
    )
    ax.add_artist(ob)
    ax.legend(loc="lower right", ncol=1)

    fps = 30
    t = 5

    num_frames = fps * t
    N = len(results[plot.WheelType.GT.value]["$p_y$"])
    show_every = N // num_frames
    loop = tqdm(total=num_frames, leave=False, desc="Animating")

    def update(frame):
        for name, values in results.items():
            lines[name].set_data(
                values["$p_y$"][: show_every * frame + 1] - min_x,
                values["$p_x$"][: show_every * frame + 1] - min_y,
            )
        loop.update(1)
        return lines.values()

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        blit=True,
        interval=1000 / fps,
        cache_frame_data=False,
        repeat=False,
    )
    ani.save(f"figures/animations/{traj}.gif", dpi=300)


if __name__ == "__main__":
    trajs = [p.stem for p in Path("/mnt/bags/best/").iterdir()]

    for t in trajs:
        try:
            print(t)
            animate(t)
        except Exception as _:
            print(f"Failed on {t}")
            continue
