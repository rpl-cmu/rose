from pathlib import Path

import earthpy.spatial as es
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import rose.jrl
import rose.plot as plot
import seaborn as sns
from rasterio.warp import Resampling, calculate_default_transform, reproject
from rose.dataset import Dataset2JRL, Sensor
from rose.flat import FlatDataset

if __name__ == "__main__":
    traj = "all_2024-02-09-14-08-29"

    is_top = True
    map_file = "/home/contagon/Downloads/ned19_n40x50_w080x00_pa_southwest_2006.img"

    is_top = False
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

    depth = 1 if is_top else 3

    destination = np.zeros((height, width, depth), dtype=src.dtypes[0])

    for i in range(1, depth + 1):
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
    print("Loaded topographical map")

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

    print("Data Loaded")
    # ------------------------- Plot data ------------------------- #
    colors = plot.setup_plot()
    fig, ax = plt.subplots(1, 1, layout="constrained", figsize=(4, 4))

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

    diff = 25
    min_x -= diff
    max_x += diff
    min_y -= diff
    max_y += diff

    x1, y1 = ~transform * (min_x, min_y)
    x2, y2 = ~transform * (max_x, max_y)
    final_map = destination[int(y2) : int(y1), int(x1) : int(x2)]
    extent = [0, max_x - min_x, 0, max_y - min_y]

    if is_top:
        cmap = sns.diverging_palette(220, 20, s=40, as_cmap=True)
        plt.imshow(final_map, extent=extent, cmap=cmap)
        plt.colorbar()

        hillshade = es.hillshade(final_map, azimuth=90, altitude=1)
        plt.imshow(hillshade, extent=extent, cmap="Greys", alpha=0.2)
    else:
        plt.imshow(final_map, extent=extent, cmap="grey", alpha=0.55)

    dark_ones = [plot.WheelType.WHEEL_ROSE_INTR_SLIP.value, plot.WheelType.GT.value]
    for name, values in results.items():
        alpha = 1 if name in dark_ones else 0.9
        lw = 1.25 if name in dark_ones else 0.8
        ax.plot(
            values["$p_y$"] - min_x,
            values["$p_x$"] - min_y,
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

    fig.legend(loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.07))

    plt.savefig("figures/header.png", bbox_inches="tight", dpi=300)
    plt.savefig("figures/header.pdf", bbox_inches="tight", dpi=300)

    # plt.show()
