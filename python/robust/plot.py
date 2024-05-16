from enum import Enum

import gtsam
import jrl
import matplotlib
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gtsam.symbol_shorthand import B, I, M, S, V, X
from matplotlib.lines import Line2D
from robust.robust_python import WheelCovIntrSlipTag, WheelCovTag, WheelDangTag


# https://stackoverflow.com/a/43343934
class AnchoredHScaleBar(matplotlib.offsetbox.AnchoredOffsetbox):
    """size: length of bar in data units
    extent : height of bar ends in axes units"""

    def __init__(
        self,
        size=1,  # number of units to show
        extent=0.03,  # length of vertical bar
        label="",  # label to insert
        loc=2,  # which corner in should be on
        ax=None,  # which axes
        pad=0.4,  # not really sure tbh
        borderpad=0.5,  # padding to outside edge
        ppad=0,  # space between lines and outside of box
        sep=2,  # distance between words and line
        prop=None,
        frameon=True,
        linekw={},
        label_outline=0,
        framecolor="black",
        **kwargs,
    ):
        if not ax:
            ax = plt.gca()
        trans = ax.get_xaxis_transform()
        size_bar = matplotlib.offsetbox.AuxTransformBox(trans)
        line = Line2D([0, size], [0, 0], **linekw)
        vline1 = Line2D([0, 0], [-extent / 2.0, extent / 2.0], **linekw)
        vline2 = Line2D([size, size], [-extent / 2.0, extent / 2.0], **linekw)
        size_bar.add_artist(line)
        size_bar.add_artist(vline1)
        size_bar.add_artist(vline2)
        txt = matplotlib.offsetbox.TextArea(
            label,
            textprops={
                "path_effects": [
                    pe.withStroke(linewidth=label_outline, foreground="white")
                ]
            },
        )
        self.vpac = matplotlib.offsetbox.VPacker(
            children=[size_bar, txt], align="center", pad=ppad, sep=sep
        )
        matplotlib.offsetbox.AnchoredOffsetbox.__init__(
            self,
            loc,
            pad=pad,
            borderpad=borderpad,
            child=self.vpac,
            prop=prop,
            frameon=frameon,
            **kwargs,
        )
        self.patch.set_edgecolor(framecolor)


class WheelType(Enum):
    GT = "GT"
    SVO = "SVO"
    WHEEL = "WO"
    WHEEL_PLANAR = "SVO + Planar"
    WHEEL_UNDER = "SVO + Under"
    WHEEL_COV = "SVO + ROSE--"
    WHEEL_COV_INTR_SLIP = "SVO + ROSE"


def tags_to_names():
    return {
        "GT": WheelType.GT.value,
        "stereo": WheelType.SVO.value,
        "wheel": WheelType.WHEEL.value,
        WheelCovTag: WheelType.WHEEL_COV.value,
        WheelCovIntrSlipTag: WheelType.WHEEL_COV_INTR_SLIP.value,
        WheelDangTag: WheelType.WHEEL_PLANAR.value,
        "under": WheelType.WHEEL_UNDER.value,
    }


def setup_plot():
    matplotlib.rc("pdf", fonttype=42)
    sns.set_context("paper")
    sns.set_style("whitegrid")
    sns.set_palette("colorblind")
    c = sns.color_palette("colorblind")

    # Make sure you install times & clear matplotlib cache
    # https://stackoverflow.com/a/49884009
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["mathtext.fontset"] = "stix"

    map_colors = {
        WheelType.GT.value: (0.2, 0.2, 0.2),
        WheelType.SVO.value: c[0],
        WheelType.WHEEL.value: c[1],
        WheelType.WHEEL_COV.value: c[2],
        WheelType.WHEEL_COV_INTR_SLIP.value: c[3],
        WheelType.WHEEL_PLANAR.value: c[4],
        WheelType.WHEEL_UNDER.value: c[-1],
    }
    return map_colors


def load_state(values, idx, use_body_velocity=False, offset=gtsam.Pose3.Identity()):
    state = np.zeros(23)
    x = gtsam.Pose3.Identity()
    if values.exists(X(idx)):
        x = offset * values.atPose3(X(idx))
        state[:3] = x.rotation().rpy() * 180 / np.pi
        state[3:6] = x.translation()
    if values.exists(V(idx)):
        v = values.atPoint3(V(idx))
        if use_body_velocity:
            v = x.rotation().matrix().T @ v
        state[6:9] = v
    if values.exists(B(idx)):
        b = values.atConstantBias(B(idx))
        state[9:15] = b.vector()
    if values.exists(S(idx)):
        s = values.atPoint2(S(idx))
        state[18:20] = s
    if values.exists(I(idx)):
        i = values.atPoint3(I(idx))
        state[20:23] = i
    return state


def load_full_state(values, n=None, pandas=True, offset=gtsam.Pose3.Identity()):
    if n is None:
        n = max_pose(values)

    state = np.zeros((n, 23))
    for i in range(n):
        state[i] = load_state(values, i, offset=offset)

    if pandas:
        columns = [
            "roll (deg)",
            "pitch (deg)",
            "yaw (deg)",
            r"$p_x$",
            r"$p_y$",
            r"$p_z$",
            r"$v_x$",
            r"$v_y$",
            r"$v_z$",
            r"$b_{ax}$",
            r"$b_{ay}$",
            r"$b_{az}$",
            r"$b_{gx}$",
            r"$b_{gy}$",
            r"$b_{gz}$",
            r"$m_1$",
            r"$m_2$",
            r"$m_3$",
            r"$s_l$",
            r"$s_r$",
            r"$b$",
            r"$r_l$",
            r"$r_r$",
        ]
        df = pd.DataFrame(state, columns=columns)

        return df
    else:
        return state


def max_pose(values):
    return (
        max([gtsam.Symbol(k).index() for k in gtsam.utilities.allPose3s(values).keys()])
        + 1
    )


def plot_3d_trajectory(
    figtitle=None, filename=None, show=False, align=False, **solutions
):
    import open3d as o3d

    # ------------------------- Plot in open3D ------------------------- #
    geo = []
    for traj_idx, (name, values) in enumerate(solutions.items()):
        poses = gtsam.utilities.allPose3s(values)
        for k in poses.keys():
            mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
            p = poses.atPose3(k)
            mesh.transform(p.matrix())
            geo.append(mesh)

    o3d.visualization.draw_geometries(geo)


def plot_xy_trajectory(
    figtitle=None,
    filename=None,
    show=False,
    align=False,
    N=None,
    states=None,
    **solutions,
):
    c = list(setup_plot().values())

    fig, ax = plt.subplots(
        2,
        2,
        layout="constrained",
        figsize=(8, 6),
        dpi=147,
        gridspec_kw={"width_ratios": [2, 1], "height_ratios": [2, 1]},
    )
    ax = ax.T.flatten()

    if figtitle is not None:
        fig.suptitle(figtitle)

    # ------------------------- Plot lines ------------------------- #
    if N is None:
        N = min([max_pose(values) for name, values in solutions.items()])

    for traj_idx, (name, values) in enumerate(solutions.items()):
        plot_values = (
            jrl.alignPose3(gtsam.utilities.allPose3s(values), solutions["GT"], False)
            if align
            else values
        )

        state = np.zeros((N, 23))
        for i in range(N):
            state[i] = load_state(plot_values, i)

        c_idx = traj_idx % len(c)

        ax[0].plot(state[:, 3], state[:, 4], label=name, c=c[c_idx])
        ax[1].plot(state[:, 3], state[:, 5], label=name, c=c[c_idx])
        ax[2].plot(state[:, 5], state[:, 4], label=name, c=c[c_idx])

    ax[0].legend()
    ax[0].set_xlabel("x")
    ax[0].set_ylabel("y")
    ax[0].set_aspect("equal", adjustable="box")

    ax[1].set_xlabel("x")
    ax[1].set_ylabel("z")
    ax[1].set_aspect("equal", adjustable="box")

    ax[2].set_xlabel("z")
    ax[2].set_ylabel("y")
    ax[2].invert_xaxis()
    ax[2].set_aspect("equal", adjustable="box")

    ax[3].axis("off")

    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    return fig, ax


def plot_error_state(
    figtitle=None, filename=None, show=False, align=False, **solutions
):
    c = list(setup_plot().values())

    fig, ax = plt.subplots(3, 2, layout="constrained", figsize=(10, 5), dpi=147)
    ax = ax.T.flatten()
    names = [
        "error - roll (deg)",
        "error - pitch (deg)",
        "error - yaw (deg)",
        r"error - $p_x$",
        r"error - $p_y$",
        r"error - $p_z$",
    ]
    for i in range(6):
        ax[i].set_title(names[i])

    if figtitle is not None:
        fig.suptitle(figtitle)

    # ------------------------- Plot lines ------------------------- #
    N = min([max_pose(values) for name, values in solutions.items()])

    # Find gt first
    gt = solutions["GT"]
    solutions.pop("GT")
    state_gt = np.zeros((N, 23))
    for i in range(N):
        state_gt[i] = load_state(gt, i)

    for traj_idx, (name, values) in enumerate(solutions.items()):
        plot_values = (
            jrl.alignPose3(values, solutions["GT"], False) if align else values
        )

        state = np.zeros((N, 23))
        for i in range(N):
            state[i] = load_state(plot_values, i)

        c_idx = traj_idx % len(c)
        for i in range(6):
            if i < 6 or not np.all(state[:, i] == 0):
                ax[i].plot(np.abs(state[:, i] - state_gt[:, i]), label=name, c=c[c_idx])

    ax[0].legend()
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    return fig, ax


def plot_state(
    figtitle=None,
    filename=None,
    show=False,
    align=False,
    states="rpv",
    N=None,
    **solutions,
):
    c = list(setup_plot().values())

    names = []
    if "r" in states:
        names.extend(["roll (deg)", "pitch (deg)", "yaw (deg)"])
    if "p" in states:
        names.extend([r"$p_x$", r"$p_y$", r"$p_z$"])
    if "v" in states:
        names.extend([r"$v_x$", r"$v_y$", r"$v_z$"])
    if "b" in states:
        names.extend([r"$b_{ax}$", r"$b_{ay}$", r"$b_{az}$"])
        names.extend([r"$b_{gx}$", r"$b_{gy}$", r"$b_{gz}$"])
    if "m" in states:
        names.extend([r"$m_1$", r"$m_2$", r"$m_3$"])
    if "i" in states:
        names.extend([r"$b$", r"$r_l$", r"$r_r$"])
    if "s" in states:
        names.extend([r"$s_l$", r"$s_r$", ""])

    always_show = ["roll (deg)", "pitch (deg)", "yaw (deg)"]

    n_states = len(names)
    fig, ax = plt.subplots(
        3, n_states // 3, layout="constrained", figsize=(10, 5), dpi=147
    )
    ax = ax.T.flatten()

    for i in range(n_states):
        ax[i].set_title(names[i])

    if figtitle is not None:
        fig.suptitle(figtitle)

    # ------------------------- Plot lines ------------------------- #
    if N is None:
        N = min([max_pose(values) for name, values in solutions.items()])

    for traj_idx, (name, values) in enumerate(solutions.items()):
        plot_values = (
            jrl.alignPose3(values, solutions["GT"], False) if align else values
        )

        state = load_full_state(plot_values, N)

        c_idx = traj_idx % len(c)
        zorder = 100 if name == "GT" else 0
        alpha = 0.75 if name == "GT" else 1
        for i, n in enumerate(names):
            if n in state and (not np.all(state[n][1:] == 0) or n in always_show):
                ax[i].plot(state[n], label=name, c=c[c_idx], zorder=zorder, alpha=alpha)

    for a in ax:
        a.ticklabel_format(useOffset=False)

    ax[0].legend().set_zorder(102)
    if filename is not None:
        plt.savefig(filename)
    if show:
        plt.show()

    return fig, ax
