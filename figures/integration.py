import rose.plot as plot
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.animation as animation
from tqdm import tqdm

f = lambda x, y: -(x**2) - y**2  # noqa: E731
fx = lambda t: 8 * np.sin(t * 3 * np.pi / 2) ** 2  # noqa: E731
fy = lambda t: -8 * t  # noqa: E731

t = 5
fps = 30

num_frames = t * fps


def run(tangent, make_animation, filename):
    c = plot.setup_plot()
    c = sns.color_palette("colorblind")

    fig, ax = plt.subplots(1, 1, figsize=(6, 6), subplot_kw=dict(projection="3d"))
    plt.axis("off")

    # ------------------------- Plot group ------------------------- #
    x = np.linspace(-4, 10, 100)
    y = np.linspace(-10, 4, 100)
    X, Y = np.meshgrid(x, y)
    Z = f(X, Y)

    ax.plot_surface(
        X, Y, Z, color=c[0], shade=True, linewidth=0.05, alpha=0.7, zorder=1
    )
    ax.view_init(30, -50)
    plt.subplots_adjust(top=1.0, bottom=0, left=0, right=1.0)

    # ------------------------- Plot trajectory ------------------------- #
    t = np.linspace(0, 1, num_frames, endpoint=True)
    fxx = fx(t)
    fyy = fy(t)
    fzz = f(fxx, fyy)
    zero = np.zeros_like(fxx)

    (traj,) = ax.plot([], [], [], color=c[3], lw=2, zorder=3)
    ax.plot(0, 0, 0, "k.", ms=10, zorder=3)
    (final,) = ax.plot([], [], [], "k.", ms=10, zorder=3)

    # ------------------------- Plot tangent space ------------------------- #
    if tangent:
        ax.plot_surface(
            X, Y, 0 * X, color=c[7], shade=True, linewidth=0.05, alpha=0.4, zorder=3
        )
        (final_tangent,) = ax.plot([], [], [], "k.", ms=10, zorder=3)

    # ------------------------- Animate ------------------------- #
    if not make_animation:
        # traj.set_data(fxx, fyy)
        # traj.set_3d_properties(zero)
        # final.set_data(fxx[-1:], fyy[-1:])
        # final.set_3d_properties(fzz[-1:])
        # if tangent:
        #     final_tangent.set_data(fxx[-1:], fyy[-1:])
        #     final_tangent.set_3d_properties(zero[-1:])
        #     end_tangent = np.array([fxx[-1], fyy[-1], 0])
        #     end_group = np.array([fxx[-1], fyy[-1], fzz[-1]])
        #     ax.arrow3D(
        #         *end_tangent,
        #         *(end_group - end_tangent),
        #         color="k",
        #         zorder=3,
        #         mutation_scale=6,
        #     )

        plt.savefig(filename, dpi=300, transparent=True)
        return

    loop = tqdm(total=num_frames, leave=False, desc="Animating")

    def update(frame):
        traj.set_data(fxx[:frame], fyy[:frame])
        if tangent:
            traj.set_3d_properties(zero[:frame])
        else:
            traj.set_3d_properties(fzz[:frame])
        loop.update(1)
        if frame == num_frames:
            final.set_data(fxx[-1:], fyy[-1:])
            final.set_3d_properties(fzz[-1:])
            if tangent:
                final_tangent.set_data(fxx[-1:], fyy[-1:])
                final_tangent.set_3d_properties(zero[-1:])
                end_tangent = np.array([fxx[-1], fyy[-1], 0])
                end_group = np.array([fxx[-1], fyy[-1], fzz[-1]])
                ax.arrow3D(
                    *end_tangent,
                    *(end_group - end_tangent),
                    color="k",
                    zorder=3,
                    mutation_scale=6,
                )
        return [traj]

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames + 1,
        blit=True,
        interval=1000 / fps,
        cache_frame_data=True,
    )
    ani.save(
        filename,
        codec="png",
        dpi=600,
        savefig_kwargs={"transparent": True, "facecolor": "none"},
        extra_args=["-loop", "-1"],
    )


if __name__ == "__main__":
    run(True, False, "figures/tangent_space.png")
    run(False, True, "figures/integration.gif")
    run(True, True, "figures/integration_tangent.gif")
