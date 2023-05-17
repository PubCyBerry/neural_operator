import os
import os.path as osp
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from matplotlib import animation, gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image

mpl.rcParams["animation.embed_limit"] = 2**128


def plot_solution(
    xs: torch.tensor,
    ts: torch.tensor,
    U: torch.tensor,
    U_pred: torch.tensor = None,
    ps: torch.tensor = None,
    save_img: bool = True,
    title: str = None,
    filename: str = None,
    result_dir: str = "results",
):
    fig, ax = plt.subplots()

    # Additional margin for title
    extra_top_margin: float = 0
    if title is not None:
        extra_top_margin = 0.08

    # Additional space for preds / error plots
    ncol: int = 1
    width_margin: float = 0
    if U_pred is not None:
        if U.shape == U_pred.shape:
            ncol = 3
        else:
            ncol = 2
        width_margin = 0.5

    gs0 = gridspec.GridSpec(1, ncol)
    gs0.update(
        top=1 - 0.06 - extra_top_margin,
        bottom=1 - 1.0 / 2.0 + 0.06,
        left=0.15,
        right=0.85,
        wspace=width_margin,
    )

    # t = 0, 25, 50, 75(%)
    snapshot_idx = torch.arange(0, len(ts), step=len(ts) * 0.25, dtype=int)

    # Row 0: Plot Solution (+ Prediction & Error)
    imshow_config = dict(
        interpolation="nearest",
        cmap="rainbow",
        extent=[ts.min(), ts.max(), xs.min(), xs.max()],
        origin="lower",
        aspect="auto",
    )

    # Draw vertical line at time to visualize
    ax = plt.subplot(gs0[0, 0])
    line = torch.linspace(xs.min(), xs.max(), 2)[:, None]
    for t in ts[snapshot_idx]:
        ax.plot(t * torch.ones((2, 1)), line, "w-", linewidth=1)

    # Plot collocation points
    if ps is not None:
        ax.plot(
            ps[:, 0],
            ps[:, 1],
            "kx",
            label="Data (%d points)" % (len(ps)),
            markersize=4,  # marker size doubled
            clip_on=False,
            alpha=0.5,
        )
        ax.legend(
            loc="upper center",
            bbox_to_anchor=(0.9, -0.05),
            ncol=5,
            frameon=False,
            prop={"size": 15},
        )

    imgs = [U]
    ax_titles = ["$u(x,t)$"]
    ax_xlabels = ["$t$"]
    ax_ylabels = ["$x$"]
    if U_pred is not None:
        if U.shape == U_pred.shape:
            imgs = [U, U_pred, abs(U - U_pred)]
            ax_titles = ["Ground Truth", "Prediction", "Error"]
        else:
            imgs = [U, U_pred]
            ax_titles = ["Ground Truth", "Prediction"]
        ax_xlabels = ax_xlabels * ncol
        ax_ylabels = ax_ylabels * ncol

    for col, img, ax_title, ax_xlabel, ax_ylabel in zip(
        range(ncol), imgs, ax_titles, ax_xlabels, ax_ylabels
    ):
        ax = plt.subplot(gs0[0, col])
        ax.set_title(ax_title, fontsize=20)
        ax.set_xlabel(ax_xlabel, fontsize=20)
        ax.set_ylabel(ax_ylabel, fontsize=20)
        h = ax.imshow(img, **imshow_config)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="3%", pad=0.10)
    cbar = fig.colorbar(h, cax=cax)
    cbar.ax.tick_params(labelsize=15)

    # Row 1: Plot solution at certain timestamps(t=0.00, 0.25, 0.50, 0.75)(%)
    ncol = len(snapshot_idx)
    gs1 = gridspec.GridSpec(1, ncol)
    gs1.update(top=1 - 1.0 / 2.0 - 0.1, bottom=0.10, wspace=0.4)

    for col, idx in zip(range(ncol), snapshot_idx):
        ax = plt.subplot(gs1[0, col])
        ax.plot(xs, U[:, idx], "b-", linewidth=2, label="Exact")
        if U_pred is not None:
            ax.plot(xs, U_pred[:, idx], "r--", linewidth=2, label="Prediction")
        ax.set_xlabel("$x$")
        if col == 0:
            ax.set_ylabel("$u(t,x)$")
        ax.set_title("$t=%.2fs$" % (ts[idx]), fontsize=8)
        ax.axis("square")
        ax.set_xlim((xs.min(), xs.max()))
        ax.set_ylim((U.min() - 0.2, U.max() + 0.2))
        ax.tick_params(axis="both", which="both", labelsize=6)
        ax.grid(which="both", axis="both", linestyle="--")

    if U_pred is not None:
        ax = plt.subplot(gs1[0, 0])
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    if title is not None:
        fig.suptitle(f"{title.replace('models.','')}", fontsize=18)

    # save image
    if save_img and filename is not None:
        os.makedirs(result_dir, exist_ok=True)

        # save plot to a memory
        # # [Option 1] use Pillow
        # canvas = plt.get_current_fig_manager().canvas
        # canvas.draw()
        # img = Image.frombytes("RGB", canvas.get_width_height(), canvas.tostring_rgb())
        # img.save(fp=osp.join(result_dir, filename.split(".")[-1]) + ".png")

        # [Option 2] use matplotlib
        plt.savefig(osp.join(result_dir, filename.split(".")[-1]) + ".png")

    # return img
    return fig


def animate_solution(
    xs: torch.tensor,
    ys: torch.tensor,
    ts: torch.tensor,
    y_pred: torch.tensor = None,
    interval: int = 50,
    fps: int = 60,
    save_img: bool = True,
    filename: str = None,
    result_dir: str = "results",
):
    """
    plot time evolution animation of solution
    x: x axis <- (num_x)
    y: u(x,t) <- (num_x, num_t)
    interval: time interval between frames (ms)
    """

    dt = ts[1] - ts[0]

    fig = plt.figure()
    ax = plt.axes(xlim=(xs.min() - 0.2, xs.max() + 0.2), ylim=(ys.min() - 0.2, ys.max() + 0.2))
    (line,) = ax.plot([], [], "b-", linewidth=2, label="Exact")
    (pred,) = ax.plot([], [], "r--", linewidth=2, label="Prediction")
    ax.set_title("Waveform $u(x,t)$ at t = 0 seconds", fontsize=16)
    ax.set_xlabel(r"$x \ [\mathrm{m}]$", fontsize=14)
    ax.set_ylabel(r"$u(x) \ [\mathrm{m}/s]$", fontsize=14)
    ax.grid(which="both", axis="both", linestyle="--")

    if y_pred is not None:
        ax.legend()

    def init():
        line.set_data([], [])
        pred.set_data([], [])
        return (
            line,
            pred,
        )

    # animation function.  This is called sequentially
    def animate(i):
        line.set_data(xs, ys[:, i])
        if y_pred is not None:
            pred.set_data(xs, y_pred[:, i])
        ax.set_title(
            f"Waveform $u(x,t)$ at t = {torch.round(dt * i, decimals=2):.2f} seconds",
            fontsize=14,
        )
        return (line,)

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=ys.shape[1], interval=interval, blit=True
    )

    # save image
    if save_img and filename is not None:
        os.makedirs(result_dir, exist_ok=True)
        anim.save(
            filename=osp.join(result_dir, filename.split(".")[-1]) + ".gif",
            writer="imagemagick",
            fps=fps,
        )
        # Writer = animation.writers['ffmpeg']
        # writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=10000)
        # anim.save('lines.mp4', writer=writer)

    return anim
