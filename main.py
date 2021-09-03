from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import (
    AutoMinorLocator,
    MultipleLocator,
)

from common.common import (
    read_numbers,
    numbers_to_averages,
    create_histogram,
    normalize_histogram,
)


def histogram_to_probabilities_of_deviation(hist, a=5.5, precision=0.01):
    res = np.zeros(hist.shape)
    total = sum(hist.T[0])  # equal to the number of iterations
    h = hist.shape[0]
    a = int(a / precision)
    up = h - a > a
    upper_bound = h - a if up else a

    for j, col in enumerate(hist.T):
        s = total - col[a]
        res[a][j] = 1.0
        for i in range(1, upper_bound):
            p = s / total
            u = a + i < h
            d = a - i >= 0
            s -= (col[a + i] if u else 0) + (col[a - i] if d else 0)
            if u:
                res[a + i][j] = p
            if d:
                res[a - i][j] = p

    return res


def setup_axis(
        ax, xlim, ylim,
        xlabel="days",
        ylabel="average",
        labels_font_size=24,
        ticks_font_size=20,
        x_major_ticks=100,
        x_minor_ticks=4,
        y_major_ticks=5,
        y_minor_ticks=5,
        major_line_width=0.75,
        minor_line_width=0.25,
        text_color="white",
):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

    ax.set_xlabel(xlabel, fontsize=labels_font_size, color=text_color)
    ax.set_ylabel(ylabel, fontsize=labels_font_size, color=text_color)

    ax.tick_params(axis="both", labelsize=ticks_font_size, labelcolor=text_color)

    ax.xaxis.set_major_locator(MultipleLocator(x_major_ticks))
    ax.xaxis.set_minor_locator(AutoMinorLocator(x_minor_ticks))
    ax.yaxis.set_major_locator(MultipleLocator(y_major_ticks))
    ax.yaxis.set_minor_locator(AutoMinorLocator(y_minor_ticks))

    ax.grid(which="major", axis="both", linestyle="-", linewidth=major_line_width)
    ax.grid(which="minor", axis="both", linestyle="-", linewidth=minor_line_width)


def setup_colorbar(fig, im, ax, text_color="white", label="probability, log10"):
    cb = fig.colorbar(im, ax=ax)
    cb.set_label(label)
    cb.ax.yaxis.set_tick_params(color=text_color)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=text_color)


def plot_data(ax, data, days, precision, cmap):
    return ax.pcolormesh(
        np.arange(0, days, 1),
        np.arange(0, 12 + precision, precision),
        data,
        vmin=data.min(), vmax=data.max(),
        cmap=cmap,
        #     norm=norm,
        shading="auto",
    )


def plot_all(
        data,
        averages,
        days,
        label,
        figsize=None,
        text_color="white",
        cmap="inferno",
        line_color="white",
        line_width=3,
        precision=0.01,
):
    figsize = (len(averages) / 10, 20) if figsize is None else figsize

    fig, ax = plt.subplots(figsize=figsize)
    fig.suptitle(label)
    setup_axis(ax, xlim, ylim)

    cmap = plt.get_cmap(cmap)
    im = plot_data(
        data=data,
        ax=ax,
        days=days, precision=precision, cmap=cmap,
    )
    ax.plot(
        np.arange(0, len(averages), 1),
        averages,
        color=line_color,
        linewidth=line_width,
    )
    setup_colorbar(
        fig=fig,
        im=im,
        ax=ax,
        text_color=text_color,
    )

    return fig


if __name__ == "__main__":
    NUMBERS = read_numbers("data.txt")
    AVERAGES = numbers_to_averages(NUMBERS)

    DAYS = len(NUMBERS) + 25
    ITERS = 250_000
    BALLS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    NBINS = 15
    PRECISION = 0.001
    SAVE_FILES = True

    hist = create_histogram(days=DAYS, iterations=ITERS, balls=BALLS, precision=PRECISION)
    normalized_hist = normalize_histogram(hist)
    log_hist = np.ma.log10(normalized_hist)
    log_hist = log_hist.filled(log_hist.min())

    probs = histogram_to_probabilities_of_deviation(hist, precision=PRECISION)
    log_probs = np.ma.log10(probs)
    log_probs = log_probs.filled(log_probs.min())

    xlim = [0, DAYS]
    ylim = [4.0, 7.0]
    text_color = "white"

    # cmap = "gist_stern_r"
    line_color = "orchid"

    colormaps = ['Accent', 'Accent_r',
                 'Blues', 'Blues_r', 'BrBG', 'BrBG_r', 'BuGn', 'BuGn_r', 'BuPu', 'BuPu_r',
                 'CMRmap', 'CMRmap_r',
                 'Dark2', 'Dark2_r',
                 'GnBu', 'GnBu_r', 'Greens', 'Greens_r', 'Greys', 'Greys_r',
                 'OrRd', 'OrRd_r', 'Oranges', 'Oranges_r',
                 'PRGn', 'PRGn_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r',
                 'PiYG', 'PiYG_r', 'PuBu', 'PuBuGn', 'PuBuGn_r', 'PuBu_r', 'PuOr', 'PuOr_r', 'PuRd',
                 'PuRd_r', 'Purples', 'Purples_r',
                 'RdBu', 'RdBu_r', 'RdGy', 'RdGy_r', 'RdPu', 'RdPu_r', 'RdYlBu', 'RdYlBu_r',
                 'RdYlGn', 'RdYlGn_r', 'Reds', 'Reds_r',
                 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r',
                 'Spectral', 'Spectral_r',
                 'Wistia', 'Wistia_r',
                 'YlGn', 'YlGnBu', 'YlGnBu_r', 'YlGn_r', 'YlOrBr', 'YlOrBr_r', 'YlOrRd', 'YlOrRd_r',
                 'afmhot', 'afmhot_r', 'autumn', 'autumn_r',
                 'binary', 'binary_r', 'bone', 'bone_r', 'brg', 'brg_r', 'bwr', 'bwr_r',
                 'cividis', 'cividis_r', 'cool', 'cool_r', 'coolwarm', 'coolwarm_r', 'copper', 'copper_r', 'cubehelix',
                 'cubehelix_r',
                 'flag', 'flag_r',
                 'gist_earth', 'gist_earth_r', 'gist_gray', 'gist_gray_r', 'gist_heat', 'gist_heat_r', 'gist_ncar',
                 'gist_ncar_r', 'gist_rainbow', 'gist_rainbow_r', 'gist_stern', 'gist_stern_r', 'gist_yarg', 'gist_yarg_r',
                 'gnuplot', 'gnuplot2', 'gnuplot2_r', 'gnuplot_r', 'gray', 'gray_r',
                 'hot', 'hot_r', 'hsv', 'hsv_r',
                 'inferno', 'inferno_r',
                 'jet', 'jet_r',
                 'magma', 'magma_r',
                 'nipy_spectral', 'nipy_spectral_r',
                 'ocean', 'ocean_r',
                 'pink', 'pink_r', 'plasma', 'plasma_r', 'prism', 'prism_r',
                 'rainbow', 'rainbow_r',
                 'seismic', 'seismic_r', 'spring', 'spring_r', 'summer', 'summer_r',
                 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r',
                 'terrain', 'terrain_r', 'turbo', 'turbo_r', 'twilight', 'twilight_r', 'twilight_shifted',
                 'twilight_shifted_r',
                 'viridis', 'viridis_r',
                 'winter', 'winter_r']

    image_path = "./images"
    Path(image_path).mkdir(exist_ok=True)

    for cmap in colormaps:
        cmap1 = cmap
        line_color1 = line_color
        fig1 = plot_all(
            data=log_hist,
            averages=AVERAGES,
            days=DAYS,
            label="probability of average, log10",
            precision=PRECISION,
            cmap=cmap1,
            line_color=line_color1
        )

        cmap2 = cmap
        line_color2 = line_color
        fig2 = plot_all(
            data=log_probs,
            averages=AVERAGES,
            days=DAYS,
            label="probability of deviation, log10",
            precision=PRECISION,
            cmap=cmap2,
            line_color=line_color2
        )
        fig1.savefig(f"{image_path}/hit_d{DAYS}_i{ITERS}_p{PRECISION}_{cmap1}-{line_color1}.png", facecolor="black")
        fig2.savefig(f"{image_path}/dev_d{DAYS}_i{ITERS}_p{PRECISION}_{cmap2}-{line_color2}.png", facecolor="black")
        plt.close('all')
