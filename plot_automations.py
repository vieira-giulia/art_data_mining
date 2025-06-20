import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.colors as mcolors
import seaborn as sns

def plot_hues(H_sin, H_cos, weights=None):
    hues_rad = np.arctan2(H_sin, H_cos)
    hues_rad = np.mod(hues_rad, 2 * np.pi)
    hues_rad_plot = hues_rad  # already in radians for polar plot

    angles = np.linspace(0, 2 * np.pi, 360)

    kde = gaussian_kde(hues_rad_plot, bw_method=0.1, weights=weights)
    density = kde(angles)
    density_norm = density / density.max()

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, polar=True)

    width = 2 * np.pi / len(angles)

    for angle, d in zip(angles, density_norm):
        hue_deg = np.rad2deg(angle) % 360
        color = mcolors.hsv_to_rgb([hue_deg / 360, 1, 1])
        ax.bar(angle, d, width=width, bottom=0, color=color, edgecolor=None, linewidth=0)

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_yticks([])
    ax.set_ylim(0, 1.1)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 10)))
    ax.set_xticklabels([str(x) for x in range(0, 360, 10)])

    plt.title('Hue degrees KDE Density')
    plt.show()


def plot_saturation(S, weights=None):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(x=S, bw_adjust=0.2, fill=True, color='skyblue', weights=weights)
    plt.title('Kernel Density Estimation of Saturation')
    plt.xlabel('Saturation')
    plt.ylabel('Density')
    plt.xticks([i/10 for i in range(11)])
    plt.grid(True)
    plt.axvline(x=0.15, color='gray', linestyle='--', label='Gray threshold')
    plt.axvline(x=0.4, color='steelblue', linestyle='--', label='Muted color threshold')
    plt.legend()
    plt.show()


def plot_brightness(B, weights=None):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(x=B, bw_adjust=0.2, fill=True, color='skyblue', weights=weights)
    plt.title('Kernel Density Estimation of Brightness')
    plt.xlabel('Brightness')
    plt.ylabel('Density')
    plt.xticks([i/10 for i in range(11)])
    plt.grid(True)
    plt.axvline(x=0.1, color='black', linestyle='--', label='Black threshold')
    plt.axvline(x=0.9, color='white', linestyle='--', label='White threshold')
    plt.legend()
    plt.show()
