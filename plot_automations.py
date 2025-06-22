import numpy as np
import pandas as pd

from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import to_rgb

import seaborn as sns
from itertools import combinations
from collections import defaultdict
import re

#################################################################################################
# COLOR CLUSTERING
#################################################################################################

"""
Plot hues as circle
"""
def plot_hues(H_sin, H_cos, weights=None, save_path=None):
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

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


"""
Plot saturation distribution
"""
def plot_saturation(S, weights=None, save_path=None):
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
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

"""
Plot brightness distribution
"""
def plot_brightness(B, weights=None, save_path=None):
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

    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

"""
Plot every color cluster
"""
def plot_clusters_color_grid(df, save_path=None):
    # Sort by hue (H column)
    df = df.sort_values('H')

    # Create a grid layout - adjust these numbers based on your preference
    rows = 40
    cols = 25
    total_colors = rows * cols

    # Make sure we don't exceed the number of colors we have
    df = df.iloc[:total_colors]

    # Create figure and axes
    fig, ax = plt.subplots(figsize=(20, 30))
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('equal')
    ax.axis('off')  # Turn off axes

    # Plot each color as a square in the grid
    for idx, (_, row) in enumerate(df.iterrows()):
        x = idx % cols
        y = rows - 1 - (idx // cols)  # Start from top
        
        # Create a rectangle with the color
        rect = patches.Rectangle((x, y), 1, 1, linewidth=0, 
                                facecolor=to_rgb(row['HEX']))
        ax.add_patch(rect)

    plt.title('Clusters color Grid ordered by hue', fontsize=16, pad=20)
    
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

#################################################################################################
# PATTERN VISUALIZATION
#################################################################################################

# ======================== BASIC VISUALIZATIONS ========================

"""
Plot distribution of pattern support counts
"""
def plot_support_analysis(df, save_path=None):
    plt.figure(figsize=(12, 6))
    
    xticks = np.arange(0, df['support'].max()+1, 100)
    plt.xticks(xticks, rotation=45)
    ax = sns.histplot(df['support'], bins=xticks, color='skyblue', linewidth=0.5)
    yticks = np.arange(0, int(ax.get_ylim()[1]) + 1, 20)
    plt.yticks(yticks)
    
    plt.gca().xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    
    plt.title("Distribution of Pattern Support Counts")
    plt.xlabel("Pattern Support Count (number of matching artworks)")
    plt.ylabel("Number of Patterns")
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

"""
Plot distribution of pattern sizes across dimensions
"""
def plot_pattern_size_distribution(df, dim, save_path=None):
    plt.figure(figsize=(12,4))
    ax = sns.histplot(df[f'n_{dim}'], bins=100, color='skyblue', linewidth=0.5)
    
    max_x = df[f'n_{dim}'].max()
    if max_x > 100: xticks = np.arange(0, max_x+1, 5)
    else: xticks = np.arange(0, max_x+1, 1)
    plt.xticks(xticks, rotation=45)

    max_y = int(ax.get_ylim()[1])
    if max_y > 300: yticks = np.arange(0, max_y+1, 50)
    else: yticks = np.arange(0, max_y, 10)
    plt.yticks(yticks)
    
    plt.title(f'Number of {dim} per pattern')
    plt.xlabel(f'Number of {dim} included')
    plt.ylabel('Number of patterns')

    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()

"""
Plot pattern frequency by start decade
"""
def plot_temporal_patterns(df, save_path=None):
    # Calculate first decade of each pattern
    df['first_decade'] = df['decades'].apply(lambda x: min(int(d) for d in x))
    decade_counts = df.groupby('first_decade').size()
    
    plt.figure(figsize=(10,5))
    ax = decade_counts.plot(kind='bar', color='skyblue')
    
    yticks = np.arange(0, int(ax.get_ylim()[1]) + 1, 100)
    plt.yticks(yticks)
    
    plt.title("Pattern Frequency by Starting Decade")
    plt.xlabel("Decade")
    plt.ylabel("Number of Patterns")
    
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()

"""
Plot pattern frequency by school included
"""

def plot_school_patterns(df, save_path=None):
    """
    Plot pattern frequency by school presence
    """
    # Explode the schools column to count each school separately
    exploded_schools = df['schools'].explode()
    
    # Count occurrences of each school
    school_counts = exploded_schools.value_counts()
    
    # Create plot
    plt.figure(figsize=(12, 6))
    ax = school_counts.plot(kind='bar', color='skyblue')
    
    # Formatting
    plt.title("Pattern Frequency by School")
    plt.xlabel("School")
    plt.ylabel("Number of Patterns")
    
    yticks = np.arange(0, int(ax.get_ylim()[1]) + 1, 100)
    plt.yticks(yticks)

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    # Add grid lines
    plt.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.show()


"""
Plot color coocurrence as heatmap
"""
def plot_color_cooccurrence_matrix(df, top_n_colors=50, save_path=None):
    # Convert string lists to actual lists if needed
    df['colors'] = df['colors'].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
    
    # Get list of all color lists
    all_color_lists = df['colors'].tolist()
    
    # Get top N most frequent colors
    all_colors = [color for sublist in all_color_lists for color in sublist]
    top_colors = pd.Series(all_colors).value_counts().head(top_n_colors).index
    
    # Initialize co-occurrence matrix
    co_matrix = pd.DataFrame(0, index=top_colors, columns=top_colors)
    
    # Fill co-occurrence matrix
    for colors in all_color_lists:
        present_colors = [c for c in colors if c in top_colors]
        for i in range(len(present_colors)):
            for j in range(i, len(present_colors)):
                c1, c2 = present_colors[i], present_colors[j]
                co_matrix.loc[c1, c2] += 1
                if i != j:  # Avoid double-counting diagonal
                    co_matrix.loc[c2, c1] += 1
    
    # Normalize by diagonal
    norm_matrix = co_matrix.div(np.diag(co_matrix), axis=0)
    
    # Plot with adjustments for label readability
    plt.figure(figsize=(20, 18))  # Increased figure size
    ax = sns.heatmap(
        norm_matrix,
        cmap="Blues",
        annot=False,
        linewidths=0.5,
        square=True
    )
    
    # Rotate and adjust color labels
    ax.set_xticks(np.arange(len(top_colors)) + 0.5)
    ax.set_yticks(np.arange(len(top_colors)) + 0.5)
    ax.set_xticklabels(top_colors, rotation=90, ha='right', fontsize=8)
    ax.set_yticklabels(top_colors, rotation=0, fontsize=8)
    
    plt.title(f"Color Co-occurrence Matrix (Top {top_n_colors} Colors)", pad=20)
    plt.xlabel("Color")
    plt.ylabel("Color")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Color co-occurrence matrix saved to {save_path}")
    plt.show()

"""
Plot pattern color palette as grid
"""
def plot_color_grid(df, dim, dim_range, save_path=None):
   
    data = df[df[f'{dim}_range'] == dim_range]
    palette_size = len(data['colors'].explode().unique())
    
    # Get colors
    df['color_tuple'] = df['colors'].apply(
        lambda x: tuple(x[:palette_size]) if isinstance(x, list) else tuple(eval(x)[:palette_size])
    )
    color_series = data['color_tuple'].explode()
    top_colors = color_series.value_counts().index.tolist()
    
    # Calculate grid layout
    total_colors = len(top_colors)
    cols = 10
    rows = int(np.ceil(total_colors / cols))
    
    # Create plot
    fig, axs = plt.subplots(rows, cols, figsize=(cols*1.5, rows*1.5))
    fig.suptitle(f"Color Palette: {dim_range} in {total_colors} colors", y=1.02)
    
    # Plot each color in grid
    for i, color in enumerate(top_colors):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        
        # Display color
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=color))
        
        # Add hex label (rotated 90 degrees)
        ax.text(0.5, 0.5, color.upper(), 
               rotation=90,
               ha='center', va='center',
               color='white' if np.mean(mcolors.to_rgb(color)) < 0.5 else 'black')
        
        # Formatting
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect('equal')
    
    # Hide empty subplots
    for i in range(len(top_colors), rows*cols):
        row = i // cols
        col = i % cols
        ax = axs[row, col] if rows > 1 else axs[col]
        ax.axis('off')
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()