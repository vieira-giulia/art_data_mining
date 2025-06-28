import colorsys
import numpy as np
import pandas as pd
import ast
import re
from itertools import combinations
from collections import defaultdict
from functools import lru_cache


#################################################################################################
# COLOR MANIPULATION AND EXTRACTION
#################################################################################################

def hex_to_hsb(hex_color):
    hex_color = hex_color.lstrip("#")
    r, g, b = [int(hex_color[i:i+2], 16) for i in (0, 2, 4)]
    return colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)

def hsb_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h / 360, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

def hue_to_sin(H):
   return np.sin(H * 2 * np.pi)

def hue_to_cos(H):
    return np.cos(H * 2 * np.pi)

def sin_cos_to_hue(H_sin, H_cos):
    hues_rad = np.arctan2(H_sin, H_cos)
    hues_rad = np.mod(hues_rad, 2 * np.pi)
    return np.rad2deg(hues_rad)

def extract_hsb_components_from_vec(colors, top_n=10):
    colors = colors[:top_n] + ["#000000"] * (top_n - len(colors))  # pad with black if needed
    h, s, b = [], [], []
    for color in colors:
        hue, sat, bri = hex_to_hsb(color)
        h.append(hue)
        s.append(sat)
        b.append(bri)
    return h, s, b

#################################################################################################
# CLASSIFIER
#################################################################################################

def artwork_to_vector(row, n_colors, color_to_index):
    vec = np.zeros(n_colors)
    for color, weight in zip(row["cluster_hex"], row["palette_count"]):
        if color in color_to_index: vec[color_to_index[color]] += weight
    return vec

def fuzzy_artwork_to_vector(row, n_colors, color_to_index):
    vector = np.zeros(n_colors)
    if isinstance(row["palette_count"], dict):
        total_pixels = sum(row["palette_count"].values())
        for color, count in row["palette_count"].items():
            if color in color_to_index: vector[color_to_index[color]] = count / total_pixels
    return vector