import colorsys
import numpy as np
import pandas as pd


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
# ITEMSET MINING
#################################################################################################

def generate_multidupehack_file(file_name, colors_df, df, target):
    all_colors = sorted(colors_df["HEX"].unique())
    all_targets = sorted(df[target].unique())
    columns = list(all_colors) + list(map(str, all_targets))
    result_df = pd.DataFrame(columns=columns)
    
    for idx, row in df.iterrows():
        for color in row["cluster_hex"]:
            if color in result_df.columns: result_df.at[idx, color] = 1
        target_col = str(row[target])
        if target_col in result_df.columns: result_df.at[idx, target_col] = 1

    with open(file_name, "w") as f:
        for _, row in result_df.iterrows():
            color_part = ",".join(str(row[col]) for col in all_colors)
            targets_part = ",".join(str(row[col]) for col in all_targets)
            f.write(f"{color_part} {targets_part} 1\n")


#################################################################################################
# CLASSIFIER
#################################################################################################

def artwork_to_vector(row, n_colors, color_to_index):
    vec = np.zeros(n_colors)
    for color, weight in zip(row["cluster_hex"], row["palette_count"]):
        if color in color_to_index: vec[color_to_index[color]] += weight
    return vec