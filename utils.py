import colorsys
import numpy as np
import ast


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

def prepare_multidupehack_input(df, color_clusters, output_file='mining.txt'):
    all_colors = set(color_clusters['HEX'].str.upper())
    
    with open(output_file, 'w') as f:
        for _, row in df.iterrows():
            dimensions = ":".join([
                str(row['decade']).strip(),
                str(row['century']).strip(),
                str(row['school']).replace(' ', '_').strip(),
                str(row['artist_full_name']).replace(' ', '_').strip()
            ])
            
            # Color dimension #HEX#COUNT
            colors = ast.literal_eval(row['cluster_hex'])
            counts = ast.literal_eval(row['palette_count'])  
            color_pairs = []
            for color, count in zip(colors, counts):
                hex_code = color.upper()
                if hex_code in all_colors: color_pairs.append(f"{hex_code}#{count}")
                    
            f.write(f"{dimensions}:{','.join(color_pairs)}:1.0\n")

#################################################################################################
# CLASSIFIER
#################################################################################################

def artwork_to_vector(row, n_colors, color_to_index):
    vec = np.zeros(n_colors)
    for color, weight in zip(row["cluster_hex"], row["palette_count"]):
        if color in color_to_index: vec[color_to_index[color]] += weight
    return vec