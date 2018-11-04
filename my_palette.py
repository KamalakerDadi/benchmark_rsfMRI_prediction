import seaborn
from matplotlib import colors
import numpy as np

seaborn.set_palette('dark')

orig_palette = seaborn.color_palette()


def shift_value(rgb, shift):
    hsv = colors.rgb_to_hsv(rgb)
    hsv[-1] += shift
    return colors.hsv_to_rgb(hsv)


def color_palette(n_colors):
    orig_palette = seaborn.color_palette(n_colors=n_colors)
    shifts = np.linspace(-.3, .3, n_colors)
    alternate_shifts = shifts.copy()
    alternate_shifts[::2] = shifts[:len(shifts[::2])]
    alternate_shifts[1::2] = shifts[len(shifts[::2]):]
    palette = [shift_value(col, shift)
               for col, shift in zip(orig_palette, alternate_shifts)]
    return palette


atlases = ['ica', 'kmeans', 'dictlearn', 'ward', 'ho', 'aal', 'basc']

datasets = ['COBRE', 'ADNI', 'ADNIDOD', 'ACPI', 'ABIDE', 'HCP']

atlas_palette = dict(zip(atlases, color_palette(len(atlases) + 3)))

datasets_palette = dict(zip(datasets, color_palette(len(datasets) + 5)))

