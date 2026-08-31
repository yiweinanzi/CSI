# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear
"""
Contains plotting setting for matplotlib and seaborn.
"""
import matplotlib
import seaborn as sns

matplotlib.use("Agg")
# Sizes
FONTSIZE = 11  # pt
PAGEWIDTH = 10  # inches

# Margins for half-width plots
LEFT_MARGIN = 0.11
RIGHT_MARGIN = 0.01
TOP_MARGIN = 0.02
BOTTOM_MARGIN = 0.10

# Margins for third-width plots
LEFT_MARGIN_SMALL = LEFT_MARGIN * 1.5
RIGHT_MARGIN_SMALL = RIGHT_MARGIN * 1.5
TOP_MARGIN_SMALL = TOP_MARGIN * 1.5
BOTTOM_MARGIN_SMALL = BOTTOM_MARGIN * 1.5

# Label positioning, to be used with ax.*axis.set_label_coords(), for half-width plots
X_LABEL_POS = -0.08
Y_LABEL_POS = -0.08

# Same for third-width plots
X_LABEL_POS_SMALL = -0.12
Y_LABEL_POS_SMALL = -0.12

# rcParams
MATPLOTLIB_PARAMS = {
    # Font sizes
    "font.size": FONTSIZE,  # controls default text sizes
    "axes.titlesize": FONTSIZE,  # fontsize of the axes title
    "axes.labelsize": FONTSIZE,  # fontsize of the x and y labels
    "xtick.labelsize": FONTSIZE,  # fontsize of the tick labels
    "ytick.labelsize": FONTSIZE,  # fontsize of the tick labels
    "legend.fontsize": FONTSIZE,  # legend fontsize
    "figure.titlesize": FONTSIZE,  # fontsize of the figure title
    # Figure size and DPI
    "figure.dpi": 100,
    "savefig.dpi": 300,
    "figure.figsize": (PAGEWIDTH / 2, PAGEWIDTH / 2),
    # Margins
    "figure.subplot.left": LEFT_MARGIN,
    "figure.subplot.right": 1.0 - RIGHT_MARGIN,
    "figure.subplot.top": 1.0 - TOP_MARGIN,
    "figure.subplot.bottom": BOTTOM_MARGIN,
    # colors
    "lines.markeredgewidth": 0.8,
    "axes.edgecolor": "black",
    "axes.grid": True,
    "grid.color": "0.9",
    "axes.grid.which": "major",
    # x-axis ticks and grid
    "xtick.bottom": True,
    "xtick.direction": "out",
    "xtick.color": "black",
    "xtick.major.bottom": True,
    "xtick.major.size": 4,
    "xtick.minor.bottom": True,
    "xtick.minor.size": 2,
    # y-axis ticks and grid
    "ytick.left": True,
    "ytick.direction": "out",
    "ytick.color": "black",
    "ytick.major.left": True,
    "ytick.major.size": 4,
    "ytick.minor.left": True,
    "ytick.minor.size": 2,
    # legend
    "legend.frameon": False,
    "legend.handlelength": 3.5,
}

# Method styles
LABELS = {
    "gatr": "Wi-GATr (ours)",
    "gatr-pretrained": "Wi-GATr (ours, pretrained)",
    "transformer": "Transformer",
    "transformer-pretrained": "Transformer (pretrained)",
    "cantransformer": "Trf. + canonicalization",
    "augtransformer": "Trf. + augmentation",
    "stupidtransformer": "Trf. w/o tokenizer",
    "segnn": "SEGNN",
    "vit": "PLViT",
    "learned materials": "Learned Materials",
    "neural materials": "Neural Materials",
}
# https://coolors.co/419108-5ea5ba-aeadd2-e9aa63
COLORS = {
    "gatr": "#419108",
    "gatr-pretrained": "#419108",
    "transformer": "#AEADD2",
    "transformer-pretrained": "#AEADD2",
    "cantransformer": "#AEADD2",
    "augtransformer": "#AEADD2",
    "stupidtransformer": "#3C3993",
    "segnn": "#E9AA63",
    "vit": "#5ea5ba",
    "learned materials": "#D4B37F",
    "neural materials": "#D4B37F",
}
LINESTYLES = {
    "gatr": "-",
    "gatr-pretrained": ":",
    "transformer": "--",
    "transformer-pretrained": ":",
    "cantransformer": "-.",
    "augtransformer": ":",
    "stupidtransformer": ":",
    "segnn": "-.",
    "vit": "-.",
    "learned materials": ":",
    "neural materials": "-.",
}
MARKERS = {
    "gatr": "o",
    "gatr-pretrained": "H",
    "transformer": "s",
    "transformer-pretrained": "D",
    "cantransformer": "<",
    "augtransformer": ">",
    "stupidtransformer": "D",
    "segnn": "<",
    "vit": ">",
    "learned materials": "D",
    "neural materials": ">",
}
MARKER_SIZES = {
    "gatr": 6.0,
    "gatr-pretrained": 6.0,
    "transformer": 6.0,
    "transformer-pretrained": 6.0,
    "cantransformer": 5.5,
    "augtransformer": 5.5,
    "stupidtransformer": 5.5,
    "segnn": 5.5,
    "vit": 5.5,
    "learned materials": 5.5,
    "neural materials": 5.5,
}
ERROR_ALPHA = 0.1

# Color map for RSRP (or everything)
COLORMAP = matplotlib.cm.inferno  # pylint: disable=no-member


# Initialization function for global state
def init_plotting():
    """Initializes global matplotlib state to our plotting defaults"""
    sns.set_style("whitegrid")
    matplotlib.rcParams.update(MATPLOTLIB_PARAMS)
