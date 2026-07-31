from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Styling
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
COLOURS = {
    "A": "#2196F3",   # blue
    "B": "#FF9800",   # orange
    "C": "#4CAF50",   # green (for later)
}
