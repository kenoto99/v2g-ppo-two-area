"""Elsevier-style figure settings: Times New Roman (metric-compatible
substitute used when the Microsoft font is not installed), 8 pt text,
single column 90 mm, double column 190 mm, vector PDF + 600 dpi PNG."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

FAMILY = ["Times New Roman", "TeX Gyre Termes", "Liberation Serif", "Nimbus Roman", "DejaVu Serif"]
for f in ["/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-regular.otf",
          "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-bold.otf",
          "/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyretermes-italic.otf"]:
    try:
        font_manager.fontManager.addfont(f)
    except Exception:
        pass

MM = 1 / 25.4
SINGLE = 90 * MM
DOUBLE = 190 * MM

# controller colours (validated categorical slots) + line style as secondary encoding
STYLE = {
    "UC":  dict(color="#eb6834", ls="--", marker="s", label="UC"),
    "RB":  dict(color="#eda100", ls="-.", marker="^", label="RB"),
    "DAO": dict(color="#1baf7a", ls=":",  marker="D", label="DAO"),
    "PPO": dict(color="#2a78d6", ls="-",  marker="o", label="PPO (proposed)"),
}
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#d9d8d4"


def setup():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": FAMILY, "mathtext.fontset": "stix",
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "lines.linewidth": 1.1, "lines.markersize": 3.2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.4,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "legend.frameon": True, "legend.framealpha": 1.0, "legend.edgecolor": "#bdbcb7",
        "legend.fancybox": False, "savefig.dpi": 600, "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02, "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save(fig, name, outdir="figures"):
    fig.savefig(f"{outdir}/{name}.pdf")
    fig.savefig(f"{outdir}/{name}.png")
    plt.close(fig)


def panel(ax, letter):
    ax.text(0.01, 0.98, f"({letter})", transform=ax.transAxes, va="top", ha="left",
            fontsize=8, bbox=dict(fc="white", ec="none", pad=0.6))
