"""Fig. 2: IEEE 33-bus feeder with V2G hubs and PV locations; Fig. 3: daily profiles."""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from figstyle import setup, save, SINGLE, DOUBLE, panel, INK, INK2
from v2g.env import DEFAULT
from v2g.profiles import (residential_load, mixed_load, clear_sky_pv,
                          residential_availability, workplace_availability, HOURS)
from v2g.feeder import load_case33bw

setup()
os.makedirs("figures", exist_ok=True)

# ---- feeder topology (1-based bus numbers in labels)
pos = {}
for b in range(1, 19):
    pos[b] = (b - 1, 0.0)
for k, b in enumerate(range(19, 23)):
    pos[b] = (1 + k, -1.2)
for k, b in enumerate(range(23, 26)):
    pos[b] = (2 + k, 1.2)
for k, b in enumerate(range(26, 34)):
    pos[b] = (5 + k, 2.4)
d = load_case33bw()
fig, axes = plt.subplots(2, 1, figsize=(DOUBLE, 2.9))
for ai, ax in enumerate(axes):
    for f, t in zip(d["fb"] + 1, d["tb"] + 1):
        (x0, y0), (x1, y1) = pos[f], pos[t]
        if y0 != y1:
            ax.plot([x0, x0, x1], [y0, y1, y1], color=INK2, lw=0.8, zorder=1)
        else:
            ax.plot([x0, x1], [y0, y1], color=INK2, lw=0.8, zorder=1)
    ev = [b + 1 for b in DEFAULT["ev_buses"][ai]]
    pv = [b + 1 for b in DEFAULT["pv_buses"][ai]]
    for b, (x, y) in pos.items():
        ax.scatter(x, y, s=16, color="white", edgecolor=INK, lw=0.6, zorder=2)
        dx = 0.28 if b in (23, 26) else 0.0
        ax.text(x + dx, y - 0.42, str(b), ha="center", va="top", fontsize=5.5, color=INK)
        if b in ev:
            ax.scatter(x - 0.18, y + 0.42, s=22, marker="s", color="#2a78d6", edgecolor=INK, lw=0.4, zorder=3)
        if b in pv:
            ax.scatter(x + 0.18, y + 0.42, s=26, marker="^", color="#eda100", edgecolor=INK, lw=0.4, zorder=3)
    ax.scatter(-0.6, 0, s=40, marker="s", color=INK, zorder=3)
    ax.text(-0.6, 0.45, "Substation", ha="center", fontsize=6)
    ax.set_xlim(-1.4, 17.6); ax.set_ylim(-2.1, 3.2)
    ax.axis("off")
    ax.text(-1.4, 3.0, f"({'ab'[ai]}) Area {ai + 1} feeder: "
            f"{'residential' if ai == 0 else 'mixed commercial-residential'} load, "
            f"PV {DEFAULT['pv_mw'][ai]:.1f} MW", fontsize=7, va="top")
h1 = axes[0].scatter([], [], s=22, marker="s", color="#2a78d6", edgecolor=INK, lw=0.4, label="V2G hub")
h2 = axes[0].scatter([], [], s=26, marker="^", color="#eda100", edgecolor=INK, lw=0.4, label="PV unit")
axes[0].legend(handles=[h1, h2], loc="lower right", ncol=2, fontsize=6.5)
fig.tight_layout(h_pad=0.2)
save(fig, "fig_feeder")

# ---- daily profiles
fig, axes = plt.subplots(2, 1, figsize=(SINGLE, 3.3), sharex=True)
ax = axes[0]
ax.plot(HOURS, residential_load(), color="#2a78d6", ls="-", marker="o", label="Load, Area 1 (residential)")
ax.plot(HOURS, mixed_load(), color="#eb6834", ls="--", marker="s", label="Load, Area 2 (mixed)")
ax.plot(HOURS, clear_sky_pv(), color="#eda100", ls="-.", marker="^", label="PV, clear sky")
ax.set_ylabel("Normalized value (p.u.)"); ax.set_ylim(0, 1.45); ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.legend(loc="upper left", fontsize=6.2, ncol=2, columnspacing=0.8, handlelength=2.4)
panel(ax, "a"); ax.texts[-1].set_position((0.93, 0.98))
ax = axes[1]
ax.plot(HOURS, residential_availability(), color="#2a78d6", ls="-", marker="o", label="Area 1 (home charging)")
ax.plot(HOURS, workplace_availability(), color="#eb6834", ls="--", marker="s", label="Area 2 (workplace charging)")
ax.set_ylabel("Plugged-in fraction"); ax.set_ylim(0, 1.25); ax.set_yticks(np.arange(0, 1.01, 0.25))
ax.set_xlabel("Hour of day (h)"); ax.set_xticks(range(0, 25, 3)); ax.set_xlim(0, 24)
ax.legend(loc="upper left", fontsize=6.2, ncol=2, columnspacing=0.8)
panel(ax, "b"); ax.texts[-1].set_position((0.93, 0.98))
fig.tight_layout(h_pad=0.3)
save(fig, "fig_profiles")
print("ok")
