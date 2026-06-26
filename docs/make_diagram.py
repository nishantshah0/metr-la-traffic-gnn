"""
Generate the architecture diagram (architecture.png + .svg) explaining the A3T-GCN:
  Panel 1 - message passing (a sensor aggregates its road-neighbours)
  Panel 2 - the pipeline: Input -> GCN -> GRU -> Attention -> Linear(+residual) -> Forecast
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

HERE = os.path.dirname(os.path.abspath(__file__))

fig, ax = plt.subplots(figsize=(13, 8))
ax.set_xlim(0, 130)
ax.set_ylim(0, 82)
ax.axis("off")
fig.patch.set_facecolor("white")

ax.text(65, 79, "A3T-GCN — how the graph network forecasts traffic",
        ha="center", va="center", fontsize=17, weight="bold", color="#1a1a1a")

# ----------------------------- Panel 1: message passing -----------------------------
ax.text(65, 72.5, "1 — Message passing:  each sensor mixes in its road-neighbours",
        ha="center", fontsize=12.5, weight="bold", color="#2a4d69")

nodes = {
    "T": (34, 57, "#ff7f0e"),   # target ("you")
    "A": (18, 65, "#d62728"),   # upstream, slowing
    "B": (15, 50, "#8c8c8c"),
    "C": (35, 69, "#8c8c8c"),
    "D": (53, 60, "#8c8c8c"),
}
for k in ("A", "B", "C", "D"):                       # light edges
    x1, y1, _ = nodes[k]; x2, y2, _ = nodes["T"]
    ax.plot([x1, x2], [y1, y2], color="#d0d0d0", lw=1.5, zorder=1)
for k in ("A", "B", "C", "D"):                       # arrows neighbour -> target
    x1, y1, _ = nodes[k]; x2, y2, _ = nodes["T"]
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
                                 color="#2ca02c", lw=1.6, shrinkA=9, shrinkB=9, zorder=2))
for k, (x, y, c) in nodes.items():
    ax.add_patch(Circle((x, y), 2.7, facecolor=c, edgecolor="white", lw=1.6, zorder=3))
ax.text(34, 57, "you", ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=4)
ax.text(18, 60.0, "upstream 25 mph", ha="center", fontsize=8, color="#d62728", weight="bold")

ax.text(64, 59.5,
        "Your next hour depends on your neighbours.\n"
        "A jam upstream (red, 25 mph) reaches you a few\n"
        "minutes later — the GCN lets each sensor borrow its\n"
        "neighbours' readings and 'see' congestion coming.\n\n"
        "A plain LSTM treats each sensor as an island and can't.\n"
        "That is THE reason a graph model wins here.",
        ha="left", va="center", fontsize=10, color="#333333")

# ----------------------------- Panel 2: pipeline -----------------------------
ax.text(65, 39.5, "2 — The pipeline:   last hour  →  next hour",
        ha="center", fontsize=12.5, weight="bold", color="#2a4d69")

def draw_box(x, text, fc):
    w, h, y = 17, 17, 13
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.5",
                                linewidth=1.5, edgecolor="#2a2a2a", facecolor=fc, zorder=2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=8.5, color="white", weight="bold", zorder=3)
    return x, y + h / 2, x + w

boxes = [
    (2,   "Input  x\n[207 x 2 x 12]\nlast hour,\nall sensors", "#6c757d"),
    (23,  "GCN\nSPACE\nmix in road\nneighbours", "#1f77b4"),
    (44,  "GRU\nTIME\nread 12 steps\nin order", "#2ca02c"),
    (65,  "ATTENTION\nweight the\nsteps ->\n[207 x 32]", "#9467bd"),
    (86,  "LINEAR\n+ last speed\n(residual)", "#e8862e"),
    (107, "Forecast  y\n[207 x 12]\nnext 60 min", "#17a2b8"),
]
mids = [draw_box(x, t, fc) for x, t, fc in boxes]
for i in range(len(boxes) - 1):
    _, ymid, xr = mids[i]
    xl = mids[i + 1][0]
    ax.add_patch(FancyArrowPatch((xr + 0.4, ymid), (xl - 0.4, ymid), arrowstyle="-|>",
                                 mutation_scale=16, color="#555555", lw=2, zorder=1))

ax.text(65, 6.5,
        "Residual trick:  forecast = current speed + learned change  —  it starts at the naive baseline and can only improve.",
        ha="center", fontsize=9, style="italic", color="#444444")

fig.tight_layout()
fig.savefig(os.path.join(HERE, "architecture.png"), dpi=150, bbox_inches="tight", facecolor="white")
fig.savefig(os.path.join(HERE, "architecture.svg"), bbox_inches="tight", facecolor="white")
print("wrote architecture.png + architecture.svg")
