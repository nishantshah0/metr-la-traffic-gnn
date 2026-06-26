"""
Step 6 (optional) — bar chart of fixed-time vs adaptive delay -> result.png.
"""
import os
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))


def score(path):
    n = wt = tl = dur = 0
    for _, el in ET.iterparse(path, events=("end",)):
        if el.tag == "tripinfo":
            n += 1
            wt += float(el.get("waitingTime"))
            tl += float(el.get("timeLoss"))
            dur += float(el.get("duration"))
            el.clear()
    return {"avg_waitingTime": wt / n, "avg_timeLoss": tl / n, "avg_duration": dur / n}


base = score(os.path.join(HERE, "tripinfo_static.xml"))
adap = score(os.path.join(HERE, "tripinfo_adaptive.xml"))

keys = [("avg_waitingTime", "Waiting time"), ("avg_timeLoss", "Time loss"), ("avg_duration", "Trip duration")]
labels = [k[1] for k in keys]
b = [base[k[0]] for k in keys]
a = [adap[k[0]] for k in keys]

x = np.arange(len(labels))
w = 0.38
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w / 2, b, w, label="fixed-time", color="#c44e52")
ax.bar(x + w / 2, a, w, label="adaptive (TraCI)", color="#55a868")
ax.set_ylabel("seconds per vehicle (lower is better)")
ax.set_title("SUMO 3×3 grid — adaptive signal control vs fixed-time")
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis="y", alpha=0.3)
for i in range(len(labels)):
    drop = 100.0 * (b[i] - a[i]) / b[i]
    ax.text(x[i] + w / 2, a[i], f"-{drop:.0f}%", ha="center", va="bottom", fontsize=10, color="#2a7d2a")
plt.tight_layout()
plt.savefig(os.path.join(HERE, "result.png"), dpi=150, bbox_inches="tight")
print("wrote", os.path.join(HERE, "result.png"))
