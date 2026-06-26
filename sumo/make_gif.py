"""
Render a clean top-down GIF of the adaptive controller running -> demo.gif.

Fully headless and reliable: runs the sim, and each frame draws the road network (grey
lines) plus every vehicle as a dot (red = stopped at a light, green = moving) with
matplotlib. No SUMO GUI needed. Run build_net.py + build_routes.py first.
"""
import os
import sys
import glob
import sumolib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMO_HOME = os.environ.get("SUMO_HOME") or os.path.dirname(os.path.dirname(sumolib.checkBinary("sumo")))
os.environ["SUMO_HOME"] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402
from run_adaptive import green_phase_lanes, MIN_GREEN, MAX_GREEN  # reuse the controller

HERE = os.path.dirname(os.path.abspath(__file__))

# network geometry for the backdrop
net = sumolib.net.readNet(os.path.join(HERE, "grid.net.xml"))
edge_shapes = [e.getShape() for e in net.getEdges()]
xmin, ymin, xmax, ymax = net.getBoundary()

FRAMES = os.path.join(HERE, "_frames")
os.makedirs(FRAMES, exist_ok=True)
for f in glob.glob(os.path.join(FRAMES, "*.png")):
    os.remove(f)

WARMUP = 150        # let traffic build before filming
EVERY = 4           # capture every 4th step
MAX_FRAMES = 45

traci.start([
    sumolib.checkBinary("sumo"),
    "-n", os.path.join(HERE, "grid.net.xml"),
    "-r", os.path.join(HERE, "grid.rou.xml"),
    "--seed", "42", "--begin", "0", "--end", "3600",
    "--no-step-log", "true",
])

tlids = list(traci.trafficlight.getIDList())
greens = {t: green_phase_lanes(t) for t in tlids}
nphases = {t: len(traci.trafficlight.getAllProgramLogics(t)[0].phases) for t in tlids}
last_switch = {t: 0.0 for t in tlids}

captured = 0
step = 0
while traci.simulation.getMinExpectedNumber() > 0 and captured < MAX_FRAMES:
    traci.simulationStep()
    now = traci.simulation.getTime()

    for t in tlids:                       # adaptive control (same policy as run_adaptive.py)
        cur = traci.trafficlight.getPhase(t)
        pl = greens[t]
        if cur not in pl or now - last_switch[t] < MIN_GREEN:
            continue
        q = {pi: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes) for pi, lanes in pl.items()}
        busiest = max(q, key=q.get)
        if (busiest != cur and q[busiest] > q[cur]) or q[cur] == 0 or (now - last_switch[t]) >= MAX_GREEN:
            traci.trafficlight.setPhase(t, (cur + 1) % nphases[t])
            last_switch[t] = now

    if step >= WARMUP and step % EVERY == 0:
        xs, ys, cs = [], [], []
        for vid in traci.vehicle.getIDList():
            x, y = traci.vehicle.getPosition(vid)
            xs.append(x); ys.append(y)
            cs.append("#d62728" if traci.vehicle.getSpeed(vid) < 0.1 else "#2ca02c")
        fig, ax = plt.subplots(figsize=(6, 6))
        for shp in edge_shapes:
            ax.plot([p[0] for p in shp], [p[1] for p in shp],
                    color="#cfcfcf", lw=3, solid_capstyle="round", zorder=1)
        ax.scatter(xs, ys, c=cs, s=16, zorder=2)
        ax.set_xlim(xmin - 20, xmax + 20)
        ax.set_ylim(ymin - 20, ymax + 20)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(f"Adaptive signals  —  t = {int(now)} s,  {len(xs)} vehicles", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(FRAMES, f"f{captured:03d}.png"), dpi=80, facecolor="white")
        plt.close(fig)
        captured += 1
    step += 1

traci.close()

from PIL import Image  # noqa: E402
files = sorted(glob.glob(os.path.join(FRAMES, "f*.png")))
imgs = [Image.open(f).convert("RGB") for f in files]
if imgs:
    out = os.path.join(HERE, "demo.gif")
    imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=150, loop=0, optimize=True)
    print(f"wrote {out}  ({len(imgs)} frames, {os.path.getsize(out)//1024} KB)")
else:
    print("no frames captured")
