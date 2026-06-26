"""
Step 3 — BASELINE run: SUMO's built-in fixed-time signal program (no control code).

Plain headless SUMO. Writes per-vehicle stats to tripinfo_static.xml. We keep unfinished
vehicles in the output so a controller that strands cars can't look good by hiding them.
"""
import os
import subprocess
import sumolib

SUMO_HOME = os.environ.get("SUMO_HOME") or os.path.dirname(os.path.dirname(sumolib.checkBinary("sumo")))
os.environ["SUMO_HOME"] = SUMO_HOME

HERE = os.path.dirname(os.path.abspath(__file__))
sumoBinary = sumolib.checkBinary("sumo")   # headless console binary (not sumo-gui)
cmd = [
    sumoBinary,
    "-n", os.path.join(HERE, "grid.net.xml"),
    "-r", os.path.join(HERE, "grid.rou.xml"),
    "--seed", "42",
    "--begin", "0", "--end", "3600",
    "--time-to-teleport", "-1",                      # disable teleporting so jams are counted honestly
    "--tripinfo-output", os.path.join(HERE, "tripinfo_static.xml"),
    "--tripinfo-output.write-unfinished", "true",
    "--no-step-log", "true",
]
print("baseline (fixed-time):", " ".join(cmd))
subprocess.run(cmd, check=True)
print("done -> tripinfo_static.xml")
