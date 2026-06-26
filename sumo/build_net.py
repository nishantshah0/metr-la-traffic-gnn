"""
Step 1 — build the road network.

Generates a 3x3 grid of signalized intersections with STATIC (fixed-time) traffic
lights. That static program is our fair baseline: the adaptive controller later runs
on this exact same network, so any difference in delay is due to the controller alone.
"""
import os
import subprocess
import sumolib

# eclipse-sumo (pip) bundles the binaries; SUMO_HOME is the folder above /bin.
SUMO_HOME = os.environ.get("SUMO_HOME") or os.path.dirname(os.path.dirname(sumolib.checkBinary("sumo")))
os.environ["SUMO_HOME"] = SUMO_HOME

HERE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(HERE, "grid.net.xml")

netgenerate = sumolib.checkBinary("netgenerate")
cmd = [
    netgenerate, "--grid",
    "--grid.number", "3",            # 3x3 junctions
    "--grid.length", "200",          # 200 m blocks
    "--grid.attach-length", "100",   # 100 m entry/exit stubs so traffic can enter at the fringe
    "--default.lanenumber", "2",     # 2 lanes per direction
    "--no-turnarounds", "true",
    "--tls.guess", "true",           # put traffic lights on the qualifying junctions
    "--tls.default-type", "static",  # fixed-time program = the fair baseline
    "--tls.cycle.time", "90",        # 90 s cycle
    "-o", NET,
]
print("netgenerate:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("wrote", NET)
