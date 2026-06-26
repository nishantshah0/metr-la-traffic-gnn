"""
Step 2 — build the traffic demand.

Uses SUMO's randomTrips.py to generate ONE seeded route file (grid.rou.xml). The SAME
file is fed to BOTH controllers, so they face identical traffic — the comparison is fair.
`--seed 42` makes it reproducible; supplying `-r` auto-runs duarouter to validate routes.
"""
import os
import sys
import subprocess
import sumolib

SUMO_HOME = os.environ.get("SUMO_HOME") or os.path.dirname(os.path.dirname(sumolib.checkBinary("sumo")))
os.environ["SUMO_HOME"] = SUMO_HOME

HERE = os.path.dirname(os.path.abspath(__file__))
NET = os.path.join(HERE, "grid.net.xml")
TRIPS = os.path.join(HERE, "grid.trips.xml")
ROU = os.path.join(HERE, "grid.rou.xml")

randomTrips = os.path.join(SUMO_HOME, "tools", "randomTrips.py")
cmd = [
    sys.executable, randomTrips,
    "-n", NET,
    "-o", TRIPS,
    "-r", ROU,                # -r => also produce validated routes (runs duarouter)
    "-b", "0", "-e", "3600",  # one simulated hour of departures
    "-p", "1.5",              # a new trip every 1.5 s (~2400 trips); smaller = heavier traffic
    "--fringe-factor", "5",   # bias trips to enter/leave at the grid edges (real through-traffic)
    "--seed", "42",           # reproducible demand
]
print("randomTrips:", " ".join(cmd))
subprocess.run(cmd, check=True)
print("wrote", ROU)
