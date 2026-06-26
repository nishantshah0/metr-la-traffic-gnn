"""
Step 4 — ADAPTIVE run: a TraCI "longest-queue-first" signal controller.

This is the closed loop. Every second we look at each intersection, count how many cars
are queued (halting) on each approach, and switch the green toward the busiest approach
instead of blindly running a fixed cycle. Guard rails:
  MIN_GREEN — never switch a green before this (stops flickering)
  MAX_GREEN — force a switch after this (stops one approach starving the others)

Why this beats fixed-time: a static program holds a green for a fixed time even when no
one is there, while cross-traffic waits. This controller gives green to whoever actually
has the queue, and moves on as soon as an approach is clear.

(Conceptually this is where a forecast plugs in: instead of reacting to the *current*
queue, you'd feed a short-horizon prediction of *incoming* traffic — the role the A3T-GCN
forecaster plays in the parent project. Here we use the live queue as the signal.)
"""
import os
import sys
import sumolib

SUMO_HOME = os.environ.get("SUMO_HOME") or os.path.dirname(os.path.dirname(sumolib.checkBinary("sumo")))
os.environ["SUMO_HOME"] = SUMO_HOME
sys.path.append(os.path.join(SUMO_HOME, "tools"))
import traci  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))

MIN_GREEN = 8     # s — minimum green before a switch is allowed
MAX_GREEN = 50    # s — maximum green before a switch is forced (fairness)


def green_phase_lanes(tlid):
    """For one traffic light, map each GREEN phase index -> the set of incoming lanes it serves.

    getControlledLinks is indexed by signal-state character position; each entry is a list of
    (incomingLane, outgoingLane, viaLane) tuples. We skip yellow/all-red phases.
    """
    logic = traci.trafficlight.getAllProgramLogics(tlid)[0]
    links = traci.trafficlight.getControlledLinks(tlid)
    out = {}
    for pi, ph in enumerate(logic.phases):
        state = ph.state
        if "y" in state.lower():            # skip transition (yellow) phases
            continue
        lanes = {links[i][0][0]
                 for i, c in enumerate(state)
                 if c in "Gg" and i < len(links) and links[i]}
        if lanes:
            out[pi] = lanes
    return out


def run():
    sumoBinary = sumolib.checkBinary("sumo")
    traci.start([
        sumoBinary,
        "-n", os.path.join(HERE, "grid.net.xml"),
        "-r", os.path.join(HERE, "grid.rou.xml"),
        "--seed", "42",
        "--begin", "0", "--end", "3600",
        "--time-to-teleport", "-1",                    # same honesty setting as the baseline
        "--tripinfo-output", os.path.join(HERE, "tripinfo_adaptive.xml"),
        "--tripinfo-output.write-unfinished", "true",
        "--no-step-log", "true",
    ])

    tlids = list(traci.trafficlight.getIDList())
    greens = {t: green_phase_lanes(t) for t in tlids}     # {tl: {green_phase_idx: {lanes}}}
    nphases = {t: len(traci.trafficlight.getAllProgramLogics(t)[0].phases) for t in tlids}
    last_switch = {t: 0.0 for t in tlids}                 # when the current green began

    while traci.simulation.getMinExpectedNumber() > 0:
        traci.simulationStep()
        now = traci.simulation.getTime()

        for t in tlids:
            cur = traci.trafficlight.getPhase(t)
            phase_lanes = greens[t]

            if cur not in phase_lanes:        # mid yellow/all-red — let SUMO finish it
                continue

            held = now - last_switch[t]
            if held < MIN_GREEN:              # respect minimum green
                continue

            # queue length per candidate green = halting vehicles on its incoming lanes
            queue = {pi: sum(traci.lane.getLastStepHaltingNumber(l) for l in lanes)
                     for pi, lanes in phase_lanes.items()}
            busiest = max(queue, key=queue.get)

            switch = (
                (busiest != cur and queue[busiest] > queue[cur])  # someone else is busier
                or queue[cur] == 0                                # current approach is empty
                or held >= MAX_GREEN                              # fairness cap
            )
            if switch:
                # Advance to the yellow right after this green; SUMO inserts the legal
                # yellow/all-red clearance, then proceeds to the next green.
                traci.trafficlight.setPhase(t, (cur + 1) % nphases[t])
                last_switch[t] = now

    traci.close()


if __name__ == "__main__":
    run()
    print("done -> tripinfo_adaptive.xml")
