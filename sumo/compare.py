"""
Step 5 — compare the two runs.

Parses both tripinfo files and prints average waiting time, time loss, and trip duration
per vehicle, plus the % improvement of adaptive over the fixed-time baseline.
  waitingTime = seconds the vehicle was stopped (speed ~0)
  timeLoss    = seconds lost vs. driving the route at the ideal/free-flow speed
"""
import os
import xml.etree.ElementTree as ET

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
    if n == 0:
        raise ValueError(f"No <tripinfo> rows in {path} — the sim may have failed.")
    return {"vehicles": n, "avg_waitingTime": wt / n, "avg_timeLoss": tl / n, "avg_duration": dur / n}


def pct(base, adap):
    return 100.0 * (base - adap) / base if base else float("nan")


if __name__ == "__main__":
    base = score(os.path.join(HERE, "tripinfo_static.xml"))
    adap = score(os.path.join(HERE, "tripinfo_adaptive.xml"))

    print(f"{'metric':<18}{'fixed-time':>12}{'adaptive':>12}{'improve %':>12}")
    print("-" * 54)
    print(f"{'vehicles':<18}{base['vehicles']:>12}{adap['vehicles']:>12}{'':>12}")
    for k in ("avg_waitingTime", "avg_timeLoss", "avg_duration"):
        print(f"{k:<18}{base[k]:>12.2f}{adap[k]:>12.2f}{pct(base[k], adap[k]):>12.1f}")

    if abs(base["vehicles"] - adap["vehicles"]) > 0.02 * base["vehicles"]:
        print("\nWARNING: vehicle counts differ >2% — demand wasn't identical; comparison is invalid.")
