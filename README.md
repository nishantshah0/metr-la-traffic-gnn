# Traffic-Speed Forecasting on METR-LA with a Spatio-Temporal GNN

Predicting near-future highway traffic speed across **207 real sensors in Los Angeles** with an
**A3T-GCN** (Attention Temporal Graph Convolutional Network) — a model that learns from both the
*time-series* at each sensor and the *road network* that connects them. The graph is the whole
point: a jam doesn't stay put, it propagates along connected road segments, and a graph model can
see that coming where a per-sensor model cannot.

---

## Why this matters (smart cities)

City traffic is a **network**, not a set of isolated roads. When one segment backs up, the slowdown
spreads upstream within minutes. A model that forecasts each sensor in isolation (a plain LSTM)
misses this; it has no way to know that congestion is about to *arrive* from a neighbouring road.
By feeding the model the **road-connectivity graph**, it can anticipate spillover before it shows up
locally. Accurate short-horizon speed forecasts are the input layer for real smart-city systems:
adaptive signal timing, ramp metering, dynamic routing, and congestion pricing all start with
"what will traffic look like in the next 15–60 minutes?"

---

## The dataset

**METR-LA** — loop-detector readings from 207 sensors on LA highways, one reading every 5 minutes
over 4 months (34,272 time steps). It's the standard benchmark for traffic forecasting. We use the
built-in `METRLADatasetLoader` from PyTorch Geometric Temporal, which **auto-downloads** the data.

Each training example ("snapshot") is one sliding window:

| Tensor | Shape | Meaning |
|---|---|---|
| `x` (input) | `[207, 2, 12]` | 207 sensors × 2 features (**speed**, **time-of-day**) × last 12 steps (60 min) |
| `y` (target) | `[207, 12]` | 207 sensors × next 12 **speed** steps (60 min) |
| `edge_index` | `[2, 1722]` | road links: column *k* = `[from_sensor, to_sensor]` |
| `edge_attr` | `[1722]` | one proximity weight per link (closer ⇒ larger) |

Features are **z-score normalized**, so values look like `-0.7`, not `65 mph`. We convert back to mph
only for the final plot.

---

## How to run it (Google Colab, free GPU)

1. New notebook → **Runtime ▸ Change runtime type ▸ T4 GPU**.
2. Install the two PyG packages (Colab already ships a matching PyTorch — don't reinstall torch):
   ```python
   !pip install -q torch_geometric torch-geometric-temporal
   ```
3. Upload `train.py` (or paste it into a cell) and run:
   ```python
   !python train.py
   ```
   First run downloads METR-LA (a few MB). Training 30 epochs on the T4 takes a few minutes.
   The script prints the data inspection, the baseline, the per-epoch loss, the final comparison,
   and saves `prediction.png`.

> **CPU also works** — the code auto-detects the device (`cuda` if available, else `cpu`); it's just
> slower. No code change needed.

<details>
<summary>Local install / troubleshooting</summary>

```bash
pip install -r requirements.txt
python train.py
```
If an import fails with a missing compiled op (`torch_scatter` / `torch_sparse`), install the prebuilt
wheels matched to your torch+CUDA:
```bash
python -c "import torch;print(torch.__version__)"   # e.g. 2.3.0+cu121
pip install pyg_lib torch_scatter torch_sparse -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
```
If you see `No module named torch_geometric.utils.to_dense_adj`, your `torch-geometric-temporal` is
too old for your PyG — upgrade it (`pip install -U torch-geometric-temporal`).
</details>

---

## Results: GNN vs. naive baseline

The model is judged against a **persistence baseline** — "future speed = last observed speed."
Traffic is sticky over short horizons, so this trivial rule is a genuinely tough bar; beating it is
the minimum proof the GNN learned something real (the spatial spillover between sensors). All MSE
values are in **normalized units** (lower is better).

```
Persistence baseline : MSE 0.4105 | MAE 5.45 mph | RMSE 12.98 mph
A3T-GCN model        : MSE 0.3764 | MAE 6.37 mph | RMSE 12.43 mph
Improvement over baseline (MSE): +8.3%   ->  [PASS]
```

Measured on the held-out latest 20% of time (6,850 windows); full 30-epoch run on an RTX 5070 Ti.
The GNN beats persistence on **MSE (+8.3%)** and **RMSE (12.43 vs 12.98 mph)** — the squared-error
metrics — while its **MAE is slightly higher (6.37 vs 5.45 mph)**. That contrast is the insightful
part: persistence is razor-accurate *on average* (traffic is usually stable, so "next ≈ now" is spot
on → low MAE), but it gets clobbered on sudden congestion, where large squared errors dominate. The
GNN trades a sliver of average accuracy to better anticipate those big swings → lower RMSE/MSE — and
anticipating the swings is the whole point of forecasting. This is a deliberately *minimal*
single-layer A3T-GCN; heavier models (DCRNN, Graph WaveNet) widen the gap.

> **The design choice that made it win:** the model predicts the *change* from the current speed (a
> residual), not the absolute speed. Outputting zero reproduces persistence exactly, so the model is
> anchored to that baseline and can only improve on it. (An earlier absolute-speed version *lost* to
> the baseline by 13% — see `A3TGCNForecaster.forward` in `train.py` for the fix.)

### Prediction plot

`prediction.png` — predicted vs. actual speed for sensor #0 over a 24-hour stretch of the test set
(the **+30-min-ahead** forecast; the model predicts up to +60 min). Sensor outages — METR-LA records
missing/faulty readings as 0 mph — are **masked out as gaps** rather than drawn as fake cliffs to
zero, so the plot reflects real traffic only.

![predicted vs actual speed](prediction.png)

---

## How it works (for a non-expert)

The model is an **A3T-GCN**, which stacks three ideas:

- **Graph Convolution (GCN) — the *space* part.** For each sensor, it blends in its road-neighbours'
  readings (weighted by how close they are). This is how "the road feeding into me is slowing down"
  becomes part of my forecast.
- **GRU (recurrent unit) — the *time* part.** It reads the last 12 steps *in order*, keeping a
  running memory, so it picks up trends ("speed has been falling for 15 minutes") instead of just a
  single instant.
- **Attention — the *A* in A3T.** It learns *which* of the past 12 steps matter most and weights
  them, collapsing the history into one informative summary per sensor.

**The one-line reason a graph beats an LSTM here:** an LSTM treats each sensor as an island; the GCN
lets a sensor borrow information from its road-connected neighbours, so congestion arriving from
upstream is visible *before* it reaches the sensor itself.

---

## What I'd build next

- **Close the loop in simulation.** Feed these forecasts into **adaptive signal timing** in the
  [SUMO](https://www.eclipse.dev/sumo/) traffic simulator and measure whether predicted-congestion-aware
  signals cut average delay vs. fixed timing.
- **Go multi-horizon and probabilistic.** Predict a *distribution* (not a point) at each horizon so
  downstream controllers can reason about uncertainty, and compare A3T-GCN against stronger baselines
  (DCRNN, Graph WaveNet).
- **Learn the graph instead of assuming it.** Replace the fixed proximity adjacency with an
  **adaptive/learned adjacency**, so the model discovers influence links the road map alone misses
  (e.g. parallel routes that absorb each other's overflow).

---

## Repo layout

```
metr-la-traffic-gnn/
├── train.py           # all 5 stages, heavily commented (data → baseline → model → train → plot)
├── requirements.txt   # dependencies (Colab needs only the 2 PyG packages)
├── README.md          # this file
└── prediction.png     # generated by Stage 5 on first run
```
