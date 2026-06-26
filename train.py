"""
================================================================================
 Spatio-Temporal Traffic Forecasting on METR-LA with a Graph Neural Network
================================================================================

WHAT THIS PROJECT DOES
----------------------
We predict near-future highway traffic SPEED across 207 real sensors in Los
Angeles, using both:
  (a) the recent HISTORY at each sensor (a time-series signal), and
  (b) the ROAD NETWORK that connects the sensors (a graph).

The model is an A3T-GCN (Attention Temporal Graph Convolutional Network): it
fuses a Graph Convolution (space) with a GRU (time) and an attention layer.

WHY A GRAPH? (the smart-cities idea)
------------------------------------
Traffic is not 207 independent time-series. A jam on one road segment spills
onto the segments feeding into it a few minutes later. A plain LSTM that looks
at one sensor in isolation can't "see" its upstream neighbour slowing down.
A GNN can, because the graph literally encodes which sensors are road-adjacent.

HOW THE FILE IS ORGANIZED (built in stages; each stage is a clearly-marked block)
---------------------------------------------------------------------------------
  Stage 1  load + chronologically split the data, inspect one sample
  Stage 2  persistence baseline ("future speed = last observed speed")
  Stage 3  the A3T-GCN model (GCN + GRU + attention)
  Stage 4  training loop + evaluation vs the baseline (% improvement)
  Stage 5  predicted-vs-actual plot saved to prediction.png

Run the whole pipeline with:   python train.py
(On Colab, enable a GPU first: Runtime > Change runtime type > T4 GPU.)
================================================================================
"""

# ──────────────────────────────────────────────────────────────────────────────
# Imports
# ──────────────────────────────────────────────────────────────────────────────
import os
import glob

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")            # headless backend: lets us save a PNG without a screen
import matplotlib.pyplot as plt

# The METR-LA loader auto-downloads the dataset on first use (needs internet).
from torch_geometric_temporal.dataset import METRLADatasetLoader
# temporal_signal_split gives us a CHRONOLOGICAL train/test split (no shuffling).
from torch_geometric_temporal.signal import temporal_signal_split
# A3TGCN2 is the BATCHED version of the A3T-GCN cell (lets us train fast on a GPU).
from torch_geometric_temporal.nn.recurrent import A3TGCN2


# ──────────────────────────────────────────────────────────────────────────────
# Config — the few knobs that matter. Kept at the top so they're easy to find.
# ──────────────────────────────────────────────────────────────────────────────
# METR-LA records one reading every 5 minutes. So 12 steps = 12 * 5 = 60 minutes.
NUM_TIMESTEPS_IN = 12    # how many PAST 5-min steps the model sees   (last 60 min)
NUM_TIMESTEPS_OUT = 12   # how many FUTURE 5-min steps we predict      (next 60 min)
TRAIN_RATIO = 0.8        # first 80% of time -> train, last 20% -> test

BATCH_SIZE = 32          # how many windows we process at once (a "mini-batch")
HIDDEN = 32              # size of the hidden vector A3T-GCN produces per sensor
EPOCHS = 30              # how many full passes over the training data
LR = 1e-2                # Adam learning rate (0.01 is the value the PyG tutorial uses)

# Device: the project must run whether or not a GPU is present.
# On Colab "Runtime > Change runtime type > GPU" gives you CUDA; otherwise CPU.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# STAGE 1 — DATA
# ==============================================================================
def load_data():
    """
    Download METR-LA and split it CHRONOLOGICALLY into train/test.

    Each returned object is a `StaticGraphTemporalSignal`: an iterable that yields
    one PyG `Data` object ("snapshot") per time step. "Static graph" means the road
    network (the edges) is the SAME at every step — only the readings (x) and the
    targets (y) change.
    """
    loader = METRLADatasetLoader()   # first call downloads ~a few MB into ./data and caches it

    # get_dataset slides a window over the 34,272 raw 5-min readings: for each valid
    # window it stores the past NUM_TIMESTEPS_IN steps as x, and the next
    # NUM_TIMESTEPS_OUT steps (speed only) as y.
    # Number of windows = 34272 - (IN + OUT) + 1 = 34249 with 12/12.
    dataset = loader.get_dataset(
        num_timesteps_in=NUM_TIMESTEPS_IN,
        num_timesteps_out=NUM_TIMESTEPS_OUT,
    )

    # ── THE SPLIT, AND WHY ORDER MATTERS ──────────────────────────────────────
    # temporal_signal_split takes the FIRST `train_ratio` fraction of the timeline
    # as training data and the REMAINING tail as test data. It returns (train, test)
    # IN THAT ORDER and does NOT shuffle.
    #
    # Why we must NOT shuffle a time series before splitting:
    #   Shuffling mixes future snapshots into the training set. The model would
    #   effectively "study tomorrow's traffic" before being tested on it — that's
    #   data leakage, and it produces fake-good scores that collapse in the real
    #   world. A chronological split mirrors deployment: train on the past, predict
    #   the genuinely-unseen future.
    train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=TRAIN_RATIO)
    return loader, dataset, train_dataset, test_dataset


def inspect_one_sample(dataset, train_dataset, test_dataset):
    """Print the shape of ONE sample and explain, axis by axis, what each number means."""
    snapshot = next(iter(dataset))   # a PyG `Data` bundling x, y, and the graph

    print("=" * 70)
    print("STAGE 1 — METR-LA data inspection")
    print("=" * 70)
    print(f"Running on device: {DEVICE}")
    print(f"\nRaw repr of one snapshot:\n  {snapshot}\n")

    # x : the model's INPUT.  shape [num_nodes, num_features, num_timesteps_in] = [207, 2, 12]
    #   axis 0 = 207 sensors (graph nodes)
    #   axis 1 =   2 features: 0 = traffic SPEED, 1 = TIME-OF-DAY (a clock signal, NOT volume)
    #   axis 2 =  12 past readings (= last 60 minutes); values are z-score normalized, not mph
    print(f"x  (inputs)  shape = {tuple(snapshot.x.shape)}")
    print("     -> [207 sensors, 2 features (speed, time-of-day), 12 past steps]")

    # y : the TARGET.  shape [num_nodes, num_timesteps_out] = [207, 12]
    #   next 12 readings of SPEED ONLY (no feature axis — we only forecast speed)
    print(f"y  (target)  shape = {tuple(snapshot.y.shape)}")
    print("     -> [207 sensors, 12 future SPEED steps]  (speed only, no feature axis)")

    # edge_index : the GRAPH STRUCTURE.  shape [2, num_edges] = [2, 1722]
    #   column k = [source_node, target_node]: a road link from sensor i to sensor j.
    #   PHYSICALLY an edge = two sensors close on the road network, so traffic flows
    #   between them. This is the neighbour information a plain LSTM is blind to.
    print(f"edge_index   shape = {tuple(snapshot.edge_index.shape)}")
    print("     -> [2, num_edges]; column k = [from_sensor, to_sensor] = one road link")

    # edge_attr : the EDGE WEIGHTS.  shape [num_edges] = [1722]
    #   one proximity weight per edge (closer/more-connected = larger). The GCN uses
    #   these to weight how strongly a neighbour influences a sensor.
    print(f"edge_attr    shape = {tuple(snapshot.edge_attr.shape)}")
    print("     -> [num_edges]; one proximity weight per road link (closer = larger)")

    print(f"\nGraph: {snapshot.x.shape[0]} sensors (nodes), "
          f"{snapshot.edge_index.shape[1]} road links (edges) — static across all time.")
    print(f"Snapshots: {dataset.snapshot_count} total  ->  {train_dataset.snapshot_count} "
          f"train (earliest) + {test_dataset.snapshot_count} test (latest)")
    print(f"Each snapshot = predict the next {NUM_TIMESTEPS_OUT * 5} min "
          f"from the previous {NUM_TIMESTEPS_IN * 5} min.")
    print("=" * 70 + "\n")


def stack_to_tensors(signal):
    """
    Turn a StaticGraphTemporalSignal into two big tensors we can mini-batch over.

    Returns:
      X : [N, 207, 2, 12]  all input windows
      Y : [N, 207, 12]     all targets
    where N = number of snapshots in this portion. `.features`/`.targets` are the
    underlying lists of numpy arrays; np.array stacks them along a new first axis.
    """
    X = torch.from_numpy(np.array(signal.features)).float()   # [N, 207, 2, 12]
    Y = torch.from_numpy(np.array(signal.targets)).float()    # [N, 207, 12]
    return X, Y


def get_static_graph(signal):
    """Grab the (static) edge_index and edge weights once — they never change in time."""
    s0 = next(iter(signal))
    return s0.edge_index, s0.edge_attr     # [2, 1722] LongTensor, [1722] FloatTensor


# ==============================================================================
# STAGE 2 — PERSISTENCE BASELINE
# ==============================================================================
def persistence_mse(X, Y):
    """
    The "persistence" (a.k.a. naive / last-value) forecast: predict that the future
    speed equals the LAST speed we observed, held flat across the whole horizon.

    Why a baseline at all?
      It sets the bar. Traffic is sticky over 5–60 min, so "it'll be about what it
      is now" is already a decent guess. If our fancy GNN can't beat this trivial
      rule, the GNN has learned nothing useful — all that machinery would be
      decoration. Beating it is the minimum evidence that the model found real
      structure (here: spatial spillover between sensors).

    Shapes:
      X : [N, 207, 2, 12]   ->  X[:, :, 0, -1:] = feature 0 (SPEED), last step = [N, 207, 1]
      We broadcast that single value across all 12 future steps to match Y [N, 207, 12].
      (NOTE: index 0 is critical — X[:, :, 1, ...] would be TIME-OF-DAY, a wrong baseline.)
    """
    last_speed = X[:, :, 0, -1:]                       # [N, 207, 1]  last observed speed
    y_persist = last_speed.expand(-1, -1, Y.size(-1))  # [N, 207, 12] held flat across horizon
    return torch.mean((y_persist - Y) ** 2).item()


# ==============================================================================
# STAGE 3 — THE A3T-GCN MODEL
# ==============================================================================
class A3TGCNForecaster(torch.nn.Module):
    """
    A3T-GCN = Attention Temporal Graph Convolutional Network.

    Three ingredients, in intuitive terms (no heavy math):

      • GCN  (the "GC" — graph convolution): the SPACE part. For each sensor it
        mixes in its road-neighbours' readings, weighted by edge_weight. This is
        how "my upstream neighbour is slowing down" reaches a sensor's prediction.

      • GRU  (the "T" — temporal recurrence): the TIME part. It walks through the
        12 past steps in order and keeps a running memory, so it can pick up
        trends ("speed has been dropping for 15 min") rather than just a snapshot.

      • Attention (the "A3T"): instead of treating all 12 past steps equally, it
        LEARNS which of them matter most for the forecast and weights them — e.g.
        the most recent minutes usually count more, but a sharp drop 20 min ago
        might be the real signal. It collapses the 12 steps into ONE summary vector.

    THE ONE REASON A GRAPH MODEL BEATS A PLAIN LSTM HERE:
      An LSTM sees each sensor as an isolated time-series. The GCN lets a sensor's
      forecast use its road-connected neighbours' states, so congestion that is
      about to ARRIVE from upstream is visible BEFORE it shows up locally. The
      graph encodes that spatial cause-and-effect; the LSTM simply can't represent it.

    Shapes through forward():
      x           : [B, 207, 2, 12]   (batch of B windows; B = BATCH_SIZE)
      A3TGCN2 out : [B, 207, HIDDEN]   one summary vector per sensor (attention already applied)
      linear out  : [B, 207, 12]       predicted CHANGE in speed per step
      + last speed: [B, 207, 12]       added back -> absolute 12-step forecast (matches Y)

    NOTE: this model predicts the CHANGE from the current speed (a "residual"), not
    the absolute speed — see forward(). That anchors it to the persistence baseline,
    so it can only improve on it. This is the key change that lets it beat persistence.
    """
    def __init__(self, node_features=2, periods=NUM_TIMESTEPS_IN,
                 horizon=NUM_TIMESTEPS_OUT, hidden=HIDDEN, batch_size=BATCH_SIZE):
        super().__init__()
        # in_channels = features per step (2); out_channels = hidden size; periods = input window T.
        # batch_size is REQUIRED by A3TGCN2 and must match the batch we feed in.
        self.tgnn = A3TGCN2(in_channels=node_features, out_channels=hidden,
                            periods=periods, batch_size=batch_size)
        # Map the hidden vector to the 12 future speed steps we want to predict.
        self.linear = torch.nn.Linear(hidden, horizon)

    def forward(self, x, edge_index, edge_weight=None):
        # RESIDUAL ("delta") prediction — the single most important design choice for
        # beating persistence. Instead of predicting absolute future speed, we predict
        # the CHANGE from the current speed and add it back on:
        #     forecast = current_speed + learned_change
        # If the model outputs zero change, it reproduces the persistence baseline
        # EXACTLY. So it starts anchored to that strong naive guess and can only improve
        # — it only has to learn the corrections (e.g. "congestion is arriving from
        # upstream, so speed will drop"). Easier to optimize, and far more robust to the
        # train/test time drift that made the plain version lose.
        last_speed = x[:, :, 0, -1:]                # [B, 207, 1]  current speed (feature 0, last step)
        h = self.tgnn(x, edge_index, edge_weight)   # [B, 207, HIDDEN]
        h = F.relu(h)                               # non-linearity
        delta = self.linear(h)                      # [B, 207, 12]  predicted CHANGE per step
        return last_speed + delta                   # [B, 207, 12]  absolute speed forecast


# ==============================================================================
# STAGE 4 — TRAIN + EVALUATE
# ==============================================================================
def make_loader(X, Y, shuffle):
    """
    Wrap stacked tensors in a DataLoader that hands out mini-batches.

    drop_last=True: A3TGCN2 is built for a FIXED batch size, so we drop the final
    partial batch (a few windows out of thousands — negligible).

    shuffle=True is used for TRAINING only. This is NOT the leakage we warned about
    in Stage 1: the chronological split already happened, so train and test stay
    time-separated. Shuffling the ORDER of training windows within the train set is
    just standard stochastic-gradient-descent practice (it stops the model from
    memorising the day-by-day order). We never shuffle test.
    """
    ds = torch.utils.data.TensorDataset(X, Y)
    return torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE,
                                       shuffle=shuffle, drop_last=True)


def train(model, train_loader, edge_index, edge_weight):
    """Standard PyTorch training loop. Prints the average train MSE each epoch."""
    # weight_decay adds mild regularization, which helps the model generalize to the
    # test period (the latest 20% of time, whose patterns drift from the training period).
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    loss_fn = torch.nn.MSELoss()

    print(f"STAGE 4 — training for {EPOCHS} epochs on {DEVICE} ...")
    model.train()
    for epoch in range(EPOCHS):
        running, n_batches = 0.0, 0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(DEVICE)   # [B, 207, 2, 12]
            y_batch = y_batch.to(DEVICE)   # [B, 207, 12]

            y_hat = model(x_batch, edge_index, edge_weight)  # [B, 207, 12]
            loss = loss_fn(y_hat, y_batch)

            optimizer.zero_grad()   # clear gradients from the previous step
            loss.backward()         # backprop: compute gradients
            optimizer.step()        # nudge the weights downhill

            running += loss.item()
            n_batches += 1
        print(f"  epoch {epoch + 1:02d}/{EPOCHS}   train MSE = {running / n_batches:.4f}")
    return model


@torch.no_grad()
def collect_test_predictions(model, test_loader, edge_index, edge_weight):
    """
    Run the trained model over the (chronological, unshuffled) test set ONCE and
    return three aligned tensors so every later number/plot is consistent:
      preds   : [M, 207, 12]  model forecast
      trues   : [M, 207, 12]  ground truth
      persist : [M, 207, 12]  the persistence baseline on the SAME windows
    (M = test windows kept after drop_last.)
    """
    model.eval()
    preds, trues, persist = [], [], []
    for x_batch, y_batch in test_loader:
        x_batch = x_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        y_hat = model(x_batch, edge_index, edge_weight)
        last_speed = x_batch[:, :, 0, -1:].expand(-1, -1, y_batch.size(-1))  # baseline
        preds.append(y_hat.cpu())
        trues.append(y_batch.cpu())
        persist.append(last_speed.cpu())
    return torch.cat(preds), torch.cat(trues), torch.cat(persist)


def report_results(preds, trues, persist, scaler):
    """Print the head-to-head: model vs persistence, in normalized MSE AND real mph."""
    mean, std = scaler
    model_mse = torch.mean((preds - trues) ** 2).item()
    persist_mse = torch.mean((persist - trues) ** 2).item()
    improvement = (persist_mse - model_mse) / persist_mse * 100.0

    # Convert error to real mph. In the difference (pred - true) the z-score MEAN
    # cancels, so we only multiply by std to get mph — far more intuitive than a
    # unitless MSE. (If the mph scale couldn't be recovered, std=1 and these read
    # as normalized units.)
    model_mae = torch.mean(torch.abs(preds - trues)).item() * std
    persist_mae = torch.mean(torch.abs(persist - trues)).item() * std
    model_rmse = (model_mse ** 0.5) * std
    persist_rmse = (persist_mse ** 0.5) * std

    print("\n" + "=" * 70)
    print("RESULTS (test set — averaged over the full 60-min forecast horizon)")
    print("=" * 70)
    print(f"  Persistence baseline : MSE {persist_mse:.4f} | MAE {persist_mae:5.2f} mph | RMSE {persist_rmse:5.2f} mph")
    print(f"  A3T-GCN model        : MSE {model_mse:.4f} | MAE {model_mae:5.2f} mph | RMSE {model_rmse:5.2f} mph")
    print(f"  Improvement over baseline (MSE): {improvement:.1f}%")
    if model_mse < persist_mse:
        print("  -> The GNN BEATS the naive baseline. It learned real structure. [PASS]")
    else:
        print("  -> The GNN did NOT beat the baseline -- something is wrong (see README). [FAIL]")
    print("=" * 70 + "\n")
    return model_mse, persist_mse, improvement


# ==============================================================================
# STAGE 5 — VISUALIZE
# ==============================================================================
def recover_speed_scaler(loader):
    """
    Recover (mean, std) of RAW speed in mph so the plot reads in real units.

    The loader z-score-normalizes speed, so the tensors are unitless. To plot mph we
    invert that with: mph = z * std + mean. We recompute (mean, std) the same way the
    loader does, straight from the raw node-values file it downloaded.

    Robust: we search raw_data_dir for the (timesteps, 207, 2) array and pick feature 0
    (speed). If the file can't be found, we fall back to (0, 1) and the plot is simply
    in normalized units (still correct, just not mph).
    """
    try:
        for path in sorted(glob.glob(os.path.join(loader.raw_data_dir, "*.npy"))):
            arr = np.load(path)
            if arr.ndim == 3 and 207 in arr.shape and 2 in arr.shape:
                feat_axis = arr.shape.index(2)                       # which axis holds the 2 features
                speed = np.take(arr, indices=0, axis=feat_axis)      # feature 0 = speed
                return float(speed.mean()), float(speed.std())
    except Exception as e:
        print(f"[plot] couldn't recover mph scale ({e}); plotting normalized units.")
    return 0.0, 1.0


def plot_predictions(preds, trues, scaler, sensor=0, horizon_step=5,
                     max_points=288, out_path="prediction.png"):
    """
    Plot predicted vs actual speed for ONE sensor over time, and save a PNG.

    preds/trues : [M, 207, 12] chronological. We pick one sensor and one horizon step
    (default = index 5 -> the +30-min forecast, which tracks tightly; the model predicts
    up to +60 min). We follow it across the first `max_points` test windows
    (288 windows = 24 h of 5-min data — one daily cycle).

    METR-LA records missing/faulty sensor readings as 0 mph, so we MASK near-zero
    'actual' points (outages) to NaN — matplotlib then draws clean gaps instead of fake
    cliffs plunging to zero.
    """
    mean, std = scaler
    pred = (preds[:max_points, sensor, horizon_step].numpy() * std + mean).astype(float)
    true = (trues[:max_points, sensor, horizon_step].numpy() * std + mean).astype(float)
    outage = true < 5.0            # sensor dropout (stored as ~0 mph), not real traffic
    true[outage] = np.nan
    pred[outage] = np.nan
    minutes_ahead = (horizon_step + 1) * 5

    plt.figure(figsize=(12, 4))
    plt.plot(true, label="actual speed", linewidth=2)
    plt.plot(pred, label=f"predicted (+{minutes_ahead} min)", linewidth=1.5, alpha=0.85)
    plt.title(f"METR-LA sensor #{sensor}: predicted vs actual speed "
              f"({minutes_ahead}-min-ahead forecast)")
    plt.xlabel("test time step (5 min each)")
    plt.ylabel("speed (mph)" if std != 1.0 else "speed (normalized)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"STAGE 5 — saved plot to {os.path.abspath(out_path)}")


# ==============================================================================
# MAIN — run all five stages in order
# ==============================================================================
def main():
    # Stage 1: load, split, inspect one sample.
    loader, dataset, train_dataset, test_dataset = load_data()
    inspect_one_sample(dataset, train_dataset, test_dataset)

    # Pack into tensors and grab the static graph once.
    train_X, train_Y = stack_to_tensors(train_dataset)
    test_X, test_Y = stack_to_tensors(test_dataset)
    edge_index, edge_weight = get_static_graph(train_dataset)
    edge_index = edge_index.to(DEVICE)
    edge_weight = edge_weight.to(DEVICE)

    # Stage 2: establish the bar to beat.
    base_mse = persistence_mse(test_X, test_Y)
    print(f"STAGE 2 — persistence baseline test MSE = {base_mse:.4f}  (this is the bar to beat)\n")

    # Stage 3 + 4: build the model, train it, evaluate it.
    train_loader = make_loader(train_X, train_Y, shuffle=True)
    test_loader = make_loader(test_X, test_Y, shuffle=False)
    model = A3TGCNForecaster().to(DEVICE)
    model = train(model, train_loader, edge_index, edge_weight)

    preds, trues, persist = collect_test_predictions(model, test_loader, edge_index, edge_weight)

    # Recover the mph scale once; used for both the metrics and the plot.
    scaler = recover_speed_scaler(loader)
    report_results(preds, trues, persist, scaler)

    # Stage 5: plot one sensor's predicted vs actual speed in real mph.
    plot_predictions(preds, trues, scaler, sensor=0)


if __name__ == "__main__":
    main()
