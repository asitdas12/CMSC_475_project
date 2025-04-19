# benchmark_heat.py  — extended benchmarking & visualization for 2-D Heat-Equation
"""
This script now **does four things**

1. **Benchmark** dataset-generation time *versus* FNO inference time for a grid of
   stored-time-step counts `T` (5 → 200).
2. **Plot** the log-scaled timing curves **and** mark the first crossover point
   where FNO inference becomes more expensive than brute-force generation.
3. **Visualize** a *single* trajectory for **every** `T` in `T_VALUES`:
   it stores each field snapshot as a PNG **and** stitches them into a GIF so you
   can watch heat diffusion evolve.  Output lives in `animations/`.
4. **Save** an (untrained) FNO model *per T* to `models/` so you can reload and
   fine-tune it later if needed.

The extra features are enabled by default and are inexpensive because only one
trajectory per `T` is rendered.

Run:
    python benchmark_heat.py              # all default options
    python benchmark_heat.py --no-gpu     # force CPU timing
    python benchmark_heat.py --skip-anim  # only timings/plot, no GIFs

The script is self-contained; the only new run-time dependency is **imageio**
(`pip install imageio`).
"""

from __future__ import annotations

import argparse, os, math, time, io
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import imageio.v2 as imageio               # ⬅ requirement: pip install imageio

from dataGen import generate_trajectory
from neuralop.models import FNO            # pip install neuralop

# ---------- hyper-parameters ----------
T_VALUES   = list(range(5, 205, 5))        # 5,10,…,200  (inclusive)
N_SAMPLES  = 128                           # trajectories per T for benchmarks
NX = NY   = 64                             # grid (must match dataGen.py)
DX = DY   = 1.0 / (NX - 1)
DT         = 0.01
T_INTERVAL = 1000                          # physical horizon (fixed)
ALPHA      = 1e-3                          # diffusivity
BATCH_SIZE = 32                            # FNO inference mini-batches
WARMUP     = 3                             # GPU warm-up forward passes
# --------------------------------------

THIS_DIR = Path(__file__).resolve().parent
PLOT_PATH = THIS_DIR / "benchmark_heat_dataset_vs_inference.png"
ANIM_DIR  = THIS_DIR / "animations"
MODEL_DIR = THIS_DIR / "models"
ANIM_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Timing helpers
# -----------------------------------------------------------------------------

def _cuda_sync_if(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def time_dataset_gen(T: int) -> float:
    """Return wall-clock seconds to generate *N_SAMPLES* trajectories of length *T*."""
    start = time.perf_counter()
    for _ in range(N_SAMPLES):
        generate_trajectory(NX, NY, DX, DY, DT, ALPHA, T_INTERVAL, T, mode="mixed")
    _cuda_sync_if(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return time.perf_counter() - start


def time_inference(T: int, device: torch.device) -> float:
    """Return seconds for *N_SAMPLES* forward passes through a size-compatible FNO."""
    model = FNO(n_modes=(20, 20), hidden_channels=64,
                in_channels=1, out_channels=T).to(device).eval()

    # synthetic batch once in RAM/VRAM (re-used)
    x = torch.randn(BATCH_SIZE, 1, NX, NY, device=device)

    def _now() -> float:
        _cuda_sync_if(device)
        return time.perf_counter()

    # warm-up (important on GPUs so the first call isn't dominated by JIT / alloc)
    with torch.no_grad():
        for _ in range(WARMUP):
            _ = model(x)

    n_iter = math.ceil(N_SAMPLES / BATCH_SIZE)
    start = _now()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = model(x)
    return _now() - start

# -----------------------------------------------------------------------------
# GIF creation helpers (ground-truth trajectories only — no FNO prediction)
# -----------------------------------------------------------------------------

def make_animation(T: int, out_dir: Path = ANIM_DIR, cmap: str = "inferno") -> None:
    """Generate **one** trajectory of length T, dump PNGs, and create a GIF."""
    traj0, traj = generate_trajectory(NX, NY, DX, DY, DT, ALPHA,
                                      T_INTERVAL, T, mode="mixed")
    frames: list[imageio.core.util.Array] = []

    print(f"🎞  Building animation for T={T} …")
    for t in range(T):
        fig, ax = plt.subplots(figsize=(3,3))
        im = ax.imshow(traj[t], cmap=cmap, origin="lower")
        ax.set_title(f"T={T}  |  t={t}")
        ax.axis("off")
        fig.tight_layout(pad=0.1)

        # render figure to RGB array *in-memory* (avoids temp PNG files)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    gif_path = out_dir / f"heat_trajectory_T{T}.gif"
    imageio.mimsave(gif_path, frames, fps=12)      # ~12 fps looks smooth enough
    print(f"   ↳ saved  →  {gif_path.relative_to(THIS_DIR)}  ({T} frames)")

# -----------------------------------------------------------------------------
# Model persistence helpers
# -----------------------------------------------------------------------------

def save_fno_stub(T: int, device: torch.device, out_dir: Path = MODEL_DIR) -> Path:
    """Instantiate a correctly sized FNO(**out_channels=T**) and *save* its weights.

    The network is **untrained** — the goal is merely to preserve the exact
    architecture that the speed benchmark used.  You can reload and fine-tune
    on a proper dataset later with `torch.load(.., map_location)`.
    """
    model = FNO(n_modes=(20, 20), hidden_channels=64,
                in_channels=1, out_channels=T).to(device)
    path = out_dir / f"fno_stub_T{T}.pth"
    torch.save(model.state_dict(), path)
    return path

# -----------------------------------------------------------------------------
# Main entry
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Benchmark Heat-Eq dataset generation vs FNO inference and make per-T animations.")
    parser.add_argument("--no-gpu",       action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--skip-anim",    action="store_true", help="Skip trajectory GIF creation to save time")
    parser.add_argument("--skip-model",   action="store_true", help="Skip saving stub FNO state_dicts")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda")
    print(f"Running on {device} …\n")

    rows: list[dict[str, float]] = []
    for T in T_VALUES:
        print(f"⏱  Profiling T={T}")
        t_gen = time_dataset_gen(T)
        t_inf = time_inference(T, device)
        rows.append(dict(T=T, dataset_sec=t_gen, inference_sec=t_inf))

        # optional extras ------------------------------------------------------
        if not args.skip_model:
            path = save_fno_stub(T, device)
            print(f"💾  Saved stub model → {path.relative_to(THIS_DIR)}")
        if not args.skip_anim:
            make_animation(T)

    # ----------------------------- summary/plot ------------------------------
    df = pd.DataFrame(rows)
    print("\nTiming summary (log-seconds):\n", df.to_string(index=False, formatters={
        "dataset_sec": "{:.3e}".format, "inference_sec": "{:.3e}".format}))

    # Plot curves -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(df.T, df.dataset_sec,  marker="o", label="Dataset generation")
    ax.plot(df.T, df.inference_sec, marker="s", label="FNO inference")
    ax.set_xlabel("Number of stored time steps T")
    ax.set_ylabel(f"Wall-clock seconds (for {N_SAMPLES} samples)")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    # crossover — first T where inference ≥ generation ------------------------
    cross = df[df.inference_sec >= df.dataset_sec].head(1)
    if not cross.empty:
        T_star = int(cross.T.iloc[0])
        ax.axvline(T_star, color="grey", ls=":")
        ax.text(T_star, ax.get_ylim()[1]*0.45,
                f"crossover ≈ T = {T_star}", rotation=90,
                ha="right", va="center", fontsize=9,
                bbox=dict(fc="white", ec="grey", alpha=0.8))

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"\n📈  Saved timing plot →  {PLOT_PATH.relative_to(THIS_DIR)}")

if __name__ == "__main__":
    main()
