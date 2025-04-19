# benchmark_heat.py — extended benchmarking & visualization for 2-D Heat-Equation
"""
This script now **benchmarks**, **plots**, and **animates** Heat-equation data while
optionally showcasing **FNO predictions with per-frame loss** alongside ground
truth.  Highlights:

1. **Benchmark** dataset-generation vs. FNO inference for stored-step counts
   `T = 5 … 200`.
2. **Plot** both timing curves (log-scale) and indicate the first crossover.
3. Animation for every `T`: each frame shows ground truth and
   prediction side-by-side, with the MSE loss in the title.
4. Model handling
    If a trained weight file `MODEL_DIR/fno_T{T}.pth` exists it is loaded.
    Otherwise an untrained stub is created and saved (use `--skip-model` to
    disable).

CLI switches:

```bash
python benchmark_heat.py          # full run: timings + GIFs + model stubs
python benchmark_heat.py --skip-anim            # no GIFs
python benchmark_heat.py --skip-model           # no model saving
python benchmark_heat.py --model-dir trained/   # directory with trained .pth
python benchmark_heat.py --no-gpu               # CPU only
```

> **New dependency**: `imageio` (`pip install imageio`).
"""

from __future__ import annotations

import argparse, io, math, time
from pathlib import Path

import imageio.v2 as imageio          # ⇐ new (GIF writing)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataGen import generate_trajectory
from neuralop.models import FNO       # pip install neuralop

# ──────────────────────────────────────────────────────────────────────────────
# Hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────
T_VALUES   = list(range(5, 205, 5))   # 5,10,…,200 (inclusive)
N_SAMPLES  = 128                      # trajectories per T for benchmarks
NX = NY   = 64                       # grid (must match dataGen.py)
DX = DY   = 1.0 / (NX - 1)
DT         = 0.01
T_INTERVAL = 1000                     # physical horizon (fixed)
ALPHA      = 1e-3                     # diffusivity
BATCH_SIZE = 32                       # FNO inference mini-batches
WARMUP     = 3                        # GPU warm-up forward passes

THIS_DIR   = Path(__file__).resolve().parent
PLOT_PATH  = THIS_DIR / "benchmark_heat_dataset_vs_inference.png"
ANIM_DIR   = THIS_DIR / "animations"
MODEL_DIR  = THIS_DIR / "models"      # default for stubs / trained nets
ANIM_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Utility helpers
# ──────────────────────────────────────────────────────────────────────────────

def _sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize()

def time_dataset_gen(T: int) -> float:
    """Return seconds to generate *N_SAMPLES* trajectories of length *T*."""
    start = time.perf_counter()
    for _ in range(N_SAMPLES):
        generate_trajectory(NX, NY, DX, DY, DT, ALPHA, T_INTERVAL, T, mode="mixed")
    _sync(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return time.perf_counter() - start

def build_fno(T: int, device: torch.device) -> FNO:
    return FNO(n_modes=(20, 20), hidden_channels=64,
               in_channels=1, out_channels=T).to(device)

def load_or_stub_fno(T: int, device: torch.device, model_dir: Path, save_stub: bool) -> FNO:
    """Load trained FNO if present else create & optionally save stub."""
    path = model_dir / f"fno_T{T}.pth"
    model = build_fno(T, device)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        print(f"🎯  Loaded trained model  ←  {path.relative_to(THIS_DIR)}")
    else:
        if save_stub:
            torch.save(model.state_dict(), path)
            print(f"💾  Saved stub model     →  {path.relative_to(THIS_DIR)}")
        else:
            print("⚠️   Using untrained stub (not saved)")
    model.eval()
    return model

def time_inference(T: int, device: torch.device) -> float:
    """Seconds for *N_SAMPLES* forward passes through an FNO of shape T."""
    model = build_fno(T, device).eval()
    x = torch.randn(BATCH_SIZE, 1, NX, NY, device=device)
    def now(): _sync(device); return time.perf_counter()
    with torch.no_grad():
        for _ in range(WARMUP):
            model(x)
    iters = math.ceil(N_SAMPLES / BATCH_SIZE)
    start = now()
    with torch.no_grad():
        for _ in range(iters):
            model(x)
    return now() - start

# ──────────────────────────────────────────────────────────────────────────────
# Animation helper
# ──────────────────────────────────────────────────────────────────────────────

def make_animation(T: int, model: FNO, device: torch.device, out_dir: Path = ANIM_DIR,
                   cmap: str = "inferno") -> None:
    """Generate one trajectory, predict with `model`, and write a side-by-side GIF."""
    u0, gt = generate_trajectory(NX, NY, DX, DY, DT, ALPHA, T_INTERVAL, T, mode="mixed")

    # ── model prediction ────────────────────────────────────────────────────
    with torch.no_grad():
        x0 = torch.tensor(u0, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
        pred = model(x0)[0].cpu().numpy()        # (T, nx, ny)
    gt   = gt                                   # (T, nx, ny) numpy already

    vmin = min(gt.min(), pred.min())
    vmax = max(gt.max(), pred.max())

    frames: list[imageio.core.util.Array] = []
    print(f"🎞  Building GT vs Pred animation for T={T} …")
    for t in range(T):
        loss = ((pred[t] - gt[t])**2).mean()

        fig, axs = plt.subplots(1, 2, figsize=(4.8, 2.4))
        im0 = axs[0].imshow(gt[t], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axs[0].set_title(f"Ground Truth\nt={t}")
        axs[0].axis("off")

        im1 = axs[1].imshow(pred[t], cmap=cmap, vmin=vmin, vmax=vmax, origin="lower")
        axs[1].set_title(f"Prediction\nMSE={loss:.2e}")
        axs[1].axis("off")

        fig.tight_layout(pad=0.1)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        frames.append(imageio.imread(buf))

    gif_path = out_dir / f"heat_GT_vs_Pred_T{T}.gif"
    imageio.mimsave(gif_path, frames, fps=12)
    print(f"   ↳ saved  →  {gif_path.relative_to(THIS_DIR)}  ({T} frames)")

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Benchmark Heat-Eq generation vs FNO inference + GT/Pred animations.")
    ap.add_argument("--no-gpu",      action="store_true", help="Force CPU even if CUDA is available")
    ap.add_argument("--skip-anim",   action="store_true", help="Skip GIF creation")
    ap.add_argument("--skip-model",  action="store_true", help="Do not save stub models if none exist")
    ap.add_argument("--model-dir",   type=Path, default=MODEL_DIR, help="Directory containing trained FNO weights *.pth")
    args = ap.parse_args()

    device = torch.device("cpu" if args.no_gpu or not torch.cuda.is_available() else "cuda")
    print(f"Running on {device}\n")

    rows: list[dict[str, float]] = []
    for T in T_VALUES:
        print(f"⏱  Profiling T={T}")
        t_gen = time_dataset_gen(T)
        t_inf = time_inference(T, device)
        rows.append(dict(T=T, dataset_sec=t_gen, inference_sec=t_inf))

        # handle model (trained or stub)
        model = load_or_stub_fno(T, device, args.model_dir, save_stub=not args.skip_model)

        # optional GIF --------------------------------------------------------
        if not args.skip_anim:
            make_animation(T, model, device)

    # ── timing plot ──────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    print("\nTiming summary (seconds):\n", df.to_string(index=False, formatters={
        "dataset_sec": "{:.3e}".format, "inference_sec": "{:.3e}".format}))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(df.T, df.dataset_sec,  marker="o", label="Dataset generation")
    ax.plot(df.T, df.inference_sec, marker="s", label="FNO inference")
    ax.set_xlabel("Number of stored time steps T")
    ax.set_ylabel(f"Wall-clock seconds (for {N_SAMPLES} samples)")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--", lw=0.5)
    ax.legend()

    cross = df[df.inference_sec >= df.dataset_sec].head(1)
    if not cross.empty:
        T_star = int(cross.T.iloc[0])
        ax.axvline(T_star, color="grey", ls=":")
        ax.text(T_star, ax.get_ylim()[1]*0.5,
                f"crossover ≈ T = {T_star}", rotation=90,
                ha="right", va="center", fontsize=9,
                bbox=dict(fc="white", ec="grey", alpha=0.8))

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=200)
    plt.close(fig)
    print(f"\n📈  Saved timing plot →  {PLOT_PATH.relative_to(THIS_DIR)}")

if __name__ == "__main__":
    main()
