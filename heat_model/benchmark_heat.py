# benchmark_heat.py
"""
Extended benchmark script for the 2-D heat-equation example
===========================================================

This script now performs *three* tasks for a sweep of stored
 time-steps **T = 5 ... 200** (step 5):

1. **Speed benchmark** ― wall-clock seconds required to
   • generate *N_SAMPLES* trajectories (traditional FD solver), and
   • run the same number of forward passes through an *FNO* model.
   Both curves are plotted on a log-scale and the first
   crossover point is annotated.

2. **Model training + evaluation** ― for every *T* a small FNO
   (unnecessarily small for scientific usage but fast enough for a
   demonstrator) is trained for *N_EPOCHS* on a freshly generated
   dataset (80 %/10 %/10 % split).  A loss-history plot is saved to
   *results/T_XX/loss_curve.png*.

3. **Trajectory visualisation** ― for the first test sample we
   predict an entire trajectory and save two-panel frames
   (ground-truth | prediction) for every stored step.  The frames are
   bundled into an animated GIF *trajectory.gif* using **imageio**.

All artefacts live below *results/T_XX/*, and an extra *models/*
folder keeps a checkpoint *fno_T_XX.pth* for each time-horizon.
Feel free to tweak hyper-parameters with command-line flags; run

    python benchmark_heat.py --help

for details.
"""

import argparse, time, math, shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import imageio.v2 as imageio

from dataGen import generate_initial_condition, generate_trajectory, save_dataset
from neuralop.models import FNO


# -------------------------- default hyper-parameters --------------------------
DEF_N_SAMPLES   = 1000         # trajectories per T  (keep small ⇒ quick)
DEF_T_MIN, DEF_T_MAX, DEF_T_STEP = 25, 500, 25 # T is how many frames for gt & inference
DEF_N_SAMPLES   = 1000                   # trajectories per T (keep small => quick)
DEF_N_EPOCHS    = 50                   # FNO training epochs
DEF_BATCH_SIZE  = 32
DEF_ALPHA       = 1e-3                  # lr
DEF_NX = DEF_NY = 64                    # spatial resolution
DEF_DX = DEF_DY = 1.0 / (DEF_NX - 1)
HIDDEN_CHANNELS = 512
DEF_DT          = 0.01                  # physical timestep
DEF_T_INTERVAL  = 500                    # total timesteps per frame
# -----------------------------------------------------------------------------

# ===== helpers ===============================================================

def torch_now(device):
    """Return *time.perf_counter()* synchronised with the given device."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    return time.perf_counter()


def save_time(time, n_samples, T):
    df = pd.DataFrame({"time": time, "n_samples": n_samples, "T": T})
    df.to_csv(f"results/time-{T}-samples-{n_samples}.csv", index=False)


def gen_dataset(T: int, n_samples: int, device):
    """Generate *n_samples* full trajectories of length *T*.

    Returns two tensors:
        u0  — shape (N, 1, H, W)
        uT  — shape (N, T, H, W)
    """
    u0_all, traj_all = [], []
    times = []
    for sample in tqdm(range(n_samples), desc=f"Generating dataset for {n_samples} samples"):
        init_condition = generate_initial_condition(DEF_NX, DEF_NY, mode="mixed")
        # Calculate the time for entire trajectory
        t0 = torch_now(device)
        u0, traj = generate_trajectory(
            nx=DEF_NX, ny=DEF_NY, dx=DEF_DX, dy=DEF_DY, dt=DEF_DT, alpha=DEF_ALPHA,
            nt=DEF_T_INTERVAL, n_frames=T, u=init_condition
        )
        times.append(torch_now(device) - t0)
        
        # print(f"[sample-{sample}]Time for calculating trajectory: {time[-1]}")

        u0_all.append(u0)
        traj_all.append(traj)

    u0_tensor = torch.tensor(np.stack(u0_all)[:, None], dtype=torch.float32)
    uT_tensor = torch.tensor(np.stack(traj_all),       dtype=torch.float32)

    save_dataset(u0_tensor, uT_tensor)
    save_time(times, n_samples, T)

    print(f"Average time per trajectory: {np.mean(times):.4f} s")

    return u0_tensor, uT_tensor, times


def build_loaders(u0_tensor: torch.Tensor, uT_tensor: torch.Tensor,
                  batch_size: int):
    """Split dataset 80/10/10 and return PyTorch DataLoaders."""
    full_ds     = TensorDataset(u0_tensor, uT_tensor)
    n_total     = len(full_ds)
    n_train     = int(0.8 * n_total)
    n_val       = int(0.1 * n_total)
    n_test      = n_total - n_train - n_val
    train_ds, val_ds, test_ds = random_split(full_ds, [n_train, n_val, n_test],
                                             generator=torch.Generator().manual_seed(0))

    def make_dl(ds):
        return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

    return make_dl(train_ds), make_dl(val_ds), make_dl(test_ds)


def train_fno(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader,
              n_epochs: int, device: torch.device, save_dir: Path):
    """Train *model* for *n_epochs* and save loss-curve plot in *save_dir*."""
    criterion  = nn.MSELoss()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    warmup_epochs = 15
    max_lr = 0.001

    train_hist, val_hist = [], []
    for epoch in range(1, n_epochs + 1):
        if epoch < warmup_epochs:
            lr = max_lr * (epoch + 1) / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == warmup_epochs:
            # Switch to cosine annealing after warmup
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-warmup_epochs, eta_min=1e-6)
        elif epoch > warmup_epochs:
            scheduler.step()

        # ---- training ----
        '''
        u0:     initial conditions (to be predicted/inferenced)
        traj:   full trajectories (ground truth)
        '''
        model.train();   train_loss = 0.0
        for u0, traj in train_loader:
            u0, traj = u0.to(device), traj.to(device)
            optimizer.zero_grad()
            pred = model(u0)
            loss = criterion(pred, traj)
            loss.backward();  optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        train_hist.append(train_loss)

        # ---- validation ----
        model.eval();   val_loss = 0.0
        with torch.no_grad():
            for u0, traj in val_loader:
                u0, traj = u0.to(device), traj.to(device)
                val_loss += criterion(model(u0), traj).item()
        val_loss /= len(val_loader)
        val_hist.append(val_loss)

        # scheduler.step()
        tqdm.write(f"    Epoch {epoch:02d}/{n_epochs} — train {train_loss:.2e}  val {val_loss:.2e}")

    # ---- plot loss history ----
    plt.figure(figsize=(4,3))
    plt.plot(train_hist, label="train")
    plt.plot(val_hist,   label="val")
    plt.yscale("log");  plt.xlabel("epoch");  plt.ylabel("MSE loss")
    plt.legend();   plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    return train_hist, val_hist


def save_frames_and_gif(T, save_dir, traj_true, traj_pred,
                             duration=0.08, cmap_name="inferno"):
    frames_dir = save_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # --- one figure reused for all frames ----------------------------------
    fig, axs = plt.subplots(1, 2, figsize=(5, 2.4))
    ims = [axs[0].imshow(traj_true[0].cpu(), cmap=cmap_name, animated=True),
           axs[1].imshow(traj_pred[0].cpu(), cmap=cmap_name, animated=True)]
    for ax in axs: ax.axis("off")
    txt0 = axs[0].set_title("")
    txt1 = axs[1].set_title("pred")
    plt.tight_layout()

    w, h = fig.canvas.get_width_height()
    buffer = np.empty((h, w, 3), dtype=np.uint8)

    # --- write GIF incrementally -------------------------------------------
    gif_path = save_dir / "trajectory.gif"
    with imageio.get_writer(gif_path, mode="I", duration=duration) as writer:
        for t in range(T):
            ims[0].set_data(traj_true[t].cpu())
            ims[1].set_data(traj_pred[t].cpu())
            txt0.set_text(f"GT  t={t}")

            fig.canvas.draw()                       # render once
            buf_rgba = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            buffer[:] = buf_rgba.reshape(buffer.shape[0], buffer.shape[1], 4)[..., :3]
            writer.append_data(buffer)

            # (optional) keep PNGs if you still want them
            # imageio.imwrite(frames_dir / f"frame_{t:03d}.png", buffer)

    plt.close(fig)
    print("Animation written ->", gif_path)


def inference_loop(model, test_dl, t_dir, T, device):
    model.eval()
    times = []

    with torch.no_grad():
        for batch_idx, (u0, traj_true) in enumerate(tqdm(test_dl)):
            u0, traj_true = u0.to(device), traj_true.to(device)

            t0 = torch_now(device)
            traj_pred = model(u0)            # shape (B, T, H, W)
            t1 = torch_now(device)

            times.append(t1 - t0)

            # save GIF for the *first* item of the *first* batch only
            if batch_idx == 0:
                save_frames_and_gif(
                    T, t_dir,
                    traj_true=traj_true[0],     # first sample in batch
                    traj_pred=traj_pred[0]
                )

    print(f"Avg. inference time per batch: {sum(times)/len(times):.4f} s")
    print(f"Batch Time: {times}")

    return times


# ===== main ==================================================================

def main():
    parser = argparse.ArgumentParser(description="Benchmark + visualise FNO on the heat equation")
    parser.add_argument("--t_min",  type=int, default=DEF_T_MIN)
    parser.add_argument("--t_max",  type=int, default=DEF_T_MAX)
    parser.add_argument("--t_step", type=int, default=DEF_T_STEP)
    parser.add_argument("--samples",  "-N", type=int, default=DEF_N_SAMPLES)
    parser.add_argument("--epochs",   "-E", type=int, default=DEF_N_EPOCHS)
    parser.add_argument("--batch",    "-B", type=int, default=DEF_BATCH_SIZE)
    parser.add_argument("--out",              default="results")
    parser.add_argument("--no-train",   action="store_true", help="skip training/visualisation steps (speed plot only)")
    parser.add_argument("--infer",      action="store_true", help="skip training step (inference only)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device};  started {datetime.now().isoformat(timespec='seconds')}")

    out_root = Path(args.out);  out_root.mkdir(exist_ok=True)
    models_dir = out_root / "models";   models_dir.mkdir(exist_ok=True)

    rows = []
    # Multiple Models per T (min, max, and steps)
    for T in range(args.t_min, args.t_max + 1, args.t_step):
        print(f"\n=====  T = {T}  =====")

        # ---------- dataset generation + timing ----------
        # Total Dataset Generation + Initial Conditions
        t0 = torch_now(device)
        u0_tensor, uT_tensor, t_gen = gen_dataset(T, args.samples, device=device)
        t1 = torch_now(device) - t0
        print(f"Dataset generated in {t1:.2f} s  ({args.samples} samples x {T} trajs (frames) x {DEF_T_INTERVAL} steps (time resolution))")

        # ---------- inference timing (tiny random network) ----------
        model_time = FNO(n_modes=(20,20), hidden_channels=HIDDEN_CHANNELS,
                         in_channels=1, out_channels=T).to(device).eval()
        x_dummy = torch.randn(args.batch, 1, DEF_NX, DEF_NY, device=device)

        # Dry Run
        with torch.no_grad():                      # warm-up
            for _ in range(3):
                _ = model_time(x_dummy)
        t0 = torch_now(device)
        n_iter = math.ceil(args.samples / args.batch)
        with torch.no_grad():
            for _ in range(n_iter):
                _ = model_time(x_dummy)
        t_inf = torch_now(device) - t0
        print(f"FNO inference (untrained) done in {t_inf:.4f} s")

        # ---------- optional training / visualisation ----------
        if not args.no_train:
            t_dir = out_root / f"T_{T}"
            if t_dir.exists():
                shutil.rmtree(t_dir)
            t_dir.mkdir(parents=True)

            train_dl, val_dl, test_dl = build_loaders(u0_tensor, uT_tensor, args.batch)

            model = FNO(n_modes=(20,20), hidden_channels=64,
                        in_channels=1, out_channels=T).to(device)

            print("Training FNO ...")
            train_fno(model, train_dl, val_dl, args.epochs, device, t_dir)

            # ---- save checkpoint ----
            ckpt_path = models_dir / f"fno_T_{T}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved model -> {ckpt_path}")

            # ---- trajectory animation ----
            t_inf = inference_loop(model, test_dl, t_dir, T, device)
        
        avg_t_data_gen = sum(t_gen)/len(t_gen)
        avg_t_inf = sum(t_inf)/len(t_inf)
        rows.append(dict(T=T, dataset_sec=avg_t_data_gen, inference_sec=avg_t_inf))

    # ===== global timing plot csv =====
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "timings.csv", index=False)

    # ------- timing plot -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["T"], df["dataset_sec"],  "o-", label="dataset generation")
    ax.plot(df["T"], df["inference_sec"], "s-", label="FNO inference")
    ax.set_xlabel("stored time-steps  T")
    ax.set_ylabel(f"seconds for {args.samples} samples (log-scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    cross = df[df.inference_sec >= df.dataset_sec].head(1)
    if not cross.empty:
        T_star = cross["T"].iloc[0]          # <─ pick the scalar safely
        ax.axvline(T_star, color="grey", ls=":")
        ax.text(
            T_star, ax.get_ylim()[1] * 0.5,
            f"crossover\nT≈{T_star}",
            ha="right", va="center", rotation=90,
            bbox=dict(fc="white", ec="grey", alpha=0.7),
        )

    plt.tight_layout()
    fig_path = out_root / "benchmark_heat_dataset_vs_inference.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("\nSaved timing plot ->", fig_path)
    print("All done!")


if __name__ == "__main__":
    main()
