import argparse,math, shutil
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

from dataGen import generate_initial_condition, generate_trajectory, save_dataset
from neuralop.models import FNO

from benchmark_heat import torch_now, save_time, inference_loop, build_loaders, train_fno

# -------------------------- default hyper-parameters --------------------------
DEF_MIN_NXY, DEF_MAX_NXY, DEF_NXY_STEP = 16, 64, 16 # spatial resolution bounds
DEF_DX = DEF_DY = 0
DEF_T           = 250                    # T is how many frames for gt & inference
DEF_N_SAMPLES   = 1500                   # trajectories per T (keep small => quick)
DEF_N_EPOCHS    = 50                   # FNO training epochs
DEF_BATCH_SIZE  = 32
DEF_ALPHA       = 1e-3                  # lr
DEF_DT          = 0.01                  # physical timestep
DEF_T_INTERVAL  = 500                    # total timesteps per frame
HIDDEN_CHANNELS = 96
# -----------------------------------------------------------------------------

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

def main():
    parser = argparse.ArgumentParser(description="Benchmark + visualise FNO on the heat equation")
    parser.add_argument("--xy_min",  type=int, default=DEF_MIN_NXY)
    parser.add_argument("--xy_max",  type=int, default=DEF_MAX_NXY)
    parser.add_argument("--xy_step", type=int, default=DEF_NXY_STEP)
    parser.add_argument("--timestep", "-T", type=int, default=DEF_T)
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

    T = args.timestep

    rows = []
    # Inferencing multiple Models per T (min, max, and steps)
    for xy in range(args.xy_min, args.xy_max + 1, args.xy_step):
        print(f"\n=====  Resolution = {xy}  =====")

        global DEF_NX, DEF_NY, DEF_DX, DEF_DY
        DEF_NX = DEF_NY = xy
        DEF_DX = DEF_DY = 1.0 / (DEF_NX - 1)

        # ---------- dataset generation + timing ----------
        # Total Dataset Generation + Initial Conditions
        t0 = torch_now(device)
        u0_tensor, uT_tensor, t_gen = gen_dataset(T, args.samples, device=device)
        t1 = torch_now(device) - t0
        print(f"Dataset generated in {t1:.2f} s  ({args.samples} samples x {T} trajs (frames) x {DEF_T_INTERVAL} steps (time resolution))")

        # ---------- data loaders ----------
        train_dl, val_dl, test_dl = build_loaders(u0_tensor, uT_tensor, args.batch)

        # ---------- inference timing (tiny random network) ----------
        model_time = FNO(n_modes=(DEF_NX,DEF_NY), hidden_channels=HIDDEN_CHANNELS,
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

        del model_time, x_dummy
        torch.cuda.empty_cache()
        
        
        model = FNO(n_modes=(DEF_NX,DEF_NY), hidden_channels=HIDDEN_CHANNELS,
                    in_channels=1, out_channels=T).to(device)

        # ---------- optional training / visualisation ----------
        if not args.no_train:
            t_dir = out_root / f"T_{T}_resolution_{xy}"
            if t_dir.exists():
                shutil.rmtree(t_dir)
            t_dir.mkdir(parents=True)

            print("Training FNO ...")
            train_fno(model, train_dl, val_dl, args.epochs, device, t_dir)

            # ---- save checkpoint ----
            ckpt_path = models_dir / f"fno_T_{T}_resolution_{xy}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved model -> {ckpt_path}")
        else: # Load pretrained model
            ckpt_path = models_dir / f"fno_T_{T}_resolution_{xy}.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print(f"Loaded model -> {ckpt_path}")

        # ---- trajectory animation ----
        t_inf = inference_loop(model, test_dl, t_dir, T, device)

        # free up GPU memory
        del model, train_dl, val_dl, test_dl, u0_tensor, uT_tensor
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # defragment the cache
        
        avg_t_data_gen = sum(t_gen)/len(t_gen)
        avg_t_inf = sum(t_inf)/len(t_inf)
        rows.append(dict(xy=xy, dataset_sec=avg_t_data_gen, inference_sec=avg_t_inf))

    # ===== global timing plot csv =====
    df = pd.DataFrame(rows)
    df.to_csv(out_root / "timings.csv", index=False)

    # ------- timing plot -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["xy"], df["dataset_sec"],  "o-", label="dataset generation")
    ax.plot(df["xy"], df["inference_sec"], "s-", label="FNO inference")
    ax.set_xlabel("spatial resolution")
    ax.set_ylabel(f"seconds for {args.samples} samples (log-scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--")
    ax.legend()

    cross = df[df.inference_sec >= df.dataset_sec].head(1)
    if not cross.empty:
        xy_star = cross["xy"].iloc[0]          # <─ pick the scalar safely
        ax.axvline(xy_star, color="grey", ls=":")
        ax.text(
            xy_star, ax.get_ylim()[1] * 0.5,
            f"crossover\nxy≈{xy_star}",
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