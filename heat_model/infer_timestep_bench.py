import argparse, shutil
from pathlib import Path
from datetime import datetime

import gc

import torch
import pandas as pd
import matplotlib.pyplot as plt

from neuralop.models import FNO

from benchmark_heat import torch_now, gen_dataset, inference_loop, build_loaders, train_fno

# -------------------------- default hyper-parameters for datagen inferencing  --------------------------
DEF_N_SAMPLES   = 500                   # trajectories per T (keep small => quick)
DEF_N_EPOCHS    = 200                   # FNO training epochs
DEF_TRAIN_BS    = 32
DEF_BATCH_SIZE  = 1                     # 1 = Stochastic, >1 = Batches
DEF_ALPHA       = 1e-3                  # diffusion coefficient
DEF_NX = DEF_NY = 32                    # spatial resolution
DEF_DX = DEF_DY = 1.0 / (DEF_NX - 1)
DEF_DT          = 0.01                  # physical timestep
DEF_T_INTERVAL  = 500                    # total timesteps per frame
HIDDEN_CHANNELS = 128
# -----------------------------------------------------------------------------

# -------------------------- parameters for loading model(s) (inference) --------------------------
# Select Model(s)
DEF_T_MIN, DEF_T_MAX, DEF_T_STEP = 25, 200, 5 # Use timesteps between MIN and MAX (models)
DEF_MODEL_N = 500                               # Number of samples the model was trained on
# ------------------------------------------------------------------------------

# -------------------------- option for training --------------------------
DEF_NO_TRAIN = True


# Train
def train(args, out_root, models_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = []
    for T in range(args.t_min, args.t_max + 1, args.t_step):
        # ---------- dataset generation + timing ----------
        # Total Dataset Generation + Initial Conditions
        t0 = torch_now(device)
        u0_tensor, uT_tensor, t_gen = gen_dataset(T=T, nx=DEF_NX, ny=DEF_NY, 
                                                  dx=DEF_DX, dy=DEF_DY, dt=DEF_DT, 
                                                  alpha=DEF_ALPHA, nt=DEF_T_INTERVAL, 
                                                  n_samples=args.samples, device=device)
        t1 = torch_now(device) - t0
        print(f"Dataset generated in {t1:.2f} s  ({args.samples} samples x {T} trajs (frames) x {DEF_T_INTERVAL} steps (time resolution))")

        # ---------- data loaders ----------
        train_dl, val_dl, test_dl = build_loaders(u0_tensor, uT_tensor, DEF_TRAIN_BS)

        # ---------- training / visualisation ----------
        model = FNO(n_modes=(int(DEF_NX), int(DEF_NY)), hidden_channels=HIDDEN_CHANNELS,
            in_channels=1, out_channels=T).to(device)
        
        t_dir = out_root / f"T_{T}"
        if t_dir.exists():
            shutil.rmtree(t_dir)
        t_dir.mkdir(parents=True)

        print("Training FNO ...")
        train_fno(model, train_dl, val_dl, args.epochs, device, t_dir)

        # ---- save checkpoint ----
        ckpt_path = models_dir / f"T_{T}"

        ckpt_path.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), ckpt_path / f"fno_T_{T}_samples_{args.samples}_epochs_{args.epochs}.pth")
        print(f"Saved model -> {ckpt_path}")

        # ---- trajectory animation ----
        t_inf, _ = inference_loop(model, test_dl, t_dir, T, device)

        # free up GPU memory
        del model, train_dl, val_dl, test_dl, u0_tensor, uT_tensor
        gc.collect()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # defragment the cache
        
        avg_t_data_gen = sum(t_gen)/len(t_gen)
        avg_t_inf = sum(t_inf)/len(t_inf)
        rows.append(dict(T=T, dataset_sec=avg_t_data_gen, inference_sec=avg_t_inf))
    
    out_root.mkdir(parents=True, exist_ok=True)

    # ===== global timing plot csv =====
    df = pd.DataFrame(rows)
    df.to_csv(out_root / f"timings_timestep_{args.t_min}_{args.t_step}.csv", index=False)

# ===== main ==================================================================

def timing_plot(df, samples, dir, res, timestep, channel=None):
    # ------- timing plot -------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["T"], df["dataset_sec"],  "o-", label="Dataset Generation")
    ax.plot(df["T"], df["inference_sec"], "s-", label="FNO Inference")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel(f"seconds for {samples} samples (log-scale)")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls="--")
    ax.set_title(f"Inference Time vs Dataset Generation Time, Model Trained Dim ({res}x{res})")
    ax.legend()

    cross = df[df.inference_sec >= df.dataset_sec].head(1)
    if not cross.empty:
        T_star = cross["T"].iloc[0]          # <- pick the scalar safely
        ax.axvline(T_star, color="grey", ls=":")
        ax.text(
            T_star, ax.get_ylim()[1] * 0.5,
            f"crossover\\T{T_star}",
            ha="right", va="center", rotation=90,
            bbox=dict(fc="white", ec="grey", alpha=0.7),
        )
    
    plt.tight_layout()
    fig_path = dir / f"benchmark_heat_dataset_vs_inference_model_time{timestep}.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("\nSaved timing plot ->", fig_path)

def plot_loss_test(results, save_dir=Path("."), samples=None):
    """
    Plot the average test-set MSE loss for every trained-T model.

    Parameters
    ----------
    results  : dict   # populated in main(); results[T] is a list of dict rows
    save_dir : Path   # where to save the figure
    samples  : int    # number of test samples used when evaluating each model
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "test_loss_curve.png"

    # ---- collate mean loss per‑model -----------------------------------
    Ts         = sorted(results.keys())
    avg_losses = [pd.DataFrame(results[T])["loss"].mean() for T in Ts]

    # ---- plotting ------------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(
        Ts,
        avg_losses,
        marker="o",
        linestyle="-",
        linewidth=2,
        markersize=6,
        label="Avg MSE"
    )
    plt.yscale("log")
    plt.xlabel("Training timestep $T$")
    plt.ylabel("Average MSE (log scale)")
    plt.title(f"Stability across timesteps (N={samples} test samples)")
    plt.grid(True, which="both", ls="--")

    # annotate exact values
    for T, loss in zip(Ts, avg_losses):
        plt.text(
            T,
            loss,
            f"{loss:.2e}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved loss-curve plot -> {fig_path}")

def plot_timing_all(results, save_dir=Path("."), samples=None):
    """
    Aggregate timing information over ALL trained-T models
    and draw a line/curve plot (log-y) similar to timing_plot.

    Parameters
    ----------
    results : dict   # same structure produced in `main`
    save_dir : Path  # where to drop the figure
    samples : int    # N used when generating each dataset
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    fig_path = save_dir / "timing_curve_all_T.png"

    # ---- collect means per‑T ------------------------------------------
    Ts             = sorted(results.keys())
    mean_dataset   = [pd.DataFrame(results[T])["dataset_sec"].mean()   for T in Ts]
    mean_infer     = [pd.DataFrame(results[T])["inference_sec"].mean() for T in Ts]

    # ---- plotting -----------------------------------------------------
    plt.figure(figsize=(8, 6))
    plt.plot(Ts, mean_dataset, "o-", label="Dataset Generation")
    plt.plot(Ts, mean_infer,   "s-", label="FNO Inference")
    plt.yscale("log")
    plt.xlabel("Training timestep T")
    plt.ylabel(f"seconds for {samples} samples (log scale)")
    plt.title(f"Avg Timing vs T (dataset gen vs inference)")
    plt.grid(True, which="both", ls="--")
    plt.legend()

    # crossover marker (first T where inference ≥ dataset)
    for T, d, inf in zip(Ts, mean_dataset, mean_infer):
        if inf >= d:
            plt.axvline(T, color="grey", ls=":")
            plt.text(T, plt.ylim()[1]*0.5, f"crossover T={T}",
                     ha="right", va="center", rotation=90,
                     bbox=dict(fc="white", ec="grey", alpha=0.7))
            break

    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print(f"Saved aggregated timing curve -> {fig_path}")


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device};  started {datetime.now().isoformat(timespec='seconds')}")

    out_root = Path(args.out);  out_root.mkdir(exist_ok=True)
    models_dir = out_root / "models";   models_dir.mkdir(exist_ok=True)
    
    out_base = out_root / "test"          # <─ mirrors infer_res_bench.py
    out_base.mkdir(exist_ok=True)

    # Train model if required
    if not args.no_train:
        train(args, out_root / "train", models_dir)

    results = {}
    # Generating Dataset of multiple res to test on models with differing res (testing resolution invariance)
    for T in range(DEF_T_MIN, DEF_T_MAX + 1, DEF_T_STEP):
        print(f"\n=====  Timesteps = {T}  =====")

        # ---------- dataset generation + timing ----------
        # Total Dataset Generation + Initial Conditions
        t0 = torch_now(device)
        u0_tensor, uT_tensor, t_gen = gen_dataset(T=T, nx=DEF_NX, ny=DEF_NY, 
                                                  dx=DEF_DX, dy=DEF_DY, dt=DEF_DT, 
                                                  alpha=DEF_ALPHA, nt=DEF_T_INTERVAL, 
                                                  n_samples=args.samples, device=device)
        t1 = torch_now(device) - t0
        print(f"Dataset generated in {t1:.2f} s  ({args.samples} samples x {T} trajs (frames) x {DEF_T_INTERVAL} steps (time resolution))")

        # ---------- data loaders ----------
        train_dl, val_dl, test_dl = build_loaders(u0_tensor, uT_tensor, batch_size=args.batch, train=0, val=0) # load entire dataset as test
        
        model = None

        # ---------- loading model for inference ------------
        # Test Inference on different resolutions datasets for each model
        t_dir = out_base / f"model_time_{T}" / f"resolution_{DEF_NX}" / f"channels_{HIDDEN_CHANNELS}"

        model = FNO(n_modes=(int(DEF_NX), int(DEF_NY)), 
                    hidden_channels=HIDDEN_CHANNELS, 
                    in_channels=1, out_channels=T).to(device)
        
        ckpt_path = models_dir / f"T_{T}" / f"fno_T_{T}_samples_{DEF_MODEL_N}_epochs_{args.epochs}.pth"

        model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
        print(f"Loaded model -> {ckpt_path}")

        if t_dir.exists():
            shutil.rmtree(t_dir)
        t_dir.mkdir(parents=True)

        # ---- trajectory animation ----
        t_inf, avg_loss = inference_loop(model, test_dl, t_dir, T, device)
        
        avg_t_data_gen = sum(t_gen)/len(t_gen)
        avg_t_inf = sum(t_inf)/len(t_inf)

        # Check if model is already in results dict
        if T not in results:
            results[T] = []
        
        # add to dataframe
        results[T].append(dict(T=T, dataset_sec=avg_t_data_gen, inference_sec=avg_t_inf, loss=avg_loss))

        # free up GPU memory
        del model, train_dl, val_dl, test_dl, u0_tensor, uT_tensor
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # defragment the cache

        # Programming like it's C :D
        model = None
    
    # print(f"results: {results}")

    # ===== global timing plot csv =====
    # each models will have their own csv
    for timestep, rows in results.items():
        t_dir = out_base / f"model_time_{timestep}" / f"resolution_{DEF_NX}" / f"channels_{HIDDEN_CHANNELS}"
        t_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(t_dir / f"dataset_timings_{args.t_min}_{args.t_max}_{args.t_step}_samples_{args.samples}.csv", index=False)
        
        timing_plot(df=df, samples=args.samples, dir=t_dir, res=DEF_NX, timestep=timestep)

    # --- aggregated summaries -----------------------------------------
    plot_timing_all(results=results, save_dir=out_base, samples=args.samples)
    plot_loss_test(results=results, save_dir=out_base, samples=args.samples)


    print("All done!")


def argparse_main():
    parser = argparse.ArgumentParser(description="Benchmark + visualise FNO on the heat equation")
    parser.add_argument("--t_min",  type=int, default=DEF_T_MIN)
    parser.add_argument("--t_max",  type=int, default=DEF_T_MAX)
    parser.add_argument("--t_step", type=int, default=DEF_T_STEP)
    parser.add_argument("--samples",  "-N", type=int, default=DEF_N_SAMPLES)
    parser.add_argument("--epochs",   "-E", type=int, default=DEF_N_EPOCHS)
    parser.add_argument("--batch",    "-B", type=int, default=DEF_BATCH_SIZE)
    parser.add_argument("--out",              default="results/timestep")
    parser.add_argument("--no-train", help="skip training/visualisation steps (speed plot only)", default=DEF_NO_TRAIN)
    parser.add_argument("--stochastic", action="store_true", help="benchmark stochastic inference")
    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    argparse_main()