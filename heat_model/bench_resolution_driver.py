import argparse, shutil
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import matplotlib.pyplot as plt

from neuralop.models import FNO

from benchmark_heat import torch_now, gen_dataset, inference_loop, build_loaders, train_fno, dry_run

# -------------------------- default hyper-parameters --------------------------
DEF_MIN_NXY, DEF_MAX_NXY, DEF_NXY_STEP = 16, 64, 8 # spatial resolution bounds
DEF_T           = 250                    # T is how many frames for gt & inference
DEF_N_SAMPLES   = 500                   # trajectories per T (keep small => quick)
DEF_N_EPOCHS    = 50                   # FNO training epochs
DEF_BATCH_SIZE  = 32
DEF_ALPHA       = 1e-3                  # lr
DEF_DT          = 0.01                  # physical timestep
DEF_T_INTERVAL  = 500                    # total timesteps per frame
HIDDEN_CHANNELS = 64
# -----------------------------------------------------------------------------

# -------------------------- parameters for inference --------------------------
# Select Model
DEF_T_MODEL     = 250
DEF_RES_MODEL  =  48
DEF_HIDDEN_CHANNELS_MODEL = 96

# ===== main ==================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device};  started {datetime.now().isoformat(timespec='seconds')}")

    out_root = Path(args.out);  out_root.mkdir(exist_ok=True)
    models_dir = out_root / "models";   models_dir.mkdir(exist_ok=True)

    T = args.timestep

    rows = []
    # Generating Dataset for multiple Resolutions and train multiple models
    for xy in range(args.xy_min, args.xy_max + 1, args.xy_step):
        print(f"\n=====  Resolution = {xy}  =====")

        dx = dy = 1.0 / (xy - 1)

        # ---------- dataset generation + timing ----------
        # Total Dataset Generation + Initial Conditions
        t0 = torch_now(device)
        u0_tensor, uT_tensor, t_gen = gen_dataset(T=T, nx=xy, ny=xy, 
                                                  dx=dx, dy=dy, dt=DEF_DT, 
                                                  alpha=DEF_ALPHA, nt=DEF_T_INTERVAL, 
                                                  n_samples=args.samples, device=device)
        t1 = torch_now(device) - t0
        print(f"Dataset generated in {t1:.2f} s  ({args.samples} samples x {T} trajs (frames) x {DEF_T_INTERVAL} steps (time resolution))")

        # ---------- data loaders ----------
        train_dl, val_dl, test_dl = build_loaders(u0_tensor, uT_tensor, args.batch)

        # ---------- inference timing (tiny random network) ----------
        # model_time = FNO(n_modes=(DEF_NX,DEF_NY), hidden_channels=HIDDEN_CHANNELS,
        #                  in_channels=1, out_channels=T).to(device).eval()
        # x_dummy = torch.randn(args.batch, 1, DEF_NX, DEF_NY, device=device)

        # # Dry Run
        # dry_run(model=model_time, data=x_dummy, device=device, samples=args.samples, batch_size=args.batch)

        # del model_time, x_dummy
        # torch.cuda.empty_cache()
        
        
        model = None

        # ---------- optional training / visualisation ----------
        if not args.no_train:
            t_dir = out_root / "train" / f"T_{T}_resolution_{xy}_channels_{HIDDEN_CHANNELS}"
            if t_dir.exists():
                shutil.rmtree(t_dir)
            t_dir.mkdir(parents=True)

            model = FNO(n_modes=(xy,xy), hidden_channels=HIDDEN_CHANNELS,
                    in_channels=1, out_channels=T).to(device)

            print("Training FNO ...")
            train_fno(model, train_dl, val_dl, args.epochs, device, t_dir)

            # ---- save checkpoint ----
            ckpt_path = models_dir / f"fno_T_{T}_resolution_{xy}_channels_{HIDDEN_CHANNELS}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved model -> {ckpt_path}")

        else: # Load pretrained model
            t_dir = out_root / f"T_{DEF_T_MODEL}_resolution_{DEF_RES_MODEL}_channels_{DEF_HIDDEN_CHANNELS_MODEL}"
            model = FNO(n_modes=(DEF_RES_MODEL,DEF_RES_MODEL), 
                        hidden_channels=DEF_HIDDEN_CHANNELS_MODEL, 
                        in_channels=1, out_channels=DEF_T_MODEL).to(device)
            ckpt_path = models_dir / f"fno_T_{DEF_T_MODEL}_resolution_{DEF_RES_MODEL}_channels_{DEF_HIDDEN_CHANNELS_MODEL}.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            print(f"Loaded model -> {ckpt_path}")

            if t_dir.exists():
                shutil.rmtree(t_dir)
            t_dir.mkdir(parents=True)

        if args.stochastic:
            del train_dl, val_dl, test_dl
            train_dl, val_dl, test_dl = build_loaders(u0_tensor, uT_tensor, batch_size=1, train=0, val=0) # set batch size to 1 and entire dataset as test
    
        # ---- trajectory animation ----
        t_inf = inference_loop(model, test_dl, t_dir, T, device)

        # free up GPU memory
        del model, train_dl, val_dl, test_dl, u0_tensor, uT_tensor
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # defragment the cache
        
        avg_t_data_gen = sum(t_gen)/len(t_gen)
        avg_t_inf = sum(t_inf)/len(t_inf)
        rows.append(dict(xy=xy, dataset_sec=avg_t_data_gen, inference_sec=avg_t_inf))

    out_base = out_root / "train" if not args.no_train else "inference"

    # ===== global timing plot csv =====
    df = pd.DataFrame(rows)
    df.to_csv(out_base / f"timings_xy_{args.xy_min}_{args.xy_max}_{args.xy_step}.csv", index=False)

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
    fig_path = out_base / "benchmark_heat_dataset_vs_inference.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("\nSaved timing plot ->", fig_path)
    print("All done!")


def argparse_main():
    parser = argparse.ArgumentParser(description="Benchmark + visualise FNO on the heat equation")
    parser.add_argument("--xy_min",  type=int, default=DEF_MIN_NXY)
    parser.add_argument("--xy_max",  type=int, default=DEF_MAX_NXY)
    parser.add_argument("--xy_step", type=int, default=DEF_NXY_STEP)
    parser.add_argument("--timestep", "-T", type=int, default=DEF_T)
    parser.add_argument("--samples",  "-N", type=int, default=DEF_N_SAMPLES)
    parser.add_argument("--epochs",   "-E", type=int, default=DEF_N_EPOCHS)
    parser.add_argument("--batch",    "-B", type=int, default=DEF_BATCH_SIZE)
    parser.add_argument("--out",              default="results/resolution")
    parser.add_argument("--no-train",   action="store_true", help="skip training/visualisation steps (speed plot only)")
    parser.add_argument("--stochastic", action="store_true", help="benchmark stochastic inference")
    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    argparse_main()