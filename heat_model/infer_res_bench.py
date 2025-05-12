# This file runs after we train a model
# and want to see how well it generalizes
# to data with different dimensions and initial conditions

import argparse, shutil
from pathlib import Path
from datetime import datetime

import torch
import pandas as pd
import matplotlib.pyplot as plt

from neuralop.models import FNO

from benchmark_heat import torch_now, gen_dataset, inference_loop, build_loaders, timing_plot

# -------------------------- default hyper-parameters for datagen inferencing  --------------------------
DEF_MIN_NXY, DEF_MAX_NXY, DEF_NXY_STEP = 16, 64, 8 # spatial resolution bounds
DEF_T           = 250                    # T is how many frames for gt & inference
DEF_N_SAMPLES   = 500                   # trajectories per T (keep small => quick)
DEF_N_EPOCHS    = 50                   # FNO training epochs
DEF_BATCH_SIZE  = 1
DEF_ALPHA       = 1e-3                  # lr
DEF_DT          = 0.01                  # physical timestep
DEF_T_INTERVAL  = 500                    # total timesteps per frame
HIDDEN_CHANNELS = 64
# -----------------------------------------------------------------------------

# -------------------------- parameters for loading model(s) --------------------------
# Select Model(s)
DEF_MIN_RES_MODEL, DEF_MAX_RES_MODEL, DEF_RES_STEP = 16, 64, 8 # spatial resolution bounds

# ===== main ==================================================================

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device};  started {datetime.now().isoformat(timespec='seconds')}")

    out_root = Path(args.out);  out_root.mkdir(exist_ok=True)
    models_dir = out_root / "models";   models_dir.mkdir(exist_ok=True)
    
    out_base = out_root / "test"

    T = args.timestep

    results = {}
    # Generating Dataset of multiple res to test on models with differing res (testing resolution invariance)
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
        train_dl, val_dl, test_dl = build_loaders(u0_tensor, uT_tensor, batch_size=args.batch, train=0, val=0) # load entire dataset as test
        
        model = None

        # ---------- loading model for inference ------------
        # Test Inference on different resolutions datasets for each model
        for res_model in range(DEF_MIN_RES_MODEL, DEF_MAX_RES_MODEL + 1, DEF_RES_STEP):
            t_dir = out_base / f"model_res_{res_model}" / f"channels_{HIDDEN_CHANNELS}" / f"T_{DEF_T}_data_res_{xy}"

            model = FNO(n_modes=(res_model, res_model), 
                        hidden_channels=HIDDEN_CHANNELS, 
                        in_channels=1, out_channels=DEF_T).to(device)
            
            ckpt_path = models_dir / f"fno_T_{DEF_T}_resolution_{res_model}_channels_{HIDDEN_CHANNELS}.pth"
            model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=False))
            print(f"Loaded model -> {ckpt_path}")

            if t_dir.exists():
                shutil.rmtree(t_dir)
            t_dir.mkdir(parents=True)
    
            # ---- trajectory animation ----
            t_inf = inference_loop(model, test_dl, t_dir, T, device)
            
            avg_t_data_gen = sum(t_gen)/len(t_gen)
            avg_t_inf = sum(t_inf)/len(t_inf)

            # Check if model is already in results dict
            if res_model not in results:
                results[res_model] = []
            
            # add to dataframe
            results[res_model].append(dict(xy=xy, dataset_sec=avg_t_data_gen, inference_sec=avg_t_inf))

            # free up GPU memory
            del model

        # free up GPU memory
        del train_dl, val_dl, test_dl, u0_tensor, uT_tensor
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect() # defragment the cache

    # ===== global timing plot csv =====
    # each models will have their own csv
    for res_model, rows in results.items():
        t_dir = out_base / f"model_res_{res_model}" / f"channels_{HIDDEN_CHANNELS}"
        df = pd.DataFrame(rows)
        df.to_csv(t_dir / f"dataset_timings_xy_{args.xy_min}_{args.xy_max}_{args.xy_step}.csv", index=False)
        
        timing_plot(df=df, samples=args.samples, dir=t_dir, res=res_model)

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
    args = parser.parse_args()

    main(args)

if __name__ == "__main__":
    argparse_main()