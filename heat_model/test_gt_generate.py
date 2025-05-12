import torch
from benchmark_heat import gen_dataset, build_loaders, save_frames_and_gif


T = 250
xy = 160
dx = dy = 1.0 / (xy - 1)
batch = 1
DEF_DT = 0.01
DEF_ALPHA = 1e-3
DEF_T_INTERVAL = 500
samples = 100

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

u0_tensor, uT_tensor, t_gen = gen_dataset(T=T, nx=xy, ny=xy, 
                                          dx=dx, dy=dy, dt=DEF_DT, 
                                          alpha=DEF_ALPHA, nt=DEF_T_INTERVAL, 
                                          n_samples=samples, device=device)
train_dl, val_dl, test_dl = build_loaders(u0_tensor, uT_tensor, batch_size=batch, train=0, val=0)


# Create a gif of ground truth for one sample
u0_batch, uT_batch = next(iter(test_dl))

true_traj = uT_batch[0]      # shape: [T=250, H=8, W=8]
pred_traj = uT_batch[0]      # or your model’s output

save_frames_and_gif(
    T,
    traj_true=true_traj, 
    traj_pred=pred_traj
)