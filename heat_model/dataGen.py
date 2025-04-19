import numpy as np
import torch
import os
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import os

os.chdir(os.path.dirname(__file__))

# === Initial condition patterns ===

def random_gaussian_sum(nx, ny, num_blobs=3):
    x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = np.zeros((nx, ny))
    for _ in range(np.random.randint(1, num_blobs + 1)):
        xc, yc = np.random.rand(2) * 0.8 + 0.1
        sigma = np.random.rand() * 0.05 + 0.02
        amp = np.random.rand() * 2.5
        u0 += amp * np.exp(-((X - xc)**2 + (Y - yc)**2) / (2 * sigma**2))
    return u0

def checkerboard_pattern(nx, ny, num_waves=4):
    x, y = np.linspace(0, 1, nx), np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    u0 = np.zeros((nx, ny))
    for _ in range(np.random.randint(1, num_waves + 1)):
        fx, fy = np.random.randint(1, 10, size=2)
        phase = np.random.rand() * 2 * np.pi
        amp = np.random.rand() * 2.5
        u0 += amp * np.sin(2 * np.pi * fx * X + phase) * np.sin(2 * np.pi * fy * Y + phase)
    return u0

def spotty_noise(nx, ny, num_spots=100, smooth_sigma=1.5):
    u0 = np.zeros((nx, ny))
    for _ in range(num_spots):
        i = np.random.randint(0, nx)
        j = np.random.randint(0, ny)
        u0[i, j] = np.random.rand() * 2.5
    return gaussian_filter(u0, sigma=smooth_sigma)

def generate_initial_condition(nx, ny, mode='mixed'):
    if mode == 'gaussian':
        return random_gaussian_sum(nx, ny)
    elif mode == 'checkerboard':
        return checkerboard_pattern(nx, ny)
    elif mode == 'spotty':
        return spotty_noise(nx, ny)
    elif mode == 'mixed':
        u_gaussian = random_gaussian_sum(nx, ny)
        u_checkerboard = checkerboard_pattern(nx, ny)
        u_spotty = spotty_noise(nx, ny)
        return (u_gaussian + u_checkerboard + u_spotty) / 3
    elif mode == 'random':
        pattern = np.random.choice(['gaussian', 'checkerboard', 'spotty']) 
        return generate_initial_condition(nx, ny, mode=pattern)
    else:
        raise ValueError(f"Invalid mode: {mode}")

# === Generate a trajectory of solutions ===

def generate_trajectory(nx, ny, dx, dy, dt, alpha, nt, n_frames, mode='mixed'):
    u = generate_initial_condition(nx, ny, mode=mode)
    traj = [u.copy()]
    steps_per_frame = nt // (n_frames - 1)
    # add a small random constant to simulate some heat already in the system
    # u += np.random.rand(nx, ny) + 1
    
    for _ in range(1, n_frames):
        for _ in range(steps_per_frame):
            lap = (
                (np.roll(u, 1, axis=0) - 2*u + np.roll(u, -1, axis=0)) / dx**2 +
                (np.roll(u, 1, axis=1) - 2*u + np.roll(u, -1, axis=1)) / dy**2
            )
            # Dirichlet boundary conditions
            # u[0, :] = u[-1, :] = u[:, 0] = u[:, -1] = 0.0
            # u[0, :len(u)//2] = 0
            # u[0, len(u)//2 :] = 1.5
            # u[-1, :] = 1.5
            # u[:, 0] = 0.0
            # u[:, -1] = np.max(u)

            # Neumann boundary conditions
            u[0, :] = u[1, :]
            u[-1, :] = u[-2, :]
            u[:, 0] = u[:, 1]
            u[:, -1] = u[:, -2]
            u += alpha * dt * lap
            

        traj.append(u.copy())

    return traj[0], np.stack(traj)  # shape: (nx, ny), (T, nx, ny)

# === Generate and save full dataset ===

def save_dataset(u0_tensor, uT_tensor, save_dir='heat_trajectory_data'):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(u0_tensor, os.path.join(save_dir, 'u0.pt'))
    torch.save(uT_tensor, os.path.join(save_dir, 'uT.pt'))
    print(f"Saved tensors to '{save_dir}/u0.pt' and 'uT.pt'")

# === Parameters ===

N = 1000
nx, ny = 64, 64
dx = 1.0 / (nx - 1)
dy = 1.0 / (ny - 1)
dt = 0.01
nt = 1000
T = 20

alpha = 0.001

u0_all, traj_all = [], []
for _ in tqdm(range(N), desc="Generating dataset"):
    u0, traj = generate_trajectory(nx, ny, dx, dy, dt, alpha, nt, T, mode='mixed')
    u0_all.append(u0)
    traj_all.append(traj)

u0_tensor = torch.tensor(np.stack(u0_all)[:, None, :, :], dtype=torch.float32)         # (N, 1, nx, ny)
uT_tensor = torch.tensor(np.stack(traj_all), dtype=torch.float32)                     # (N, T, nx, ny)

save_dataset(u0_tensor, uT_tensor)
