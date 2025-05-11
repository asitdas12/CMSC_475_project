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
    torch.save(u0_tensor, os.path.join(save_dir, f'u0{u0_tensor.shape}.pt'))
    torch.save(uT_tensor, os.path.join(save_dir, f'uT{uT_tensor.shape}.pt'))
    print(f"Saved tensors to '{save_dir}/u0{u0_tensor.shape}.pt' and 'uT{uT_tensor.shape}.pt'")





# asit_work data

def fokker_planck(): 
    # Parameters
    nx, ny = 64, 64          # Grid resolution
    dx = dy = 1.0 / (nx - 1)   # Grid spacing
    dt = 0.001                 # Time step
    D = 0.01                   # Diffusion coefficient
    num_steps = 1000            # Total time steps

    # Create coordinate grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial condition: 2D Gaussian blob
    sigma = 0.05
    p = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma**2))
    p /= np.sum(p) * dx * dy  # Normalize to integrate to 1

    # Store solution snapshots
    solution = np.zeros((num_steps, nx, ny), dtype=np.float32)
    solution[0] = p

    # Fokker-Planck evolution (pure diffusion using finite differences)
    for t in range(1, num_steps):
        laplacian = (
            np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) +
            np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1) -
            4 * p
        ) / dx**2

        p = p + D * dt * laplacian
        p[p < 0] = 0  # avoid numerical negatives
        p /= np.sum(p) * dx * dy  # renormalize
        solution[t] = p

    # Convert to PyTorch tensor and save
    tensor_data = torch.tensor(solution)
    torch.save(tensor_data, "./fokker_planck_data/fokker_planck_2d.pt")
    print("Dataset saved to fokker_planck_2d.pt with shape:", tensor_data.shape)


def fp_dirichlet(): 
    # Parameters
    nx, ny = 64, 64
    dx = dy = 1.0 / (nx - 1)
    dt = 0.001
    D = 0.01
    num_steps = 1000

    # Create coordinate grid
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Initial condition: 2D Gaussian blob
    sigma = 0.05
    p = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma**2))
    p /= np.sum(p) * dx * dy  # Normalize

    # Store snapshots
    solution = np.zeros((num_steps, nx, ny), dtype=np.float32)
    solution[0] = p

    # Evolution loop with Dirichlet boundaries (zero at edges)
    for t in range(1, num_steps):
        # Compute Laplacian only for interior points
        laplacian = np.zeros_like(p)
        laplacian[1:-1, 1:-1] = (
            p[2:, 1:-1] + p[:-2, 1:-1] +
            p[1:-1, 2:] + p[1:-1, :-2] -
            4 * p[1:-1, 1:-1]
        ) / dx**2

        # Update interior
        p[1:-1, 1:-1] += D * dt * laplacian[1:-1, 1:-1]

        # Enforce Dirichlet BCs: boundary = 0
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0

        # Normalize
        p /= np.sum(p) * dx * dy
        solution[t] = p

    # Save tensor
    tensor_data = torch.tensor(solution)
    torch.save(tensor_data, "./fokker_planck_data/fokker_planck_dirichlet.pt")
    print("Dataset saved to fokker_planck_dirichlet.pt with shape:", tensor_data.shape)


def generate_fokker_planck_autonomous(): 
    import torch.nn.functional as F
    
    nx=64
    ny=64
    dx=1.0 / (nx - 1)
    dy=1.0 / (nx - 1)
    D=0.01
    mu=1.0
    dt=0.001
    num_steps=1000
    
    # Create 2D grid
    x = torch.linspace(-5, 5, steps=nx)
    y = torch.linspace(-5, 5, steps=ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    # Define static potential V(x, y), e.g. double-well or harmonic
    V = 0.5 * (X**2 + Y**2)  # harmonic potential
    Fx = -torch.gradient(V, spacing=(dx, dy))[0]  # -dV/dx
    Fy = -torch.gradient(V, spacing=(dx, dy))[1]  # -dV/dy

    # Initialize p(x, y, 0): a Gaussian blob
    p = torch.exp(-((X - 1)**2 + (Y + 1)**2) / 0.5)
    p /= p.sum()  # Normalize

    # Storage for time evolution
    data = torch.zeros((num_steps, nx, ny))
    data[0] = p

    for t in range(1, num_steps):
        # Compute Laplacian (diffusion)
        lap_p = (
            -4 * p
            + F.pad(p, (0, 0, 1, 0))[0:-1, :]  # up
            + F.pad(p, (0, 0, 0, 1))[1:, :]    # down
            + F.pad(p, (1, 0, 0, 0))[:, 0:-1]  # left
            + F.pad(p, (0, 1, 0, 0))[:, 1:]    # right
        ) / (dx * dy)

        # Compute drift: ∇·(F p)
        div_Fp_x = (
            F.pad(Fx * p, (0, 0, 1, 0))[0:-1, :] - F.pad(Fx * p, (0, 0, 0, 1))[1:, :]
        ) / (2 * dx)
        div_Fp_y = (
            F.pad(Fy * p, (1, 0, 0, 0))[:, 0:-1] - F.pad(Fy * p, (0, 1, 0, 0))[:, 1:]
        ) / (2 * dy)

        div_Fp = div_Fp_x + div_Fp_y

        # Update rule: p_{t+1} = p_t + dt (D * ∇²p - ∇·(Fp))
        dpdt = D * lap_p - mu * div_Fp
        p = p + dt * dpdt

        # Clamp to non-negative and renormalize
        p = torch.clamp(p, min=0)
        p /= p.sum()

        data[t] = p

    # return data

    # save
    torch.save(data, "./fokker_planck_data/fokker_planck_autonomous.pt")
    print("Dataset saved as fokker_planck_autonomous.pt")


def generate_heat_graph():

    import networkx as nx
    
    nn_x = nn_y = 16

    # Create a 2D grid graph (10x10)
    G = nx.grid_2d_graph(nn_x, nn_y)
    n_nodes = G.number_of_nodes()
    
    # Map 2D grid node labels to integer labels
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
    
    # Adjacency and Laplacian
    A = nx.adjacency_matrix(G).todense()
    A = torch.tensor(A, dtype=torch.float32)
    D = torch.diag(A.sum(dim=1))
    L = D - A  # Unnormalized Laplacian

    # Initial heat distribution (hot at center node)
    u = torch.zeros(n_nodes)
    center_node = n_nodes // 2
    u[center_node] = 1.0

    # Time parameters
    dt = 0.01
    num_steps = 1000
    data = torch.zeros((num_steps, n_nodes))
    data[0] = u

    # Time evolution
    for t in range(1, num_steps):
        du_dt = -L @ u
        u = u + dt * du_dt
        data[t] = u

    data = data.view(num_steps, nn_x, nn_y)

    # Save tensor
    torch.save(data, "./heat_data/heat_graph.pt")
    print("Dataset saved to heat_graph.pt with shape:", data.shape)


def heston_joint_density(): 
    # Parameters
    num_paths = 10000
    num_steps = 1000
    T = 1.0
    dt = T / num_steps
    mu = 0.05
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7

    S0 = 100.0
    v0 = 0.04

    # Preallocate
    S = np.zeros((num_paths, num_steps))
    v = np.zeros((num_paths, num_steps))
    S[:, 0] = S0
    v[:, 0] = v0

    # Correlated Brownian motions
    Z1 = np.random.randn(num_paths, num_steps - 1)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(num_paths, num_steps - 1)

    for t in range(1, num_steps):
        vt = v[:, t - 1]
        vt = np.clip(vt, 0, None)  # Ensure non-negative variance
        S[:, t] = S[:, t - 1] + mu * S[:, t - 1] * dt + np.sqrt(vt * dt) * S[:, t - 1] * Z1[:, t - 1]
        v[:, t] = v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma * np.sqrt(vt * dt) * Z2[:, t - 1]

    # Build 2D joint histogram (heatmap) over time
    xmin, xmax = 50, 200
    vmin, vmax = 0, 0.2
    nbins = 100
    heatmaps = np.zeros((num_steps, nbins, nbins), dtype=np.float32)
    edges_x = np.linspace(xmin, xmax, nbins + 1)
    edges_y = np.linspace(vmin, vmax, nbins + 1)

    for t in range(num_steps):
        hist, _, _ = np.histogram2d(S[:, t], v[:, t], bins=[edges_x, edges_y], density=True)
        heatmaps[t] = hist

    # Convert to torch tensor and save
    tensor_data = torch.tensor(heatmaps)
    torch.save(tensor_data, "./heston_data/heston_joint_density.pt")
    print("Saved to heston_joint_density.pt with shape:", tensor_data.shape)


def solve_navier_stokes_2d(N=64, T=10.0, dt=0.01, ν=0.001):
    import numpy as np
    from scipy.fftpack import fft2, ifft2, fftfreq
    import os
    os.chdir(os.path.dirname(__file__))
    
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y)

    # Initial condition: Taylor-Green vortex
    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)

    # Fourier wave numbers
    kx = fftfreq(N, 1.0 / N)
    ky = fftfreq(N, 1.0 / N)
    kx, ky = np.meshgrid(kx, ky)
    ksq = kx**2 + ky**2
    ksq[0, 0] = 1  # avoid division by zero

    # Initialize velocity fields in Fourier space
    u_hat = fft2(u)
    v_hat = fft2(v)

    steps = int(T / dt)
    results = []

    for _ in range(steps):
        u = np.real(ifft2(u_hat))
        v = np.real(ifft2(v_hat))

        u_x = np.real(ifft2(1j * kx * u_hat))
        u_y = np.real(ifft2(1j * ky * u_hat))
        v_x = np.real(ifft2(1j * kx * v_hat))
        v_y = np.real(ifft2(1j * ky * v_hat))

        nonlinear_u = u * u_x + v * u_y
        nonlinear_v = u * v_x + v * v_y

        nonlinear_u_hat = fft2(nonlinear_u)
        nonlinear_v_hat = fft2(nonlinear_v)

        div_hat = 1j * kx * nonlinear_u_hat + 1j * ky * nonlinear_v_hat
        pressure_hat = div_hat / ksq

        u_hat -= dt * (nonlinear_u_hat - 1j * kx * pressure_hat + ν * ksq * u_hat)
        v_hat -= dt * (nonlinear_v_hat - 1j * ky * pressure_hat + ν * ksq * v_hat)

        if _ % 10 == 0:
            results.append((np.real(ifft2(u_hat)), np.real(ifft2(v_hat))))

    # return np.stack(results)  # shape: [T, N, N, 2]

    data = np.stack(results)
    data = data.astype(np.float32)
    # np.save("navier_stokes_data.npy", data)

    # Convert to PyTorch tensor and save
    tensor_data = torch.from_numpy(data)
    torch.save(tensor_data, "./navier_stokes_data/navier_stokes_2d.pt")
    print("Dataset saved to navier_stokes_2d.pt with shape:", tensor_data.shape)


def spectral_navier_stokes(): 
    import numpy as np
    import torch
    import os
    from numpy.fft import fft2, ifft2, fftfreq
    from tqdm import tqdm

    os.chdir(os.path.dirname(__file__))

    # === Initial Condition Generators ===

    def taylor_green(nx, ny, Lx=2*np.pi, Ly=2*np.pi):
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        u =  np.sin(X) * np.cos(Y)
        v = -np.cos(X) * np.sin(Y)
        return u, v

    def shear_layer(nx, ny):
        y = np.linspace(0, 2*np.pi, ny, endpoint=False)
        u = np.tanh((y - np.pi) * 20)  # sharp transition
        u = np.tile(u, (nx, 1))
        v = 0.05 * np.random.randn(nx, ny)
        return u, v

    def random_vortex(nx, ny, num_vortices=5):
        x = np.linspace(0, 2*np.pi, nx, endpoint=False)
        y = np.linspace(0, 2*np.pi, ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        u = np.zeros_like(X)
        v = np.zeros_like(Y)
        for _ in range(num_vortices):
            cx, cy = np.random.rand(2) * 2 * np.pi
            strength = np.random.randn()
            r2 = (X - cx)**2 + (Y - cy)**2
            u += -strength * (Y - cy) * np.exp(-r2)
            v +=  strength * (X - cx) * np.exp(-r2)
        return u, v

    def random_initial_velocity(nx, ny):
        choice = np.random.choice(['tg', 'shear', 'vortex'])
        if choice == 'tg':
            return taylor_green(nx, ny)
        elif choice == 'shear':
            return shear_layer(nx, ny)
        else:
            return random_vortex(nx, ny)

    # === Spectral Navier–Stokes Solver ===

    def solve_navier_stokes(u, v, ν, dt, steps, save_every):
        nx, ny = u.shape
        kx = fftfreq(nx, 1.0 / nx)
        ky = fftfreq(ny, 1.0 / ny)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        ksq = kx**2 + ky**2
        ksq[0, 0] = 1  # avoid divide by zero

        u_hat = fft2(u)
        v_hat = fft2(v)

        traj = []

        for step in range(steps):
            u = np.real(ifft2(u_hat))
            v = np.real(ifft2(v_hat))

            u_x = np.real(ifft2(1j * kx * u_hat))
            u_y = np.real(ifft2(1j * ky * u_hat))
            v_x = np.real(ifft2(1j * kx * v_hat))
            v_y = np.real(ifft2(1j * ky * v_hat))

            nonlin_u = u * u_x + v * u_y
            nonlin_v = u * v_x + v * v_y

            nonlin_u_hat = fft2(nonlin_u)
            nonlin_v_hat = fft2(nonlin_v)

            div_hat = 1j * kx * nonlin_u_hat + 1j * ky * nonlin_v_hat
            pressure_hat = div_hat / ksq

            u_hat -= dt * (nonlin_u_hat - 1j * kx * pressure_hat + ν * ksq * u_hat)
            v_hat -= dt * (nonlin_v_hat - 1j * ky * pressure_hat + ν * ksq * v_hat)

            if step % save_every == 0:
                u_real = np.real(ifft2(u_hat))
                v_real = np.real(ifft2(v_hat))
                traj.append(np.stack([u_real, v_real], axis=0))  # shape: [2, nx, ny]

        return np.stack(traj)  # shape: [T, 2, nx, ny]

    # === Save Tensor Dataset ===

    def save_dataset(u0_tensor, uT_tensor, save_dir='navier_stokes_data'):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(u0_tensor, os.path.join(save_dir, 'spectral_navier_stokes_u0.pt'))
        torch.save(uT_tensor, os.path.join(save_dir, 'spectral_navier_stokes_uT.pt'))
        print(f"✅ Saved to '{save_dir}/spectral_navier_stokes_u0.pt' and 'spectral_navier_stokes_uT.pt'")

    # === Parameters ===

    N = 1000           # Number of samples
    nx = ny = 64       # Grid size
    dt = 0.01
    ν = 0.001
    T_total = 10.0
    steps = int(T_total / dt)
    T_frames = 100
    save_every = steps // (T_frames - 1)

    u0_all, traj_all = [], []
    for _ in tqdm(range(N), desc="Generating Navier–Stokes trajectories"):
        u, v = random_initial_velocity(nx, ny)
        traj = solve_navier_stokes(u, v, ν, dt, steps, save_every)
        u0_all.append(traj[0])
        traj_all.append(traj)

    u0_tensor = torch.tensor(np.stack(u0_all), dtype=torch.float32)         # (N, 2, nx, ny)
    uT_tensor = torch.tensor(np.stack(traj_all), dtype=torch.float32)       # (N, T, 2, nx, ny)

    save_dataset(u0_tensor, uT_tensor)


# === Parameters ===
if __name__ == "__main__":
    # N = 1000
    # nx, ny = 64, 64
    # dx = 1.0 / (nx - 1)
    # dy = 1.0 / (ny - 1)
    # dt = 0.01
    # nt = 1000
    # T = 20

    # alpha = 0.001

    # u0_all, traj_all = [], []
    # for _ in tqdm(range(N), desc="Generating dataset"):
    #     u0, traj = generate_trajectory(nx, ny, dx, dy, dt, alpha, nt, T, mode='mixed')
    #     u0_all.append(u0)
    #     traj_all.append(traj)

    # u0_tensor = torch.tensor(np.stack(u0_all)[:, None, :, :], dtype=torch.float32)         # (N, 1, nx, ny)
    # uT_tensor = torch.tensor(np.stack(traj_all), dtype=torch.float32)                     # (N, T, nx, ny)

    # save_dataset(u0_tensor, uT_tensor)

    fokker_planck()
    fp_dirichlet()
    generate_fokker_planck_autonomous()
    generate_heat_graph()
    heston_joint_density()
    solve_navier_stokes_2d()
    spectral_navier_stokes()