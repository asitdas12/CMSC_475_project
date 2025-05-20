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







def fokker_planck(): 
    nx, ny = 64, 64          
    dx = dy = 1.0 / (nx - 1)   
    dt = 0.001                 
    D = 0.01                   
    num_steps = 10000            

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    sigma = 0.05
    
    p = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma**2))
    p /= np.sum(p) * dx * dy  

    solution = np.zeros((num_steps, nx, ny), dtype=np.float32)
    solution[0] = p
    
    for t in range(1, num_steps):
        laplacian = (
            np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) +
            np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1) -
            4 * p
        ) / dx**2

        p = p + D * dt * laplacian
        p[p < 0] = 0  
        p /= np.sum(p) * dx * dy  
        solution[t] = p

    tensor_data = torch.tensor(solution)
    torch.save(tensor_data, "./fokker_planck_data/fokker_planck_2d.pt")
    print("Dataset saved to fokker_planck_2d.pt with shape:", tensor_data.shape)


def fp_dirichlet(): 

    nx, ny = 64, 64
    dx = dy = 1.0 / (nx - 1)
    dt = 0.001
    D = 0.01
    num_steps = 1000


    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
   
    sigma = 0.05
    p = np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / (2 * sigma**2))
    p /= np.sum(p) * dx * dy  
   
    solution = np.zeros((num_steps, nx, ny), dtype=np.float32)
    solution[0] = p
    
    for t in range(1, num_steps):
        
        laplacian = np.zeros_like(p)
        laplacian[1:-1, 1:-1] = (
            p[2:, 1:-1] + p[:-2, 1:-1] +
            p[1:-1, 2:] + p[1:-1, :-2] -
            4 * p[1:-1, 1:-1]
        ) / dx**2
        
        p[1:-1, 1:-1] += D * dt * laplacian[1:-1, 1:-1]
    
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0

        p /= np.sum(p) * dx * dy
        solution[t] = p

   
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
    
    x = torch.linspace(-5, 5, steps=nx)
    y = torch.linspace(-5, 5, steps=ny)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    V = 0.5 * (X**2 + Y**2)  
    Fx = -torch.gradient(V, spacing=(dx, dy))[0] 
    Fy = -torch.gradient(V, spacing=(dx, dy))[1]  

    p = torch.exp(-((X - 1)**2 + (Y + 1)**2) / 0.5)
    p /= p.sum()  

    data = torch.zeros((num_steps, nx, ny))
    data[0] = p

    for t in range(1, num_steps):
        lap_p = (
            -4 * p
            + F.pad(p, (0, 0, 1, 0))[0:-1, :]  
            + F.pad(p, (0, 0, 0, 1))[1:, :]    
            + F.pad(p, (1, 0, 0, 0))[:, 0:-1]  
            + F.pad(p, (0, 1, 0, 0))[:, 1:]    
        ) / (dx * dy)

        
        div_Fp_x = (
            F.pad(Fx * p, (0, 0, 1, 0))[0:-1, :] - F.pad(Fx * p, (0, 0, 0, 1))[1:, :]
        ) / (2 * dx)
        div_Fp_y = (
            F.pad(Fy * p, (1, 0, 0, 0))[:, 0:-1] - F.pad(Fy * p, (0, 1, 0, 0))[:, 1:]
        ) / (2 * dy)

        div_Fp = div_Fp_x + div_Fp_y

        
        dpdt = D * lap_p - mu * div_Fp
        p = p + dt * dpdt

        
        p = torch.clamp(p, min=0)
        p /= p.sum()

        data[t] = p
   
    torch.save(data, "./fokker_planck_data/fokker_planck_autonomous.pt")
    print("Dataset saved as fokker_planck_autonomous.pt")


def fokker_planck_sine(): 
    nx, ny = 32, 32         
    dx = dy = 1.0 / (nx - 1)  
    dt = 0.001                 
    D = 0.01                   
    num_steps = 10000            

    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    sigma = 0.05
    p = 1 + 0.5 * np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y)
    p /= np.sum(p) * dx * dy  

    solution = np.zeros((num_steps, nx, ny), dtype=np.float32)
    solution[0] = p
    
    for t in range(1, num_steps):
        laplacian = (
            np.roll(p, 1, axis=0) + np.roll(p, -1, axis=0) +
            np.roll(p, 1, axis=1) + np.roll(p, -1, axis=1) -
            4 * p
        ) / dx**2

        p = p + D * dt * laplacian
        p[p < 0] = 0 
        p /= np.sum(p) * dx * dy  
        solution[t] = p

    tensor_data = torch.tensor(solution)
    torch.save(tensor_data, "./fokker_planck_data/fokker_planck_sine.pt")
    print("Dataset saved to fokker_planck_sine.pt with shape:", tensor_data.shape)


def generate_heat_graph():
    import networkx as nx
    nn_x = nn_y = 16
    
    G = nx.grid_2d_graph(nn_x, nn_y)
    n_nodes = G.number_of_nodes()
    
    mapping = {node: i for i, node in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, mapping)
   
    A = nx.adjacency_matrix(G).todense()
    A = torch.tensor(A, dtype=torch.float32)
    D = torch.diag(A.sum(dim=1))
    L = D - A  

    u = torch.zeros(n_nodes)
    center_node = n_nodes // 2
    u[center_node] = 1.0

    # x = torch.arange(nn_x).float()
    # y = torch.arange(nn_y).float()
    # X, Y = torch.meshgrid(x, y, indexing='ij')
    # cx, cy = nn_x // 2, nn_y // 2
    # dist = torch.sqrt((X - cx)**2 + (Y - cy)**2)
    # ring = ((dist >= radius - 0.5) & (dist <= radius + 0.5)).float()
    # u = ring.flatten()
    
    dt = 0.01
    num_steps = 1000
    data = torch.zeros((num_steps, n_nodes))
    data[0] = u
    
    for t in range(1, num_steps):
        du_dt = -L @ u
        u = u + dt * du_dt
        data[t] = u

    data = data.view(num_steps, nn_x, nn_y)

    torch.save(data, "./heat_data/heat_graph.pt")
    print("Dataset saved to heat_graph.pt with shape:", data.shape)


def heston_joint_density(): 
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

    S = np.zeros((num_paths, num_steps))
    v = np.zeros((num_paths, num_steps))
    S[:, 0] = S0
    v[:, 0] = v0

    Z1 = np.random.randn(num_paths, num_steps - 1)
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.randn(num_paths, num_steps - 1)

    for t in range(1, num_steps):
        vt = v[:, t - 1]
        vt = np.clip(vt, 0, None)  # Ensure non-negative variance
        S[:, t] = S[:, t - 1] + mu * S[:, t - 1] * dt + np.sqrt(vt * dt) * S[:, t - 1] * Z1[:, t - 1]
        v[:, t] = v[:, t - 1] + kappa * (theta - v[:, t - 1]) * dt + sigma * np.sqrt(vt * dt) * Z2[:, t - 1]

    xmin, xmax = 50, 200
    vmin, vmax = 0, 0.2
    nbins = 100
    heatmaps = np.zeros((num_steps, nbins, nbins), dtype=np.float32)
    edges_x = np.linspace(xmin, xmax, nbins + 1)
    edges_y = np.linspace(vmin, vmax, nbins + 1)

    for t in range(num_steps):
        hist, _, _ = np.histogram2d(S[:, t], v[:, t], bins=[edges_x, edges_y], density=True)
        heatmaps[t] = hist

    tensor_data = torch.tensor(heatmaps)
    torch.save(tensor_data, "./heston_data/heston_joint_density.pt")
    print("Saved to heston_joint_density.pt with shape:", tensor_data.shape)


def solve_navier_stokes_2d(N=64, T=10.0, dt=0.01, ν=0.001, save_path="./navier_stokes_data/navier_stokes_2d.pt"):
    import numpy as np
    import torch
    from scipy.fftpack import fft2, ifft2, fftfreq
    import os
    
    L = 2 * np.pi
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u = np.sin(X) * np.cos(Y)
    v = -np.cos(X) * np.sin(Y)
    
    kx = fftfreq(N, 1.0 / N) * 2 * np.pi / L
    ky = fftfreq(N, 1.0 / N) * 2 * np.pi / L
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    ksq = kx**2 + ky**2
    ksq[0, 0] = 1  
    
    u_hat = fft2(u)
    v_hat = fft2(v)

    steps = int(T / dt)
    results = []

    def dealias(f_hat):
        cutoff = N // 3
        f_hat[cutoff:-cutoff, :] = 0
        f_hat[:, cutoff:-cutoff] = 0
        return f_hat

    for step in range(steps):
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

        nonlinear_u_hat = dealias(nonlinear_u_hat)
        nonlinear_v_hat = dealias(nonlinear_v_hat)

        div_hat = 1j * kx * nonlinear_u_hat + 1j * ky * nonlinear_v_hat
        nonlinear_u_hat -= kx * div_hat / ksq
        nonlinear_v_hat -= ky * div_hat / ksq

        u_hat -= dt * (nonlinear_u_hat + ν * ksq * u_hat)
        v_hat -= dt * (nonlinear_v_hat + ν * ksq * v_hat)

        if step % 10 == 0:
            u_real = np.real(ifft2(u_hat))
            v_real = np.real(ifft2(v_hat))
            results.append(np.stack([u_real, v_real], axis=0))  

    data = np.stack(results).astype(np.float32)
    tensor_data = torch.from_numpy(data)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(tensor_data, save_path)
    print(f"Dataset saved to {save_path} with shape:", tensor_data.shape)



def spectral_navier_stokes(): 
    import numpy as np
    import torch
    import os
    from numpy.fft import fft2, ifft2, fftfreq
    from tqdm import tqdm

    os.chdir(os.path.dirname(__file__))

    def taylor_green(nx, ny, Lx=2*np.pi, Ly=2*np.pi):
        x = np.linspace(0, Lx, nx, endpoint=False)
        y = np.linspace(0, Ly, ny, endpoint=False)
        X, Y = np.meshgrid(x, y, indexing='ij')
        u =  np.sin(X) * np.cos(Y)
        v = -np.cos(X) * np.sin(Y)
        return u, v

    def shear_layer(nx, ny):
        y = np.linspace(0, 2*np.pi, ny, endpoint=False)
        u = np.tanh((y - np.pi) * 20)  
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

    #

    def solve_navier_stokes(u, v, ν, dt, steps, save_every):
        nx, ny = u.shape
        kx = fftfreq(nx, 1.0 / nx)
        ky = fftfreq(ny, 1.0 / ny)
        kx, ky = np.meshgrid(kx, ky, indexing='ij')
        ksq = kx**2 + ky**2
        ksq[0, 0] = 1  

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
                traj.append(np.stack([u_real, v_real], axis=0))  

        return np.stack(traj)  

    

    def save_dataset(u0_tensor, uT_tensor, save_dir='navier_stokes_data'):
        os.makedirs(save_dir, exist_ok=True)
        torch.save(u0_tensor, os.path.join(save_dir, 'spectral_navier_stokes_u0.pt'))
        torch.save(uT_tensor, os.path.join(save_dir, 'spectral_navier_stokes_uT.pt'))
        print(f"Saved to '{save_dir}/spectral_navier_stokes_u0.pt' and 'spectral_navier_stokes_uT.pt'")

    

    N = 1000          
    nx = ny = 64       
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

    u0_tensor = torch.tensor(np.stack(u0_all), dtype=torch.float32)        
    uT_tensor = torch.tensor(np.stack(traj_all), dtype=torch.float32)      

    save_dataset(u0_tensor, uT_tensor)


# === Parameters ===
if __name__ == "__main__":

    fokker_planck()
    fp_dirichlet()
    generate_fokker_planck_autonomous()
    fokker_planck_sine()

    generate_heat_graph()
    heston_joint_density()
    # solve_navier_stokes_2d()
    # spectral_navier_stokes()