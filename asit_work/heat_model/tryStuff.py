import torch
from torch.utils.data import DataLoader
# from neuralop.data.datasets.pt_dataset import TensorDataset
from torch.utils.data import TensorDataset
from neuralop.models import FNO
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
# os.chdir('C:/Users/jmmil/workspace/school/475/project')
os.chdir('./')
# === Load Data ===

def load_dataset(save_dir='heat_trajectory_data'):
    u0 = torch.load(os.path.join(save_dir, 'u0.pt'))
    uT = torch.load(os.path.join(save_dir, 'uT.pt'))
    print(f"Loaded dataset from {save_dir}")
    return u0, uT


def plot_and_save_trajectory(x, y_true, y_pred, sample_idx=0, save_dir="figures", prefix="sample"):
    os.makedirs(save_dir, exist_ok=True)
    T = y_true.shape[1]
    _, axs = plt.subplots(3, T, figsize=(2.5*T, 7))

    for t in range(T):
        axs[0, t].imshow(x[sample_idx, 0].cpu(), cmap='inferno')
        axs[0, t].set_title("Initial" if t == 0 else "")
        axs[1, t].imshow(y_true[sample_idx, t].cpu(), cmap='inferno')
        axs[1, t].set_title(f"True $t_{t}$")
        axs[2, t].imshow(y_pred[sample_idx, t].cpu(), cmap='inferno')
        axs[2, t].set_title(f"Pred $t_{t}$")
        for row in axs[:, t]:
            row.axis('off')

    plt.tight_layout()
    path = os.path.join(save_dir, f"{prefix}_{sample_idx}.png")
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


import matplotlib.pyplot as plt
from matplotlib import animation

def create_animation(y_pred, y_true, sample_idx=0, save_path="animation.gif", save_path2="animationT.gif"):
    T = y_pred.shape[1]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(y_pred[sample_idx, 0].cpu(), cmap="inferno")
    ax.axis('off')

    def update(t):
        im.set_array(y_pred[sample_idx, t].cpu())
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=T, interval=200)
    anim.save(save_path, writer='pillow')  # Use 'ffmpeg' for .mp4
    plt.close()
    print(f"Saved animation to {save_path}")

    T = y_true.shape[1]

    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(y_true[sample_idx, 0].cpu(), cmap="inferno")
    ax.axis('off')

    def update(t):
        im.set_array(y_true[sample_idx, t].cpu())
        return [im]

    anim = animation.FuncAnimation(fig, update, frames=T, interval=200)
    anim.save(save_path2, writer='pillow')  # Use 'ffmpeg' for .mp4
    plt.close()
    print(f"Saved animation to {save_path2}")


def animate_comparison(x, y_true, y_pred, sample_idx=0, save_path="compare.gif"):
    T = y_true.shape[1]
    fig, axs = plt.subplots(3, 1, figsize=(4, 12))
    ims = []

    row_titles = ["Initial Condition", "Ground Truth", "Prediction"]
    for t in range(T):
        ims.append([
            axs[0].imshow(x[sample_idx, 0].cpu(), cmap='inferno', animated=True),
            axs[1].imshow(y_true[sample_idx, t].cpu(), cmap='inferno', animated=True),
            axs[2].imshow(y_pred[sample_idx, t].cpu(), cmap='inferno', animated=True)
        ])
        # for ax in axs:
        #     ax.axis('off')

        for ax, title in zip(axs, row_titles):
            ax.set_title(title, fontsize=12)
            ax.axis('off')

    ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True)
    ani.save(save_path, writer='pillow')
    plt.close()
    print(f"Saved comparison animation: {save_path}")



# === Predict and Save Figures ===
def main():
     u0_tensor, uT_tensor = load_dataset()
     #debug
     u0_tensor = u0_tensor.to(torch.float32)
     uT_tensor = uT_tensor.to(torch.float32)
     #debug
     T = uT_tensor.shape[1]

     train_dataset = TensorDataset(u0_tensor[:800], uT_tensor[:800])
     val_dataset = TensorDataset(u0_tensor[800:900], uT_tensor[800:900])
     test_dataset = TensorDataset(u0_tensor[900:], uT_tensor[900:])

     # === Define and Train Model ===
     train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
     val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
     test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

     model = FNO(n_modes=(20, 20), hidden_channels=64, in_channels=1, out_channels=T).to(torch.float32)
     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     model = model.to(torch.float32)
     model.to(device, dtype=torch.float32)

     optimizer = AdamW(model.parameters(), lr=1e-3)
     scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
     criterion = torch.nn.MSELoss()

     n_epochs = 3
     for epoch in tqdm(range(1, n_epochs + 1), desc = "Training"):
          model.train()
          train_loss = 0.0
          with tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False, position=1) as pbar:
               for batch in pbar:
                    x = batch[0].to(device, dtype=torch.float32)
                    y = batch[1].to(device, dtype=torch.float32)

                    optimizer.zero_grad()
                    y_pred = model(x)

                    loss = criterion(y_pred, y)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())

     scheduler.step()
     print(f"Epoch {epoch} — Avg Train Loss: {train_loss / len(train_loader):.4e}")

     model.eval()
     val_loss = 0.0
     with torch.no_grad(), tqdm(val_loader, desc=f"[Epoch {epoch}] Validating") as pbar:
          for batch in pbar:
               x = batch[0].to(device, dtype=torch.float32)
               y = batch[1].to(device, dtype=torch.float32)
               y_pred = model(x)

               loss = criterion(y_pred, y)
               val_loss += loss.item()
               pbar.set_postfix(val_loss=loss.item())

     print(f"Epoch {epoch} — Avg Val Loss: {val_loss / len(val_loader):.4e}")

# === Visualization ===

     model.eval()
     with torch.no_grad():
          for batch in test_loader:
               # x = batch['x'].to(device)
               # y_true = batch['y'].to(device)
               x = batch[0].to(device, dtype=torch.float32)
               y_true = batch[1].to(device, dtype=torch.float32)
               y_pred = model(x)

               break

     for i in range(5):
          plot_and_save_trajectory(x, y_true, y_pred, sample_idx=i, save_dir="figures", prefix="epoch_final")

     create_animation(y_pred=y_pred, y_true=y_true)

     animate_comparison(x=x, y_true=y_true, y_pred=y_pred)




if __name__ == "__main__":
    main()