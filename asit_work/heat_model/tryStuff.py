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
     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     device = torch.device('mps' if torch.mps.is_available() else 'cpu')
     model = model.to(torch.float32)
     model.to(device, dtype=torch.float32)

     # #debug
     # # Dummy input [batch_size, channels, height, width]
     # x = torch.randn(1, 1, 64, 64, device=device, dtype=torch.float32)

     # model.eval()
     # with torch.no_grad():
     #      y = model(x)
     #      print("Output stats:", y.mean().item(), y.std().item(), y.min().item(), y.max().item())
     # quit()
     # #debug

     optimizer = AdamW(model.parameters(), lr=1e-3)
     scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
     criterion = torch.nn.MSELoss()

     n_epochs = 100
     for epoch in tqdm(range(1, n_epochs + 1), desc = "Training"):
          model.train()
          train_loss = 0.0
          with tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False, position=1) as pbar:
               for batch in pbar:
                    # debug
                    # print(f"batch: {batch}")
                    # print(f"len(batch): {len(batch)}")
                    # print(f"batch[0].shape: {batch[0].shape}")
                    # print(f"batch[1].shape: {batch[1].shape}")
                    # print(f"batch[0]: {batch[0]}")
                    # print(f"batch[1]: {batch[1]}")
                    # quit()
                    # debug


                    # x = batch['x'].to(device)
                    # y = batch['y'].to(device)
                    x = batch[0].to(device, dtype=torch.float32)
                    y = batch[1].to(device, dtype=torch.float32)

                    optimizer.zero_grad()
                    y_pred = model(x)

                    # #debug
                    print(f"y_pred.shape: {y_pred.shape}")
                    # print("Pred:", y_pred)
                    # print(y_pred.mean(), y_pred.std(), y_pred.min(), y_pred.max())
                    quit()
                    # #debug

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
               # x = batch['x'].to(device)
               # y = batch['y'].to(device)
               x = batch[0].to(device, dtype=torch.float32)
               y = batch[1].to(device, dtype=torch.float32)
               y_pred = model(x)

               # #debug
               # print("Pred:", y_pred)
               # print("STATS:", y_pred.mean(), y_pred.std(), y_pred.min(), y_pred.max())
               # # quit()
               # #debug

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

               # #debug
               # print("Pred:", y_pred)
               # print("STATS:", y_pred.mean(), y_pred.std(), y_pred.min(), y_pred.max())
               # # quit()
               # #debug

               break

     for i in range(5):
          plot_and_save_trajectory(x, y_true, y_pred, sample_idx=i, save_dir="figures", prefix="epoch_final")


if __name__ == "__main__":
    main()