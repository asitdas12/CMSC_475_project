import torch
from torch.utils.data import DataLoader
from neuralop.data.datasets.pt_dataset import TensorDataset
from neuralop.models import FNO
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

# asit_work imports
from torch.utils.data import random_split
# import imageio
import matplotlib.animation as animation
from matplotlib import rcParams
import numpy as np

os.chdir(os.path.dirname(__file__))

# === Load Data ===

# def load_dataset(save_dir='heat_trajectory_data'):
#     u0 = torch.load(os.path.join(save_dir, 'u0.pt'))
#     uT = torch.load(os.path.join(save_dir, 'uT.pt'))
#     print(f"Loaded dataset from {save_dir}")
#     return u0, uT

# load_datasets
def load_dataset(mode=""):
     data = None

     if mode == "fokker_planck_2d": 
          data = torch.load("./fokker_planck_data/fokker_planck_2d.pt")
     elif mode == "fokker_planck_dirichlet": 
          data = torch.load("./fokker_planck_data/fokker_planck_dirichlet.pt")
     elif mode == "fokker_planck_autonomous": 
          data = torch.load("./fokker_planck_data/fokker_planck_autonomous.pt")
     elif mode == "fokker_planck_sine": 
          data = torch.load("./fokker_planck_data/fokker_planck_sine.pt")
     elif mode == "heat_graph": 
          data = torch.load("./heat_data/heat_graph.pt")
     elif mode == "heston_joint_density": 
          data = torch.load("./heston_data/heston_joint_density.pt")
     elif mode == "navier_stokes_2d": 
          data = torch.load("./navier_stokes_data/navier_stokes_2d.pt")
     elif mode == "spectral_navier_stokes": 
          u0 = torch.load("./navier_stokes_data/spectral_navier_stokes_u0.pt")
          uT = torch.load("./navier_stokes_data/spectral_navier_stokes_uT.pt")
          data = [u0, uT]

     else: 
          print("need to specify mode")
          quit()

     return data

     


# def plot_and_save_trajectory(x, y_true, y_pred, sample_idx=0, save_dir="figures", prefix="sample"):
#     os.makedirs(save_dir, exist_ok=True)
#     T = y_true.shape[1]
#     _, axs = plt.subplots(3, T, figsize=(2.5*T, 7))

#     for t in range(T):
#         axs[0, t].imshow(x[sample_idx, 0].cpu(), cmap='inferno')
#         axs[0, t].set_title("Initial" if t == 0 else "")
#         axs[1, t].imshow(y_true[sample_idx, t].cpu(), cmap='inferno')
#         axs[1, t].set_title(f"True $t_{t}$")
#         axs[2, t].imshow(y_pred[sample_idx, t].cpu(), cmap='inferno')
#         axs[2, t].set_title(f"Pred $t_{t}$")
#         for row in axs[:, t]:
#             row.axis('off')

#     plt.tight_layout()
#     path = os.path.join(save_dir, f"{prefix}_{sample_idx}.png")
#     plt.savefig(path, dpi=300)
#     plt.close()
#     print(f"Saved: {path}")



# === Predict and Save Figures ===
def main():
     mode = "navier_stokes_2d"
     # u0_tensor, uT_tensor = load_dataset()
     tensor_data = load_dataset(mode)
     # Prepare (input, target) pairs: (t) -> (t+1)
     if mode == "navier_stokes_2d": 
          input_tensor = tensor_data[:-1]
          target_tensor = tensor_data[1:] 
     elif mode == "spectral_navier_stokes": 
          initial_cond = tensor_data[0][0].unsqueeze(0)
          rest_of_data = tensor_data[1][0]
          tensor_data = torch.cat([initial_cond, rest_of_data], dim=0)
          input_tensor = tensor_data[:-1]
          target_tensor = tensor_data[1:]
     else: 
          input_tensor = tensor_data[:-1].unsqueeze(1)     # shape: (99, 1, 100, 100)
          target_tensor = tensor_data[1:].unsqueeze(1)     # shape: (99, 1, 100, 100)

     # T = uT_tensor.shape[1]
     T = target_tensor.shape[1]

     # #debug
     # print(f"input_tensor.shape: {input_tensor.shape}")
     # print(f"target_tensor.shape: {target_tensor.shape}")
     # quit()
     # #debug

     # train_dataset = TensorDataset(u0_tensor[:800], uT_tensor[:800])
     # val_dataset = TensorDataset(u0_tensor[800:900], uT_tensor[800:900])
     # test_dataset = TensorDataset(u0_tensor[900:], uT_tensor[900:])

     # full_dataset = TensorDataset(input_tensor, target_tensor)

     # Split
     num_total = input_tensor.shape[0]
     num_train = int(0.7 * num_total)
     num_val = int(0.15 * num_total)
     num_test = num_total - num_train - num_val

     # #debug
     # print(f"num_total: {num_total}")
     # print(f"num_train: {num_train}")
     # print(f"num_val: {num_val}")
     # print(f"num_test: {num_test}")
     # quit()
     # #debug

     # train_set, val_set, test_set = random_split(full_dataset, [num_train, num_val, num_test])

     train_dataset = TensorDataset(input_tensor[:num_train], target_tensor[:num_train])
     val_dataset = TensorDataset(
          input_tensor[num_train:num_train+num_val], 
          target_tensor[num_train:num_train+num_val]
          )
     test_dataset = TensorDataset(input_tensor[num_train+num_val:], target_tensor[num_train+num_val:])


     # # === Define and Train Model ===
     # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
     # val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)
     # test_loader = DataLoader(test_dataset, batch_size=32, num_workers=0)

     # DataLoaders
     batch_size = 32
     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
     val_loader = DataLoader(val_dataset, batch_size=batch_size)
     test_loader = DataLoader(test_dataset, batch_size=batch_size)

     model = FNO(n_modes=(20, 20), hidden_channels=64, in_channels=T, out_channels=T)
     # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
     device = torch.device('mps' if torch.mps.is_available() else 'cpu')

     model.to(device)

     optimizer = AdamW(model.parameters(), lr=1e-3)
     scheduler = StepLR(optimizer, step_size=25, gamma=0.5)
     criterion = torch.nn.MSELoss()

     if (1==1): # train new model or load existing model
          n_epochs = 50
          for epoch in tqdm(range(1, n_epochs + 1), desc = "Training"):
               model.train()
               train_loss = 0.0
               with tqdm(train_loader, desc=f"[Epoch {epoch}] Training", leave=False, position=1) as pbar:
                    for batch in pbar:
                         x = batch['x'].to(device)
                         y = batch['y'].to(device)

                         optimizer.zero_grad()
                         y_pred = model(x)
                         loss = criterion(y_pred, y)
                         loss.backward()
                         optimizer.step()

                         train_loss += loss.item()
                         pbar.set_postfix(loss=loss.item())

                         # #debug
                         # print(loss.item())
                         # #debug

          scheduler.step()
          print(f"Epoch {epoch} — Avg Train Loss: {train_loss / len(train_loader):.4e}")

          model.eval()
          val_loss = 0.0
          with torch.no_grad(), tqdm(val_loader, desc=f"[Epoch {epoch}] Validating") as pbar:
               for batch in pbar:
                    x = batch['x'].to(device)
                    y = batch['y'].to(device)
                    y_pred = model(x)
                    loss = criterion(y_pred, y)
                    val_loss += loss.item()
                    pbar.set_postfix(val_loss=loss.item())

          print(f"Epoch {epoch} — Avg Val Loss: {val_loss / len(val_loader):.4e}")

          print(f"Saving model to ./saved_models/{mode}/{mode}_saved_model.pt")
          torch.save(model.state_dict(), f"./saved_models/{mode}/{mode}_saved_model.pt")
     else: 
          print(f"Loading model from ./saved_models/{mode}/{mode}_saved_model.pt")
          model.load_state_dict(torch.load(f"./saved_models/{mode}/{mode}_saved_model.pt"))
          model.to(device)

# === Visualization ===

     # model.eval()
     # with torch.no_grad():
     #      for batch in test_loader:
     #           x = batch[0].to(device)
     #           y_true = batch[1].to(device)
     #           y_pred = model(x)
     #           break

     # for i in range(5):
     #      plot_and_save_trajectory(x, y_true, y_pred, sample_idx=i, save_dir="figures", prefix="epoch_final")

     def generate_ground_truth_gif(target_tensor, filename=f"./{mode}/{mode}_ground_truth.gif", steps=30, start_index=0, skip=1):
          # Set up figure
          fig, ax = plt.subplots()
          rcParams['animation.embed_limit'] = 2**128  # Increase limit if needed
          ax.axis('off')

          # Initial image
          initial_image = target_tensor[start_index].squeeze().cpu().numpy()

          if ("navier" in mode): 
               u = target_tensor[start_index][0].cpu().numpy()
               v = target_tensor[start_index][1].cpu().numpy()
               initial_image = np.sqrt(u**2 + v**2).squeeze()

          img_display = ax.imshow(initial_image, cmap='hot')

          def update(frame_idx):

               if ("navier" in mode): 
                    u = target_tensor[start_index + frame_idx][0].cpu().numpy()
                    v = target_tensor[start_index + frame_idx][1].cpu().numpy()
                    img = np.sqrt(u**2 + v**2)
               else: 
                    img = target_tensor[start_index + frame_idx].squeeze().cpu().numpy()


               img_display.set_data(img)
               return [img_display]

          ani = animation.FuncAnimation(
               fig,
               update,
               frames=range(0,steps,skip), # skip frames here
               blit=True,
               repeat=False
          )

          ani.save(filename, writer='pillow', fps=5)
          plt.close(fig)

          print(f"Ground truth GIF saved as: {filename}")

     if ("navier" in mode): 
          generate_ground_truth_gif(target_tensor=tensor_data, steps=100, skip=1)
     else: 
          generate_ground_truth_gif(target_tensor=tensor_data.unsqueeze(1), steps=10000, skip=20)


     def generate_gif(model, start_input, steps=30, filename=f"./{mode}/{mode}_prediction.gif", skip=1):
          model.eval()
          current = start_input.unsqueeze(0).to(device)  # shape: (1, 1, H, W)
          outputs = []

          with torch.no_grad():
               for _ in range(steps):
                    output = model(current)

                    if ("navier" in mode): 
                         u = output[0, 0].cpu().numpy()  # shape: (64, 64)
                         v = output[0, 1].cpu().numpy()  # shape: (64, 64)
                         mag = np.sqrt(u**2 + v**2)
                         outputs.append(mag)
                    else: 
                         outputs.append(output[0].cpu().squeeze().numpy())

                    current = output  # autoregressive step

          outputs = outputs[::skip] # skip frames here
          
          # Create animation
          fig, ax = plt.subplots()
          img_display = ax.imshow(outputs[0], cmap='hot')
          ax.axis('off')
          plt.tight_layout()

          def update(frame_idx):
               img_display.set_data(outputs[frame_idx])
               return [img_display]

          ani = animation.FuncAnimation(
               fig,
               update,
               frames=len(outputs),
               blit=True,
               repeat=False
          )

          ani.save(filename, writer='pillow', fps=5)
          plt.close(fig)

          print(f"GIF saved as: {filename}")


     sample_input = train_loader.dataset[0]['x'].cpu()
     # sample_input = target_tensor

     # #debug
     # print(f"test_loader.dataset[0]['x'].shape: {test_loader.dataset[0]['x'].shape}")
     # quit()
     # #debug

     generate_gif(model, sample_input, steps=100, skip=1)



if __name__ == "__main__":
    main()