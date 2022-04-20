import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tqdm.notebook import tqdm

# from expert_dataset import ExpertDataset
# from models.cilrs import CILRS
import torch.nn.functional as F
import matplotlib.pyplot as plt


def calc_loss(preds, batch, loss_fn):
    """Calculate loss for a batch"""
    action_pred, speed_pr = preds
#     action_pred = branches[torch.arange(branches.shape[0]),batch['command'],:] 
    loss = 1.5*loss_fn(action_pred[:,0:1], batch['labels'][:,0:1])
    loss = loss_fn(action_pred[:,1:3], batch['labels'][:,1:3])
    loss += loss_fn(speed_pr, batch['speed'])
    return loss/3

def validate(model, dataloader, loss_fn):
    """Validate model performance on the validation dataset"""
    # Your code here
    total_loss = 0.0
    model.eval()
    for batch in dataloader:
        preds = model(batch['rgb'], batch['speed'], batch['command'])
        loss = calc_loss(preds, batch, loss_fn)
        total_loss += loss.item()
    total_loss /= len(dataloader)
    return total_loss

def train(model, dataloader, optim, loss_fn):
    """Train model on the training dataset for one epoch"""
    # Your code here
#     total_loss = 0.0
    losses = []
    model.train()
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            optim.zero_grad()
            preds = model(batch['rgb'], batch['speed'], batch['command'])
            loss = calc_loss(preds, batch, loss_fn)
            loss.backward()
            optim.step()
            tepoch.set_postfix(loss=loss.item())
            losses.append(loss.item())
#             total_loss += loss.item()
#     total_loss /= len(dataloader)
    return losses


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    plt.plot(np.linspace(0, len(val_loss)-1,len(train_loss)), train_loss, label = "Train")
    plt.plot(np.arange(len(val_loss)), val_loss, label = "Val")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('losses.png')

def main():
#     # Change these paths to the correct paths in your downloaded expert dataset
train_root = "/userfiles/eozsuer16/expert_data/train"
val_root = "/userfiles/eozsuer16/expert_data/val"
model = CILRS()
model = model.cuda()
train_dataset = ExpertDataset(train_root)
val_dataset = ExpertDataset(val_root)

# You can change these hyper parameters freely, and you can add more
num_epochs = 12
batch_size = 64
save_path = "cilrs_model_no_act_weighted_512_mse.ckpt"

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size//2, shuffle=False)

params = model.parameters()
optim = torch.optim.Adam(params, lr=0.00015)
# loss_fn = nn.L1Loss()
loss_fn = nn.MSELoss()

train_losses = []
val_losses = []
for i in range(num_epochs):
    train_losses.append(train(model, train_loader, optim, loss_fn))
    val_losses.append(validate(model, val_loader, loss_fn))
    print(val_losses)
    torch.save(model, save_path)

train_losses2 = sum(train_losses, [])
plot_losses(train_losses2, val_losses)


if __name__ == "__main__":
    main()
