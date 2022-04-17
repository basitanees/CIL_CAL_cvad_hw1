import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
# from tqdm.notebook import tqdm

import torch.nn.functional as F
import matplotlib.pyplot as plt

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor


def calc_acc(preds, batch):
    """Calculate loss for a batch"""
    _, disc_affordance = preds
    tl_state_trues = torch.sum(((disc_affordance>0.5)*1.0) == batch['tl_state'])

    return tl_state_trues

def validate_acc(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
#     total_loss = 0.0
    total_loss = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
#     losses = {'route_angle': 0.0, 'lane_dist': 0.0, 'tl_dist':0.0, 'tl_state': 0.0, 'loss':0.0}
    model.eval()
    total_true = 0.0
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            preds = model(batch['rgb'])
            trues = calc_acc(preds, batch)
            total_true += trues
    print(total_true)
    acc = total_true / len(dataloader)
    return acc

def calc_loss2(preds, batch):
    """Calculate loss for a batch"""
    cont_affordance, disc_affordance = preds
    cont_affordance_true = batch['affordance_cont']
    # ['route_angle'], stuff['lane_dist'], stuff['tl_dist']
    
    route_angle_true = cont_affordance_true[:,0]
    lane_dist_true = cont_affordance_true[:,1]
    tl_dist_true = cont_affordance_true[:,2]
    route_angle_pr = cont_affordance[:,0]
    lane_dist_pr = cont_affordance[:,1]
    tl_dist_pr = cont_affordance[:,2]
    route_angle_loss = F.mse_loss(route_angle_pr, route_angle_true)
    lane_dist_loss = F.mse_loss(lane_dist_pr, lane_dist_true)
    tl_dist_loss = F.mse_loss(tl_dist_pr, tl_dist_true)
    tl_state_loss = F.mse_loss(disc_affordance, batch['tl_state'])
    total_loss = route_angle_loss+ lane_dist_loss+ tl_dist_loss+ tl_state_loss
#     loss = {'route_angle': route_angle_loss, 'lane_dist': lane_dist_loss, 'tl_dist':tl_dist_loss, 'tl_state': tl_state_loss, 'loss':total_loss}
    loss = torch.tensor([route_angle_loss, lane_dist_loss, tl_dist_loss, tl_state_loss, total_loss])
    return loss

def calc_loss(preds, batch):
    """Calculate loss for a batch"""
    cont_affordance, disc_affordance = preds
    loss = 0.75 * F.mse_loss(cont_affordance, batch['affordance_cont'])
    loss += 0.25 * F.mse_loss(disc_affordance, batch['tl_state'])
    return loss

def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
    total_loss = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0])
#     losses = {'route_angle': 0.0, 'lane_dist': 0.0, 'tl_dist':0.0, 'tl_state': 0.0, 'loss':0.0}
    model.eval()
    for batch in dataloader:
        preds = model(batch['rgb'])
        loss = calc_loss2(preds, batch)
        total_loss += loss
    total_loss /= len(dataloader)
    return total_loss

def train(model, dataloader, optim):
    """Train model on the training dataset for one epoch"""
    # Your code here
    losses = []
    model.train()
    with tqdm(dataloader, unit="batch") as tepoch:
        for batch in tepoch:
            optim.zero_grad()
            preds = model(batch['rgb'])
            loss = calc_loss(preds, batch)
            loss.backward()
            optim.step()
            tepoch.set_postfix(loss=loss.item())
            losses.append(loss.item())
    return losses


def plot_losses2(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    plt.plot(np.linspace(0, len(val_loss)-1,len(train_loss)), train_loss, label = "Train")
    plt.plot(np.arange(len(val_loss)), val_loss, label = "Val")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('losses2.png')

def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    plt.plot(np.linspace(0, len(val_loss)-1,len(train_loss)), train_loss, label = "Train")
    plt.plot(np.arange(len(val_loss)), val_loss[:,0], label = "Val_route_angle")
    plt.plot(np.arange(len(val_loss)), val_loss[:,1], label = "Val_lane_dist")
    plt.plot(np.arange(len(val_loss)), val_loss[:,2], label = "Val_tl_dist")
    plt.plot(np.arange(len(val_loss)), val_loss[:,3], label = "Val_state")
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('losses3.png')

def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/home/aanees20/My Data/Comp 523/Homework 1/expert_data/train"
    val_root = "/home/aanees20/My Data/Comp 523/Homework 1/expert_data/val"
    model = AffordancePredictor().cuda()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    params = model.parameters()
    optim = torch.optim.Adam(params, lr=0.0002)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader, optim))
        val_losses.append(validate(model, val_loader))
        print("Train: ")
        print(train_losses)
        print("Val: ")
        print(val_losses)
        torch.save(model, save_path)
    # torch.save(model, save_path)
    train_losses2 = sum(train_losses, [])
    plot_losses(train_losses2, val_losses)


if __name__ == "__main__":
    main()
