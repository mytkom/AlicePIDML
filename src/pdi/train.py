import torch
import torch.nn as nn
import wandb
from tqdm import tqdm

from pdi.data.constants import GROUP_ID_KEY
from pdi.evaluate import validate_model


def train_one_epoch(model, target_code, train_loader, device, optimizer,
                    loss_fun):
    model.train()
    LOG_EVERY = 50
    loss_run_sum = 0
    for i, (input_data, targets, data_dict) in enumerate(tqdm(train_loader),
                                                         start=1):
        input_data = input_data.to(device)
        binary_targets = (targets == target_code).type(torch.float).to(device)
        optimizer.zero_grad()

        group_id = data_dict.get(GROUP_ID_KEY)
        if group_id is None:
            out = model(input_data)
        else:
            out = model(input_data, group_id)
        loss = loss_fun(out, binary_targets)
        loss.backward()
        optimizer.step()

        loss_run_sum += loss.item()
        if i % LOG_EVERY == 0:
            wandb.log({"loss": loss_run_sum})
            loss_run_sum = 0


def train(model, target_code, device, train_loader, val_loader, pos_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                       wandb.config.gamma)

    if pos_weight is not None:
        loss_fun = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fun = nn.BCEWithLogitsLoss()

    min_loss = torch.inf
    counter = 0
    for epoch in range(wandb.config.max_epochs):
        train_one_epoch(model, target_code, train_loader, device, optimizer,
                        loss_fun)
        val_loss, val_f1, val_prec, val_rec, val_thres = validate_model(
            model, target_code, val_loader, device, loss_fun)
        model.thres = val_thres
        scheduler.step()
        wandb.log({
            "epoch": epoch,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_threshold": val_thres,
            "scheduled_lr": scheduler.get_last_lr()[0],
        })
        print(f"Epoch: {epoch}, F1: {val_f1:.4f}")

        if 1 - val_loss / min_loss > wandb.config.patience_threshold:
            min_loss = val_loss
            counter = 0
        else:
            counter += 1

        if counter > wandb.config.patience:
            print(f"Finishing training early at epoch: {epoch}")
            break
