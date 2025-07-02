import torch
from torch import nn
from tqdm import tqdm
import wandb

from pdi.data.constants import GROUP_ID_KEY
from pdi.evaluate import validate_model
from pdi.models import NeuralNetEnsemble, AttentionModelDANN

def train_one_epoch_dann(model, target_code, source_train_loader, target_train_loader, device, optimizer, loss_fun_class, loss_fun_domain):
    model.train()
    LOG_EVERY = 50
    loss_run_sum = 0
    final_loss = 0.0
    count = 0
    loader_len = min(len(source_train_loader), len(target_train_loader))

    for i in tqdm(range(loader_len), desc="Training DANN", total=loader_len):
        source_input_data, source_targets, _ = next(iter(source_train_loader))
        target_input_data = next(iter(target_train_loader))

        source_input_data = source_input_data.to(device)
        target_input_data = target_input_data.to(device)

        source_binary_targets = (source_targets == target_code).type(torch.float).to(device)
        optimizer.zero_grad()

        out_source_class, out_source_domain = model(source_input_data)
        _, out_target_domain = model(target_input_data)

        loss_source_class = loss_fun_class(out_source_class, source_binary_targets)
        loss_source_domain = loss_fun_domain(out_source_domain, torch.zeros_like(out_source_domain))
        loss_target_domain = loss_fun_domain(out_target_domain, torch.ones_like(out_target_domain))

        loss = loss_source_class + loss_source_domain + loss_target_domain
        loss.backward()
        optimizer.step()

        loss_run_sum += loss.item()
        if i % LOG_EVERY == 0:
            wandb.log({"loss": loss_run_sum})
            loss_run_sum = 0

        final_loss += loss.item()
        count += 1
    return final_loss / count


def train_one_epoch(model, target_code, train_loader, device, optimizer, loss_fun):
    model.train()
    LOG_EVERY = 50
    loss_run_sum = 0
    final_loss = 0.0
    count = 0
    for i, (input_data, targets, data_dict) in enumerate(tqdm(train_loader), start=1):
        input_data = input_data.to(device)
        binary_targets = (targets == target_code).type(torch.float).to(device)
        optimizer.zero_grad()

        group_id = data_dict.get(GROUP_ID_KEY)
        # TODO: move NNEnsemble group choice inside model
        if isinstance(model, NeuralNetEnsemble):
            out = model(input_data, group_id)
        else:
            out = model(input_data)
        loss = loss_fun(out, binary_targets)
        loss.backward()
        optimizer.step()

        loss_run_sum += loss.item()
        if i % LOG_EVERY == 0:
            wandb.log({"loss": loss_run_sum})
            loss_run_sum = 0

        final_loss += loss.item()
        count += 1
    return final_loss / count


def train(model, target_code, device, dataloaders, pos_weight):
    optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, wandb.config.gamma)

    if pos_weight is not None:
        loss_fun = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        loss_fun = nn.BCEWithLogitsLoss()

    min_loss = torch.inf
    counter = 0
    loss_arr = []
    val_loss_arr = []
    for epoch in range(wandb.config.max_epochs):
        if isinstance(model, AttentionModelDANN):
            source_train_loader, val_loader = dataloaders["source"]
            target_train_loader = dataloaders["target"]
            loss_fun_domain = nn.BCEWithLogitsLoss()
            loss = train_one_epoch_dann(
                model, target_code, source_train_loader, target_train_loader, device, optimizer, loss_fun_class=loss_fun, loss_fun_domain=loss_fun_domain
            )
        else:
            train_loader, val_loader = dataloaders
            loss = train_one_epoch(
                model, target_code, train_loader, device, optimizer, loss_fun
            )

        val_loss, val_f1, val_prec, val_rec, val_thres = validate_model(
            model, target_code, val_loader, device, loss_fun
        )
        model.thres = val_thres
        scheduler.step()
        wandb.log(
            {
                "epoch": epoch,
                "val_loss": val_loss,
                "val_f1": val_f1,
                "val_precision": val_prec,
                "val_recall": val_rec,
                "val_threshold": val_thres,
                "scheduled_lr": scheduler.get_last_lr()[0],
            }
        )
        print(
            f"Epoch: {epoch}, F1: {val_f1:.4f}, Loss: {loss:.4f}, Val_Loss:{val_loss:.4f}"
        )
        loss_arr.append(loss)
        val_loss_arr.append(val_loss)

        if 1 - val_loss / min_loss > wandb.config.patience_threshold:
            min_loss = val_loss
            if counter > 0:
                counter -= 1
        else:
            counter += 1

        if counter > wandb.config.patience:
            print(f"Finishing training early at epoch: {epoch}")
            break

    return loss_arr, val_loss_arr
