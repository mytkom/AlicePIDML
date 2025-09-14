from torch import nn
from torch.optim import SGD, AdamW
from pdi.config import TrainingConfig


def build_optimizer(cfg: TrainingConfig, model: nn.Module):
    if cfg.optimizer == "adamw":
        # TODO: consider more parameters for AdamW
        return AdamW(
            model.parameters(),
            lr=cfg.start_lr,
            weight_decay=cfg.optimizers.adamw.weight_decay,
        )
    if cfg.optimizer == "sgd":
        return SGD(
            model.parameters(),
            lr=cfg.start_lr,
            momentum=cfg.optimizers.sgd.momentum,
            nesterov=cfg.optimizers.sgd.nesterov,
            weight_decay=cfg.optimizers.sgd.weight_decay,
        )
    raise KeyError(f"Optimizer {cfg.optimizer} is not known!")
