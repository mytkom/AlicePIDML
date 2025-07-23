from torch.optim import Optimizer
from torch.optim.lr_scheduler import ExponentialLR
from pdi.data.config import TrainingConfig

def build_lr_scheduler(cfg: TrainingConfig, optimizer: Optimizer):
    if cfg.lr_scheduler == "Exponential":
        return ExponentialLR(optimizer, gamma=cfg.lr_schedulers.exponential.gamma)
    else:
        raise KeyError(f"Learning rate scheduler: {cfg.lr_scheduler} is not known!")
