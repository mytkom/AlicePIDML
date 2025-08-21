from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ConstantLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    PolynomialLR,
)

from pdi.config import TrainingConfig


def build_lr_scheduler(cfg: TrainingConfig, optimizer: Optimizer):
    if cfg.lr_scheduler is None:
        # If no lr schedule is set, we return Constant scheduler with factor 1 to
        # simulate the situation of no lr scheduler, while persisting common interface
        return ConstantLR(optimizer, factor=1.0)
    elif cfg.lr_scheduler == "exponential":
        return ExponentialLR(optimizer, gamma=cfg.lr_schedulers.exponential.gamma)
    elif cfg.lr_scheduler == "cosine_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=cfg.lr_schedulers.cosine_restarts.first_cycle_epochs,
            T_mult=cfg.lr_schedulers.cosine_restarts.cycle_epoch_inc,
        )
    elif cfg.lr_scheduler == "polynomial":
        return PolynomialLR(
            optimizer,
            total_iters=cfg.max_epochs,
            power=cfg.lr_schedulers.polynomial.power,
        )
    elif cfg.lr_scheduler == "constant":
        return ConstantLR(
            optimizer,
            factor=cfg.lr_schedulers.constant.factor,
            total_iters=cfg.lr_schedulers.constant.total_iters,
        )
    else:
        raise KeyError(f"Learning rate scheduler: {cfg.lr_scheduler} is not known!")
