from typing import Optional
from torch.functional import Tensor
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from pdi.config import TrainingConfig

def build_loss(cfg: TrainingConfig, pos_weight: Optional[Tensor] = None) -> _Loss:
    if cfg.loss == "cross entropy":
        if pos_weight:
            return BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            return BCEWithLogitsLoss()
    else:
        raise KeyError(f"Loss function {cfg.loss} is not known!")
