from pdi.config import Config
from pdi.engines.classic_engine import ClassicEngine
from pdi.engines.domain_adaptation_engine import DomainAdaptationEngine
from pdi.engines.base_engine import BaseEngine

__all__ = [
    "build_engine",
    "ClassicEngine",
    "DomainAdaptationEngine",
    "BaseEngine",
]

def build_engine(cfg: Config):
    if cfg.model.architecture in ["MLP", "Ensemble", "Attention"]:
        return ClassicEngine(cfg)
    elif cfg.model.architecture in ["AttentionDANN"]:
        return DomainAdaptationEngine(cfg)
    else:
        raise KeyError(f"There is no suitable engine for model architecture: {cfg.model.architecture}! Add one.")
