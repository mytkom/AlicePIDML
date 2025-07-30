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

def build_engine(cfg: Config, target_code: int) -> BaseEngine:
    if cfg.model.architecture in ["mlp", "ensemble", "attention"]:
        return ClassicEngine(cfg, target_code)
    elif cfg.model.architecture in ["attention_dann"]:
        return DomainAdaptationEngine(cfg, target_code)
    else:
        raise KeyError(f"There is no suitable engine for model architecture: {cfg.model.architecture}! Add one.")
