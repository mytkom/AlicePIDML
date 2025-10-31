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


def build_engine(
    cfg: Config, target_code: int, base_dir: str | None = None
) -> BaseEngine:
    if cfg.model.architecture in ["mlp", "ensemble", "attention"]:
        return ClassicEngine(cfg, target_code, base_dir)
    if cfg.model.architecture in ["attention_dann", "attention_cdan"]:
        return DomainAdaptationEngine(cfg, target_code, base_dir)
    raise KeyError(
        f"There is no suitable engine for model architecture: {cfg.model.architecture}! Add one."
    )
