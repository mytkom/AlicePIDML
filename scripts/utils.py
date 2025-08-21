import dataclasses
import sys
import os

module_path = os.path.abspath('src')

if module_path not in sys.path:
    sys.path.append(module_path)

import wandb
import json
from pdi.constants import TARGET_CODE_TO_PART_NAME
from pdi.config import Config
from pdi.engines import build_engine, BaseEngine
from pdi.engines import build_engine, BaseEngine

def merge_configs(dataclass_cfg, dict_conf):
    """
    Recursively merge dictionary override config into dataclass config object,
    preserving defaults from dataclass.
    """
    for key, value in dict_conf.items():
        if hasattr(dataclass_cfg, key):
            sub_cfg = getattr(dataclass_cfg, key)
            if dataclasses.is_dataclass(sub_cfg) and isinstance(value, dict):
                merge_configs(sub_cfg, value)
            else:
                setattr(dataclass_cfg, key, value)
        else:
            raise ValueError(f"Key {key} not found in dataclass config.")

def load_config(config_path: str) -> Config:
    """
    Load the configuration from a JSON file and return a Config object.
    """
    with open(config_path, 'r') as f:
        config_data = json.load(f)
    return Config.from_dict(config_data)

def dump_default_config():
    # Config object will change in the future, this saving of
    # default values is nice starting point for creating your own config.
    # It is done every time, to make sure it is updated to latest Config.
    default_config = Config()
    with open("default_config.json", "w") as config_file:
        config_dict = dataclasses.asdict(default_config)
        json.dump(config_dict, config_file, indent=4)


def engine_single_run(config: Config, target_code: int, test: bool, sweep: bool) -> tuple[BaseEngine, dict | None]:
    particle_name = TARGET_CODE_TO_PART_NAME[target_code]

    sweep_config = None
    # Initialize logging in wandb
    if sweep:
        wandb.init(name=particle_name)
        sweep_config = wandb.config["sweep"]
        merge_configs(config, wandb.config["sweep"])
        print(f"Sweep config: {wandb.config['sweep']}")
        wandb.config.update(dataclasses.asdict(config))
    else:
        wandb.init(name=particle_name, config=dataclasses.asdict(config))

    engine: BaseEngine = build_engine(config, target_code)
    wandb.log({"base_dir": engine._base_dir})

    print(f"Running training for {particle_name}.")
    engine.train()
    print(f"End of training for {particle_name}.")

    if test:
        print(f"Running test for {particle_name}.")
        engine.test()
        print(f"Test ended for {particle_name}, end of main.")
    
    return engine, sweep_config

