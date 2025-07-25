import dataclasses
import sys
import os

import wandb
from wandb.apis.public import Run

module_path = os.path.abspath('src')

if module_path not in sys.path:
    sys.path.append(module_path)

import json
import yaml
import tyro
from pdi.constants import PARTICLES_DICT, TARGET_CODES
from pathlib import Path
from pdi.config import Config
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

def main(config: Config):
    """
    Main function to handle the configuration.
    """
    wandb.init()
    merge_configs(config, wandb.config["sweep"])
    wandb.config.update(dataclasses.asdict(config))

    engine: BaseEngine = build_engine(config)

    for target_code in TARGET_CODES:
        print(f"Training model for {PARTICLES_DICT[target_code]}!")
        engine.train(target_code)

    for target_code in TARGET_CODES:
        print(f"Testing model for {PARTICLES_DICT[target_code]}!")
        engine.test(target_code)

if __name__ == "__main__":
    # Config object will change in the future, this saving of
    # default values is nice starting point for creating your own config.
    # It is done every time, to make sure it is updated to latest Config.
    default_config = Config()
    with open("default_config.json", "w") as config_file:
        config_dict = dataclasses.asdict(default_config)
        json.dump(config_dict, config_file, indent=4)

    # Use tyro to parse CLI arguments and load the configuration
    config = tyro.cli(Config)

    # TODO: make CLI arguments overwrite JSON arguments (now it is the other
    # way around and JSON must be complete---CLI is not used if JSON is provided)
    if config.config:
        config.config = str(Path(config.config).resolve())
        config = load_config(config.config)

    # Print the loaded or overridden configuration
    print("Loaded configuration before appliying sweep values:")
    print(config)

    with open(config.sweep.config) as f:
        sweep_configuration = json.load(f)

    print("Loaded sweep configuration:")
    print(sweep_configuration)

    if config.sweep.name:
        sweep_configuration["name"] = config.sweep.name

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=config.sweep.project_name,
    )
    wandb.agent(sweep_id, function=lambda: main(config))
