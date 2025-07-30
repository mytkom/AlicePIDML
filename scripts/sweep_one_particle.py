import dataclasses
import sys
import os

import wandb

module_path = os.path.abspath('src')

if module_path not in sys.path:
    sys.path.append(module_path)

import json
import tyro
from pdi.constants import PART_NAME_TO_TARGET_CODE, TARGET_CODE_TO_PART_NAME
from pathlib import Path
from pdi.config import Config, OneParticleConfig
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

def main(config: Config, target_code: int):
    """
    Main function to handle the configuration.
    """
    particle_name = TARGET_CODE_TO_PART_NAME[target_code]

    # Initialize logging in wandb
    wandb.init(name=particle_name)
    merge_configs(config, wandb.config["sweep"])
    print(f"Sweep config: {wandb.config['sweep']}")
    wandb.config.update(dataclasses.asdict(config))

    engine: BaseEngine = build_engine(config, target_code)
    print(f"Running training for {particle_name}")
    engine.train()
    print(f"Train ended for {particle_name}, end of main")
    # print(f"Running test for {particle_name}")
    # engine.test()
    # print(f"Test ended for {particle_name}, end of main")

if __name__ == "__main__":
    # Config object will change in the future, this saving of
    # default values is nice starting point for creating your own config.
    # It is done every time, to make sure it is updated to latest Config.
    default_config = Config()
    with open("default_config.json", "w") as config_file:
        config_dict = dataclasses.asdict(default_config)
        json.dump(config_dict, config_file, indent=4)

    # Use tyro to parse CLI arguments and load the configuration
    cli_config = tyro.cli(OneParticleConfig)

    if cli_config.config:
        cli_config.config = str(Path(cli_config.config).resolve())
        config = load_config(cli_config.config)
    else:
        raise KeyError("Config path must be specified!")

    # Print the loaded or overridden configuration
    print(f"Loaded configuration for {cli_config.particle}:")
    print(config)

    with open(config.sweep.config) as f:
        sweep_configuration = json.load(f)

    print("Loaded sweep configuration:")
    print(sweep_configuration)

    if config.sweep.name:
        sweep_configuration["name"] = f"{config.sweep.name}: {cli_config.particle}"

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=config.sweep.project_name,
    )
    wandb.agent(sweep_id, function=lambda: main(config, PART_NAME_TO_TARGET_CODE[cli_config.particle]))

