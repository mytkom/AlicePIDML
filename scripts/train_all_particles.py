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
from pdi.config import AllParticlesConfig, Config
from pdi.engines import build_engine, BaseEngine

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
    wandb.init(name=particle_name, config=dataclasses.asdict(config))

    engine: BaseEngine = build_engine(config, target_code)
    print(f"Running training for {particle_name}")
    engine.train()
    print(f"Running test for {particle_name}")
    engine.test()
    print(f"Test ended for {particle_name}, end of main")

if __name__ == "__main__":
    # Config object will change in the future, this saving of
    # default values is nice starting point for creating your own config.
    # It is done every time, to make sure it is updated to latest Config.
    default_config = Config()
    with open("default_config.json", "w") as config_file:
        config_dict = dataclasses.asdict(default_config)
        json.dump(config_dict, config_file, indent=4)

    # Use tyro to parse CLI arguments and load the configuration
    cli_config = tyro.cli(AllParticlesConfig)

    if cli_config.all:
        cli_config.all = str(Path(cli_config.all).resolve())
        all_config = load_config(cli_config.all)
    else:
        raise KeyError("Config path must be specified!")

    for part_name in PART_NAME_TO_TARGET_CODE.keys():
        if cli_config.__getattribute__(part_name):
            cli_config.__setattr__(part_name, str(Path(cli_config.__getattribute__(part_name)).resolve()))
            config = load_config(cli_config.__getattribute__(part_name))
        else:
            config = all_config
            print(f"Loading default (all) config for {part_name}")

        # Print the loaded or overridden configuration
        print(f"Loaded configuration for {part_name}:")
        print(config)

        main(config, PART_NAME_TO_TARGET_CODE[part_name])
