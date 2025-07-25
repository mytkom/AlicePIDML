import dataclasses
import sys
import os


module_path = os.path.abspath('src')

if module_path not in sys.path:
    sys.path.append(module_path)

import json
import tyro
from pdi.constants import PARTICLES_DICT, TARGET_CODES
from pathlib import Path
from pdi.config import Config
from pdi.engines import build_engine, BaseEngine

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
    # Save the JSON filepath to the config object
    # TODO: make CLI arguments overwrite JSON arguments (now it is the other
    # way around and JSON must be complete---CLI is not used if JSON is provided)
    if config.config:
        config.config = str(Path(config.config).resolve())
        config = load_config(config.config)

    # Print the loaded or overridden configuration
    print("Loaded Configuration:")
    print(config)

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
    cli_config = tyro.cli(Config)
    main(cli_config)
