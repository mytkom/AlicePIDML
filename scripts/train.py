import dataclasses
import sys
import os

module_path = os.path.abspath('src')

if module_path not in sys.path:
    sys.path.append(module_path)

import json
from pdi.constants import PARTICLES_DICT, TARGET_CODES
from pdi.engines.classic_engine import ClassicEngine
import tyro
from pathlib import Path
from pdi.data.config import Config

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
    with open("default_config.json", "w") as config_file:
        config_dict = dataclasses.asdict(config)
        json.dump(config_dict, config_file, indent=4)

    # Save the JSON filepath to the config object
    if config.config:
        config.config = str(Path(config.config).resolve())

    config = load_config(config.config)
    
    # Print the loaded or overridden configuration
    print("Loaded Configuration:")
    print(config)

    engine = ClassicEngine(config)

    for target_code in TARGET_CODES:
        print(f"Training model for {PARTICLES_DICT[target_code]}!")
        engine.train(target_code)

    for target_code in TARGET_CODES:
        print(f"Testing model for {PARTICLES_DICT[target_code]}!")
        engine.test(target_code)

if __name__ == "__main__":
    # Use tyro to parse CLI arguments and load the configuration
    cli_config = tyro.cli(Config)
    main(cli_config)
