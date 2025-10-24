import json
import tyro
import os
import wandb
import hashlib
import datetime
from utils import dump_default_config, engine_single_run, load_config
from pdi.constants import PART_NAME_TO_TARGET_CODE
from pathlib import Path
from pdi.config import OneParticleConfig

if __name__ == "__main__":
    dump_default_config()

    # Use tyro to parse CLI arguments and load the configuration
    cli_config = tyro.cli(OneParticleConfig)

    if cli_config.config:
        cli_config.config = str(Path(cli_config.config).resolve())
        config = load_config(cli_config.config)
    else:
        raise KeyError("Config path must be specified!")

    print(f"Loaded configuration for {cli_config.particle}:")
    # Calculate checksum of cli_config and save the file as training_runs/{checksum}.json
    cli_config_dict = cli_config.__dict__
    cli_config_dict["timestamp"] = str(datetime.datetime.now())
    cli_config_json = json.dumps(cli_config_dict, sort_keys=True)
    checksum = hashlib.md5(cli_config_json.encode()).hexdigest()
    config.project_dir = f"{config.project_dir}/sweep_{checksum}"
    output_file_path = f"{config.results_dir}/{config.project_dir}/sweep_metadata.json"
    print(f"Config project subdirectory {config.results_dir}/{config.project_dir}")
    print(config)

    with open(config.sweep.config) as f:
        sweep_configuration = json.load(f)

    print("Loaded sweep configuration:")
    print(sweep_configuration)

    sweep_metadata: dict[int, list[dict]] = {}

    if config.sweep.name:
        sweep_configuration["name"] = f"{config.sweep.name}: {cli_config.particle}"

    target_code = PART_NAME_TO_TARGET_CODE[cli_config.particle]
    sweep_metadata[target_code] = []

    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=config.sweep.project_name,
    )

    def run_and_collect():
        engine, sweep_config = engine_single_run(config, target_code, cli_config.test, sweep=True)
        sweep_metadata[target_code].append({
            "base_dir": engine._base_dir,
            "sweep_config": sweep_config
        })

        with open(output_file_path, "w+") as output_file:
            json.dump(sweep_metadata, output_file, indent=4)
        print(f"Updated metadata saved to {output_file_path}")

    wandb.agent(sweep_id, function=run_and_collect)

