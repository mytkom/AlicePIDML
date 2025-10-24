import json
import os
import tyro
import wandb
import hashlib
import datetime
from utils import dump_default_config, engine_single_run, load_config
from pdi.constants import PART_NAME_TO_TARGET_CODE, TARGET_CODES
from pathlib import Path
from pdi.config import AllParticlesConfig

if __name__ == "__main__":
    dump_default_config()

    cli_config = tyro.cli(AllParticlesConfig)

    if cli_config.all:
        cli_config.all = str(Path(cli_config.all).resolve())
        all_config = load_config(cli_config.all)
    else:
        raise KeyError("Config path must be specified!")

    sweep_metadata: dict[int, list[dict]] = {}

    cli_config_dict = cli_config.__dict__
    cli_config_dict["timestamp"] = str(datetime.datetime.now())
    cli_config_json = json.dumps(cli_config_dict, sort_keys=True)
    checksum = hashlib.md5(cli_config_json.encode()).hexdigest()

    for part_name, target_code in PART_NAME_TO_TARGET_CODE.items():
        if target_code not in TARGET_CODES:
            continue

        sweep_metadata[target_code] = []

        if cli_config.__getattribute__(part_name):
            cli_config.__setattr__(part_name, str(Path(cli_config.__getattribute__(part_name)).resolve()))
            config = load_config(cli_config.__getattribute__(part_name))
        else:
            config = all_config
            print(f"Loading default (all) config for {part_name}")

        print(f"Loaded configuration for {part_name}:")
        # Calculate checksum of cli_config and save the file as training_runs/{checksum}.json

        config.project_dir = f"{config.project_dir}/sweep_{checksum}"
        output_file_path = f"{config.results_dir}/{config.project_dir}/sweep_metadata.json"
        print(f"Config project subdirectory {config.results_dir}/{config.project_dir}")
        print(config)

        with open(config.sweep.config) as f:
            sweep_configuration = json.load(f)

        print("Loaded sweep configuration:")
        print(sweep_configuration)

        if config.sweep.name:
            sweep_configuration["name"] = f"{config.sweep.name}: {part_name}"

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
