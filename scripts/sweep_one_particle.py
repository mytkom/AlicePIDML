import json
import tyro
import wandb
import hashlib
from scripts.utils import dump_default_config, engine_single_run, load_config
from pdi.constants import PART_NAME_TO_TARGET_CODE
from pathlib import Path
from pdi.config import OneParticleConfig

if __name__ == "__main__":
    dump_default_config()

    # Use tyro to parse CLI arguments and load the configuration
    cli_config = tyro.cli(OneParticleConfig)

    if not cli_config.output_file:
        # Calculate checksum of cli_config and save the file as training_runs/{checksum}.json
        cli_config_dict = cli_config.__dict__
        cli_config_json = json.dumps(cli_config_dict, sort_keys=True)
        checksum = hashlib.md5(cli_config_json.encode()).hexdigest()
        output_dir = Path("training_runs")
        output_dir.mkdir(exist_ok=True)
        cli_config.output_file = str(output_dir / f"sweep_{checksum}.json")
        
    output_file_path = str(Path(cli_config.output_file).resolve())

    if cli_config.config:
        cli_config.config = str(Path(cli_config.config).resolve())
        config = load_config(cli_config.config)
    else:
        raise KeyError("Config path must be specified!")

    print(f"Loaded configuration for {cli_config.particle}:")
    print(config)

    with open(config.sweep.config) as f:
        sweep_configuration = json.load(f)

    print("Loaded sweep configuration:")
    print(sweep_configuration)

    sweep_metadata: dict[int, list[dict]] = {}

    if config.sweep.name:
        sweep_configuration["name"] = f"{config.sweep.name}: {cli_config.particle}"

    target_code = PART_NAME_TO_TARGET_CODE[cli_config.particle]

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

        with open(output_file_path, "w") as output_file:
            json.dump(sweep_metadata, output_file, indent=4)
        print(f"Updated metadata saved to {cli_config.output_file}")

    wandb.agent(sweep_id, function=run_and_collect)

