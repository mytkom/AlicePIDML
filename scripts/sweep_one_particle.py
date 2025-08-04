import json
import tyro
import wandb
from scripts.utils import dump_default_config, engine_single_run, load_config
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
    wandb.agent(sweep_id, function=lambda: engine_single_run(config, PART_NAME_TO_TARGET_CODE[cli_config.particle], cli_config.test, sweep=True))

