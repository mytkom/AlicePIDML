import tyro
from pathlib import Path
from utils import dump_default_config, engine_single_run, load_config
from pdi.constants import PART_NAME_TO_TARGET_CODE
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

    print("Loaded Configuration:")
    print(config)

    engine_single_run(config, PART_NAME_TO_TARGET_CODE[cli_config.particle], cli_config.test, sweep=False)
