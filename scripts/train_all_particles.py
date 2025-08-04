import tyro
from scripts.utils import dump_default_config, engine_single_run, load_config
from pdi.constants import PART_NAME_TO_TARGET_CODE
from pathlib import Path
from pdi.config import AllParticlesConfig

if __name__ == "__main__":
    dump_default_config()

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

        print(f"Loaded configuration for {part_name}:")
        print(config)

        engine_single_run(config, PART_NAME_TO_TARGET_CODE[part_name], cli_config.test, sweep=False)
