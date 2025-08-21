import tyro
import json
import hashlib
from utils import dump_default_config, engine_single_run, load_config
from pdi.constants import PART_NAME_TO_TARGET_CODE
from pathlib import Path
from pdi.config import AllParticlesConfig

if __name__ == "__main__":
    dump_default_config()

    # Use tyro to parse CLI arguments and load the configuration
    cli_config = tyro.cli(AllParticlesConfig)

    if not cli_config.output_file:
        # Calculate checksum of cli_config and save the file as training_runs/{checksum}.json
        cli_config_dict = cli_config.__dict__
        cli_config_json = json.dumps(cli_config_dict, sort_keys=True)
        checksum = hashlib.md5(cli_config_json.encode()).hexdigest()
        output_dir = Path("training_runs")
        output_dir.mkdir(exist_ok=True)
        cli_config.output_file = str(output_dir / f"train_{checksum}.json")
        
    output_file_path = str(Path(cli_config.output_file).resolve())

    if cli_config.all:
        cli_config.all = str(Path(cli_config.all).resolve())
        all_config = load_config(cli_config.all)
    else:
        raise KeyError("Config path must be specified!")

    run_metadata: dict[int, dict] = {}

    for part_name, target_code in PART_NAME_TO_TARGET_CODE.items():
        if cli_config.__getattribute__(part_name):
            cli_config.__setattr__(part_name, str(Path(cli_config.__getattribute__(part_name)).resolve()))
            config = load_config(cli_config.__getattribute__(part_name))
        else:
            config = all_config
            print(f"Loading default (all) config for {part_name}")

        print(f"Loaded configuration for {part_name}:")
        print(config)

        engine, _ = engine_single_run(config, target_code, cli_config.test, sweep=True)
        run_metadata[target_code] = {
            "base_dir": engine._base_dir,
        }

        with open(output_file_path, "w") as output_file:
            json.dump(run_metadata, output_file, indent=4)
        print(f"Updated metadata saved to {cli_config.output_file}")
