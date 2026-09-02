from pathlib import Path
import blaze

config = blaze.Config.from_file(Path(__file__).with_suffix(".toml"))
study = blaze.run(config)
for result in study["results"]:
    print(result["job_index"], result["metadata"]["sweep_parameters"])
