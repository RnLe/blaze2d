from pathlib import Path
import blaze

config = blaze.Config.from_file(Path(__file__).with_suffix(".toml"))
result = blaze.solve(config)
print(result["eigenvalues"])
