"""Command-line access to the shared configuration and execution contract."""
import argparse
import json
import sys
from contextlib import nullcontext
from . import _native
from .config import Config
from .api import run
from .io import save, _encode


def main(argv=None):
    parser = argparse.ArgumentParser(prog="blaze2d", description="Photonic bands and projected operators")
    parser.add_argument("--version", action="version", version=_native.__version__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Run a TOML calculation or study")
    run_parser.add_argument("file")
    run_parser.add_argument("-o", "--output", help="JSON, NDJSON, or NPZ destination")
    run_parser.add_argument("--progress", action="store_true", help="Show completed jobs on stderr (requires blaze2d[progress])")
    run_parser.add_argument("--threads", type=int, default=0)
    run_parser.add_argument("--error-policy", choices=("stop", "continue"), default="stop")
    config_parser = commands.add_parser("config", help="Inspect the configuration contract")
    config_commands = config_parser.add_subparsers(dest="action", required=True)
    for name in ("validate", "normalize"):
        config_commands.add_parser(name).add_argument("file")
    config_commands.add_parser("describe")
    args = parser.parse_args(argv)
    try:
        if args.command == "config":
            if args.action == "describe":
                print(json.dumps(_native.describe(), indent=2))
            else:
                config = Config.from_file(args.file)
                print(config.to_toml() if args.action == "normalize" else json.dumps(config.summary, indent=2))
            return 0
        if args.progress:
            from .progress import TerminalProgress
            display = TerminalProgress()
        else:
            display = nullcontext()
        with display as progress:
            study = run(Config.from_file(args.file), threads=args.threads, error_policy=args.error_policy, progress=progress)
        study['statistics']['runtime']['output'] = str(args.output) if args.output else None
        if args.output:
            save(study, args.output)
        else:
            print(json.dumps(_encode(study), allow_nan=False))
        return 0 if study["statistics"]["status"] == "completed" else 1
    except (ValueError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
