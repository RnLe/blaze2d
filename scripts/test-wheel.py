"""Install a wheel in a clean environment and run tests outside the checkout."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--wheel-dir', type=Path, required=True)
parser.add_argument('--revision', required=True)
args = parser.parse_args()
wheels = list(args.wheel_dir.resolve().glob('*.whl'))
if len(wheels) != 1:
    raise SystemExit('Expected exactly one wheel for this interpreter and platform')
root = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='blaze-wheel-') as directory:
    temporary = Path(directory)
    subprocess.run([sys.executable, '-m', 'venv', '--without-pip', str(temporary / 'venv')], check=True)
    python = temporary / 'venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    environment = dict(os.environ)
    environment.pop('PYTHONPATH', None)
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--disable-pip-version-check', str(wheels[0]), 'pytest==9.1.1'], check=True, cwd=temporary, env=environment)
    shutil.copytree(root / 'crates/python/tests', temporary / 'crates/python/tests')
    shutil.copytree(root / 'crates/interface/tests/fixtures', temporary / 'crates/interface/tests/fixtures')
    shutil.copytree(root / 'examples/calculations', temporary / 'examples/calculations')
    check = 'import blaze,json; info=blaze.build_info(); print(json.dumps(info)); assert info["source_revision"] == ' + repr(args.revision)
    subprocess.run([str(python), '-c', check], check=True, cwd=temporary, env=environment)
    subprocess.run([str(python), '-m', 'pytest', 'crates/python/tests', '-q'], check=True, cwd=temporary, env=environment)
