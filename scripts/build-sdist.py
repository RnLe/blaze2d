"""Build a source archive with the locked dependency subset it actually contains."""
import argparse
import gzip
import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import tomllib

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--out', type=Path, default=root / 'dist')
args = parser.parse_args()
args.out.mkdir(parents=True, exist_ok=True)
subprocess.run([sys.executable, str(root / 'scripts/prepare-sdist.py')], check=True)
subprocess.run(['cargo', 'fetch', '--locked'], cwd=root, check=True)
with tempfile.TemporaryDirectory(prefix='blaze-sdist-') as folder:
    temporary = Path(folder)
    subprocess.run([sys.executable, '-m', 'maturin', 'sdist', '--manifest-path', str(root / 'crates/python/Cargo.toml'), '--out', str(temporary)], check=True)
    archive = next(temporary.glob('*.tar.gz'))
    with tarfile.open(archive) as source:
        members = source.getmembers()
        source.extractall(temporary / 'source', filter='data')
    package = next((temporary / 'source').iterdir())
    original = tomllib.loads((package / 'Cargo.lock').read_text(encoding='utf8'))
    # Maturin omits unrelated workspace members. Cargo must prune their lock
    # entries before --locked can be used by the source-distribution consumer.
    subprocess.run(['cargo', 'metadata', '--offline', '--format-version', '1'], cwd=package, stdout=subprocess.DEVNULL, check=True)
    pruned = tomllib.loads((package / 'Cargo.lock').read_text(encoding='utf8'))
    def identities(lock):
        return {(p['name'], p['version'], p.get('source'), p.get('checksum')) for p in lock['package']}
    if not identities(pruned) <= identities(original):
        raise SystemExit('Source packaging changed a locked dependency')
    subprocess.run(['cargo', 'metadata', '--locked', '--offline', '--format-version', '1'], cwd=package, stdout=subprocess.DEVNULL, check=True)
    epoch = int(subprocess.check_output(['git', 'show', '-s', '--format=%ct', 'HEAD'], cwd=root))
    destination = args.out / archive.name
    with destination.open('wb') as file, gzip.GzipFile(filename='', mode='wb', fileobj=file, mtime=epoch) as compressed, tarfile.open(fileobj=compressed, mode='w') as output:
        for member in sorted(members, key=lambda item: item.name):
            member.mtime = epoch
            member.uid = member.gid = 0
            member.uname = member.gname = ''
            member.pax_headers = {}
            if member.isfile():
                data = (temporary / 'source' / member.name).read_bytes()
                member.size = len(data)
                output.addfile(member, io.BytesIO(data))
            else:
                output.addfile(member)
    print(destination)
