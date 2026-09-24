"""Freeze matched admission builds in fresh, source-specific Cargo targets."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True,
                        help='fresh build directory, preferably on the Linux filesystem')
    args = parser.parse_args()
    subprocess.run(['git', 'diff', 'HEAD', '--exit-code'], cwd=ROOT, check=True)
    revision = lambda ref: subprocess.check_output(
        ['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
    parent, candidate = revision('d0919e0'), revision('HEAD')
    harness = ROOT / 'crates/core/examples/symmetry_memory_bench.rs'
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'binaries').mkdir()
    manifest = dict(parent_revision=parent, candidate_revision=candidate,
                    harness_sha256=sha(harness), cargo_lock_sha256=sha(ROOT / 'Cargo.lock'),
                    rustc=subprocess.check_output(['rustc', '-Vv'], text=True).strip(),
                    builder_sha256=sha(__file__), artifacts={}, complete=False)

    def save():
        (output / 'builds.json').write_text(json.dumps(manifest, indent=2) + '\n')

    save()
    for label, commit in [('baseline', parent), ('candidate', candidate)]:
        source = output / (label + '-source')
        source.mkdir()
        paths = ['Cargo.toml', 'Cargo.lock', 'crates']
        if subprocess.check_output(['git', 'ls-tree', '--name-only', commit, '.cargo'],
                                   cwd=ROOT, text=True).strip():
            paths.append('.cargo')
        archive = subprocess.check_output(['git', 'archive', commit, *paths], cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(source, filter='data')
        shutil.copy2(harness, source / 'crates/core/examples/symmetry_memory_bench.rs')
        assert sha(source / 'Cargo.lock') == manifest['cargo_lock_sha256']
        for kind in ['unprofiled', 'profiled']:
            key = label + '_' + kind
            target = output / (key + '-target')
            target.mkdir()
            binary = output / 'binaries' / key
            features = 'batched-fft' + (',profiling' if kind == 'profiled' else '')
            command = ['cargo', 'build', '--release', '--locked', '-p', 'blaze2d-core',
                       '--example', 'symmetry_memory_bench', '--features', features]
            env = dict(os.environ, CARGO_TARGET_DIR=str(target), BLAZE_SOURCE_REVISION=commit)
            log_path = output / (key + '-build.log')
            with log_path.open('x') as log:
                subprocess.run(command, cwd=source, env=env, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
            shutil.copy2(target / 'release/examples/symmetry_memory_bench', binary)
            manifest['artifacts'][key] = dict(path=str(binary), sha256=sha(binary),
                command=command, cwd=str(source), target_directory=str(target),
                source_revision=commit, source_archive_sha256=hashlib.sha256(archive).hexdigest(),
                build_log_sha256=sha(log_path))
            save()
            print(json.dumps(dict(built=key, source_revision=commit, sha256=sha(binary))), flush=True)
    for kind in ['unprofiled', 'profiled']:
        assert manifest['artifacts']['baseline_' + kind]['sha256'] != manifest['artifacts']['candidate_' + kind]['sha256'], \
            'parent and candidate must differ for this implementation change'
    manifest['complete'] = True
    save()


if __name__ == '__main__':
    main()
