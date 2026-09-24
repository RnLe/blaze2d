"""Freeze the parent and PT-sector candidate in four fresh Cargo targets."""
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
HARNESS = 'crates/core/examples/symmetry_pt_sector_bench.rs'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    subprocess.run(['git', 'diff', 'HEAD', '--exit-code'], cwd=ROOT, check=True)
    revision = lambda ref: subprocess.check_output(['git', 'rev-parse', ref], cwd=ROOT, text=True).strip()
    parent, candidate = revision('14abd91'), revision('HEAD')
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    (out/'binaries').mkdir()
    manifest = dict(schema='blaze2d/pt-sector-builds/1', parent_revision=parent,
        candidate_revision=candidate, implementation_revision=revision('15c5aa6'),
        harness_sha256=sha(ROOT/HARNESS), cargo_lock_sha256=sha(ROOT/'Cargo.lock'),
        rustc=subprocess.check_output(['rustc','-Vv'],text=True).strip(),
        build_environment={k:os.environ.get(k) for k in ('RUSTFLAGS','CARGO_ENCODED_RUSTFLAGS',
            'CARGO_BUILD_TARGET','RUSTC_WRAPPER','RUSTC_WORKSPACE_WRAPPER')},
        builder_sha256=sha(__file__), artifacts={}, complete=False)
    def save():
        (out/'builds.json').write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    for label, commit in [('baseline',parent),('candidate',candidate)]:
        source=out/(label+'-source'); source.mkdir()
        archive=subprocess.check_output(['git','archive',commit,'Cargo.toml','Cargo.lock','crates','.cargo'],cwd=ROOT)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar: tar.extractall(source,filter='data')
        shutil.copy2(ROOT/HARNESS,source/HARNESS)
        assert sha(source/'Cargo.lock')==manifest['cargo_lock_sha256']
        for kind in ['unprofiled','profiled']:
            if shutil.disk_usage(out).free < 5*1024**3:
                raise RuntimeError('less than 5 GiB free before a fresh Cargo build')
            key=label+'_'+kind; target=out/(key+'-target'); target.mkdir()
            features='batched-fft'+(',profiling' if kind=='profiled' else '')
            command=['cargo','build','--locked','--release','-p','blaze2d-core',
                     '--example','symmetry_pt_sector_bench','--features',features]
            env=dict(os.environ,CARGO_TARGET_DIR=str(target),BLAZE_SOURCE_REVISION=commit)
            log_path=out/(key+'-build.log')
            with log_path.open('x') as log:
                subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
            binary=out/'binaries'/key
            shutil.copy2(target/'release/examples/symmetry_pt_sector_bench',binary)
            manifest['artifacts'][key]=dict(path=str(binary),sha256=sha(binary),command=command,
                cwd=str(source),target_directory=str(target),source_revision=commit,
                source_archive_sha256=hashlib.sha256(archive).hexdigest(),build_log_sha256=sha(log_path))
            save(); print(json.dumps(dict(built=key,sha256=sha(binary))),flush=True)
    for kind in ['unprofiled','profiled']:
        assert manifest['artifacts']['baseline_'+kind]['sha256']!=manifest['artifacts']['candidate_'+kind]['sha256']
    manifest['complete']=True;save()


if __name__=='__main__': main()
