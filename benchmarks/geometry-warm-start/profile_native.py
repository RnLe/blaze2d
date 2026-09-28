"""Explanatory operator/phase accounting on the frozen native sweep inputs."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'benchmarks/symmetry'))
from quiet_host import QuietHost,assess
from run_public_sweep import inspect


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--threads',type=int,default=2)
    args=parser.parse_args();args.binary=args.binary.resolve()
    args.output.mkdir(parents=True,exist_ok=True)
    inputs={}
    for case in sorted(args.inputs.glob('*/candidate.toml')):
        inputs[case.parent.name]=case
        reference=case.with_name('reference.toml')
        warm=tomllib.loads(reference.read_text())['bands']['warm_start']
        method=warm['method']
        if method=='response':method+=f"-{warm.get('response_iterations',2)}"
        inputs.setdefault(case.parent.name.split('-gamma-')[0]+f'-gamma-{method}',reference)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    binary_hash=hashlib.sha256(args.binary.read_bytes()).hexdigest()
    with QuietHost() as host:
        for name,source in inputs.items():
            output=args.output/f'{name}.profile.json'
            manifest=args.output/f'{name}.manifest.json'
            if manifest.exists():continue
            command=[str(args.binary),'run',str(source),'--threads',str(args.threads),'--profile-output',str(output)]
            before=host.snapshot()
            run=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,cwd=ROOT)
            after=host.snapshot()
            (args.output/f'{name}.result.json.gz').write_bytes(gzip.compress(run.stdout,mtime=0))
            try:validation=inspect(json.loads(run.stdout),expected_jobs=24)
            except (ValueError,KeyError,TypeError) as error:validation={'errors':[str(error)]}
            validation.pop('spectra',None)
            manifest.write_text(json.dumps(dict(command=command,returncode=run.returncode,stderr=run.stderr.decode(errors='replace'),
                validation=validation,source_revision=revision,binary_sha256=binary_hash,input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),features=['profiling'],
                host=assess(before,after,limit=2,windows_limit=4),
                scope='explanatory work counters; separate frozen unprofiled trials determine latency'),indent=2)+'\n')
            print(name,run.returncode,validation['errors'],flush=True)
            if run.returncode or validation['errors']:raise SystemExit('profile workload failed validation')


if __name__=='__main__':main()
