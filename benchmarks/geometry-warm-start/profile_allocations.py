"""Complete heap counts and allocation-site peaks; never use these runs for latency."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();a.binary=a.binary.resolve();a.input=a.input.resolve()
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'input.toml').write_bytes(a.input.read_bytes())
    summary={'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        'input_sha256':hashlib.sha256(a.input.read_bytes()).hexdigest(),'threads':a.threads,
        'valgrind':subprocess.check_output(['valgrind','--version'],text=True).strip(),
        'scope':'Instrumented allocator counts/sites, not timing or hardware DRAM. Valgrind may select different SIMD implementations.'}
    for mode in ['memcheck','massif']:
        command=['valgrind',f'--tool={mode}']
        if mode=='memcheck':command+=['--leak-check=no','--track-origins=no']
        else:command +=[f'--massif-out-file={a.output / "massif.out"}','--stacks=no','--time-unit=B','--detailed-freq=1','--max-snapshots=30']
        command += [str(a.binary),'run',str(a.input),'--threads',str(a.threads)]
        run=subprocess.run(command,capture_output=True)
        (a.output/f'{mode}-stdout.json.gz').write_bytes(gzip.compress(run.stdout,mtime=0))
        text=run.stderr.decode(errors='replace');(a.output/f'{mode}.log').write_text(text)
        summary[mode]={'command':command,'returncode':run.returncode}
        if mode=='memcheck':
            match=re.search(r'total heap usage: ([\d,]+) allocs, ([\d,]+) frees, ([\d,]+) bytes allocated',text)
            if match:summary[mode].update(dict(zip(['allocations','frees','allocated_bytes'],map(lambda x:int(x.replace(',','')),match.groups()))))
            match=re.search(r'ERROR SUMMARY: ([\d,]+) errors',text)
            if match:summary[mode]['errors']=int(match[1].replace(',',''))
        else:
            path=a.output/'massif.out'
            if path.exists():
                data=path.read_text()
                summary[mode]['peak_payload_bytes']=max(map(int,re.findall(r'^mem_heap_B=(\d+)$',data,re.M)),default=0)
                path.with_suffix('.out.gz').write_bytes(gzip.compress(path.read_bytes(),mtime=0));path.unlink()
        (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(mode,summary[mode],flush=True)
        if run.returncode:break

if __name__=='__main__':main()
