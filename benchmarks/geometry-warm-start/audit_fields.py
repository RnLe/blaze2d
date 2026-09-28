"""Recheck unique archived studies using the existing f64 precision auditor."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    seen, records = {}, []
    with tempfile.TemporaryDirectory(prefix='blaze-stored-audit-') as folder:
        restored = Path(folder) / 'study.json'
        for source in sorted(a.inputs.glob('*/*.json.gz')):
            if not source.stem.endswith(('-reference.json', '-candidate.json')):
                continue
            document = json.loads(gzip.decompress(source.read_bytes()))
            for result in document['results']:
                digest = result.pop('arrays_object_sha256')
                encoded = gzip.decompress((source.parent / 'objects' / f'{digest}.json.gz').read_bytes())
                assert hashlib.sha256(encoded).hexdigest() == digest
                result['arrays'] = json.loads(encoded)
            # Include configuration and array contents, exclude wall-clock metadata.
            identity = hashlib.sha256(json.dumps([document['config'],
                [(r['job_index'], r['arrays']) for r in document['results']], document['errors']],
                sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            if identity in seen:
                records.append(dict(source=str(source), identical_fields=seen[identity]))
                continue
            seen[identity] = str(source)
            restored.write_text(json.dumps(document, separators=(',', ':')))
            run = subprocess.run([str(a.binary.resolve()), '--audit', str(restored)], capture_output=True)
            destination = a.output / f'{source.parent.name}-{source.name}'
            destination.write_bytes(gzip.compress(run.stdout, mtime=0))
            try:
                report = json.loads(run.stdout)
                passed = run.returncode == 0 and report['pass']
            except (ValueError, KeyError):
                passed = False
            records.append(dict(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                output=str(destination), pass_=passed, returncode=run.returncode,
                stderr=run.stderr.decode(errors='replace')))
            print(source.parent.name, source.name, passed, flush=True)
    if not seen:
        raise SystemExit('No archived studies with retained fields were found')
    manifest = dict(binary=str(a.binary), binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),
        collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), records=records,
        pass_=all(r.get('pass_', True) for r in records),
        scope='all unique stored field studies; fresh f64 residual/Gram actions and an ordinary f64 spectral control')
    (a.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    if not manifest['pass_']:
        raise SystemExit('Stored-field audit failed; consult the retained records')


if __name__ == '__main__':
    main()
