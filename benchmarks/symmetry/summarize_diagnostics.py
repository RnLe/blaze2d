"""Collect completed diagnostic audits without promoting their times to benchmarks."""
from pathlib import Path
import hashlib
import json

root = Path(__file__).resolve().parent
paths = [root/'diagnostic-raw-20260928-02'/name for name in
         ['D04', 'D05', 'dispatch', 'PT', 'memory', 'root-profiles']]
paths.append(root/'diagnostic-raw-20260928-01/coarse')

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

families = []
for path in paths:
    manifest = json.loads((path/'manifest.json').read_text())
    audit = json.loads((path/'audit.json').read_text())
    if not audit.get('numerical_reproduction_passed') or audit.get('primary_latency_qualified') is not False:
        raise ValueError('Incomplete or incorrectly scoped diagnostic family: '+str(path))
    if sha(path/'records.jsonl') != audit['records_sha256']:
        raise ValueError('Diagnostic records changed: '+str(path))
    rows = [json.loads(line) for line in (path/'records.jsonl').read_text().splitlines()]
    if len(rows) != audit['unique_commands'] or any(row['issues'] for row in rows):
        raise ValueError('Diagnostic mapping is incomplete')
    families.append(dict(family=manifest['family'], evidence=str(path.relative_to(root)),
                         manifest_sha256=sha(path/'manifest.json'), audit_sha256=sha(path/'audit.json'),
                         original_records=audit['original_record_count'],
                         unique_commands=audit['unique_commands'],
                         exact_numerical_reproduction=True,
                         host_flagged_windows=sum(not r['window']['assessment']['accepted'] for r in rows)))

data = dict(schema='blaze2d/diagnostic-replay-summary/1', families=families,
            original_records=sum(r['original_records'] for r in families),
            unique_commands=sum(r['unique_commands'] for r in families),
            primary_latency_qualified=False,
            scope='Exact frozen numerical/work reproduction; times descriptive only. '
                  'Identical archived commands are deduplicated and mapped explicitly.')
(root/'diagnostic-replay-results.json').write_text(json.dumps(data, indent=2)+'\n')
lines = ['# Diagnostic remeasurement', '',
         f"All {data['unique_commands']} distinct frozen commands reproduce their numerical outputs, "
         f"certificates, rejection reasons and work, covering {data['original_records']} archived records.", '',
         '| Family | Archived records | Fresh commands | Host-flagged windows |', '|---|---:|---:|---:|']
for r in families:
    lines.append(f"| [{r['family']}]({r['evidence']}) | {r['original_records']} | {r['unique_commands']} | {r['host_flagged_windows']} |")
lines += ['',
          'These are separate explanatory executions, **not paired latency measurements**. '
          'Host observations remain visible but do not invalidate a deterministic numerical reproduction. '
          'No environment-based retries were used for the final diagnostic sweep. Complete stdout and '
          'stderr are retained losslessly; manifests pin sources, executable hashes, backend dependencies '
          'and the mapping to every archived record.', '',
          'D04 preserves the corrected 200-record convergence dataset, including the 41 repaired '
          'cluster-boundary diagnostics. All 184 uninstrumented conditions pass the projected solver '
          'checks, but only 33 meet the separate approximation target against their matching full '
          'rectangle; all 33 are TM. Exact reproduction also preserves the negative TE convergence '
          'results and sampled-material covariance defects. D05 independently qualifies sectors on '
          'that fixed pencil and preserves the material rejection controls.', '',
          'The root profiles cover the actual C08 shared engine and L01–L08: enrichment, response '
          'prediction and selection, coarse starts, precision, owned workers, routing, and native/portable/'
          'WASM interfaces. The two additional L01 held-out TE diagnostics retain the failed baseline '
          'and improved candidate. Negative outcomes are expected data, not process errors.', '',
          'The public coarse profiles add nonempty instrumentation to the same frozen implementation. '
          'Fields, physical certificates and every iteration ledger must match their unprofiled '
          'reference. The other families require exact stable counters as well. Timing fields, '
          'resident memory and timing-dependent serialization metadata are excluded from exact '
          'numerical identity; numerical output and work are not.', '',
          'The 28 archived incremental profiled comparisons are additionally addressed in the '
          '[263-comparison inventory](remeasurement-summary.md). They remain explicitly instrumented '
          'and never enter the admitted primary timing set. F02’s global-profile numerical output '
          'is identical to the F03 baseline and is covered there rather than counted twice.', '',
          'See the [current findings](remeasurement-findings.md) for mechanism-level interpretation '
          'and the [diagnostic protocol](diagnostic-replay-protocol.md) for selection and identity rules. '
          'FFT work and modeled payloads are not physical DRAM-bandwidth measurements.', '']
(root/'diagnostic-replay-results.md').write_text('\n'.join(lines))
print(json.dumps(data, indent=2))
