"""Summarize the complete M7/M8 records without discarding negative outcomes."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    records = []
    for manifest in sorted(args.directory.glob('*/*/manifest.json')):
        case = json.loads(manifest.read_text()).get('case')
        if not case:
            continue
        blocks = json.loads(manifest.with_name('blocks.json').read_text())
        summary = json.loads(manifest.with_name('summary.json').read_text())
        measured = [b for b in blocks[1:] if b['valid'] and b['admitted']]
        descriptive = [b for b in blocks[1:] if b['valid']]
        selected = measured or descriptive
        row = dict(case=case, campaign=manifest.parent.parent.name, **summary)
        row['measurement_count'] = len(blocks)-1
        row['rejected_timing_pairs'] = sum(not b['admitted'] for b in blocks[1:])
        row['failed_blocks'] = [dict(index=i, errors={s:b[s]['validation']['errors'] for s in ['reference','candidate']},
            max_frequency_difference=b.get('max_frequency_difference'), field_comparison=b.get('field_comparison'))
            for i,b in enumerate(blocks) if not b['valid']]
        if selected:
            for side in ['reference', 'candidate']:
                row[side] = dict(seconds=statistics.median(b[side]['elapsed_seconds'] for b in selected),
                    rss_MiB=statistics.median(b[side]['resources']['rss_kib']/1024 for b in selected),
                    correction_iterations=sorted(set(b[side]['validation']['work']['iterations'] for b in selected)),
                    native_work=selected[0][side]['native_work'])
            row['max_frequency_difference'] = max(b['max_frequency_difference'] for b in blocks if b['valid'])
        records.append(row)
    result = dict(records=records, scope='Ratios are reference/candidate complete CLI time; above one is faster. '
        'Admitted medians exclude host-load failures; descriptive medians retain numerically valid trials. '
        'Three randomized pairs are qualitative decision gates, not precise hardware constants. '
        'RSS includes the process, runtime libraries and requested outputs. Failed training is included.')
    (args.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    lines = ['# Complete comparison table', '',
        'Speedup is reference time / candidate time. Parenthesized values have no',
        'host-admitted pair and are descriptive only. All numerical failures remain',
        'in the JSON records. RSS and rounds use admitted pairs when available.', '',
        '| Campaign / case | Pass | Speedup | Admitted | Ref / new seconds | Ref / new rounds | Ref / new RSS MiB |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in records:
        name = row['campaign']+'/'+row['case']['name']
        ratio = row['paired_median_ratio']
        speed = f'{ratio:.3f}×' if ratio is not None else f'({row["descriptive_median_ratio"]:.3f}×)' if row['descriptive_median_ratio'] else '—'
        if 'reference' in row:
            a,b = row['reference'],row['candidate']
            values = f'{a["seconds"]:.3f} / {b["seconds"]:.3f} | {a["correction_iterations"]} / {b["correction_iterations"]} | {a["rss_MiB"]:.1f} / {b["rss_MiB"]:.1f}'
        else:
            values = '— | — | —'
        lines.append(f'| {name} | {row["numerical_pass"]} | {speed} | {row["admitted_pairs"]}/{row["measurement_count"]} | {values} |')
    (args.directory/'table.md').write_text('\n'.join(lines)+'\n')
    print(f'{len(records)} comparisons; {sum(r["numerical_pass"] for r in records)} numerical passes; '
          f'{sum(r["admitted_pairs"] for r in records)}/{sum(r["measurement_count"] for r in records)} admitted pairs')


if __name__ == '__main__':
    main()
