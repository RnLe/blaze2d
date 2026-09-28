"""Rebuild concise tables from complete archived paired workloads."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    lines = ['# Public workflow tables', '',
        'Ratios are reference/candidate complete workflow time; above one favors the candidate. '
        'Only valid, host-admitted measured pairs enter ratios. A dash is missing evidence, not zero speedup. '
        'Rounds include reported failed correction attempts. RSS is the maximum child-process resident set '
        'over measured trials, including output; it is not native cache payload.', '']
    for group in sorted(a.directory.iterdir()):
        if not group.is_dir():
            continue
        cases = sorted(group.glob('*/summary.json'))
        if not cases:
            continue
        lines += [f'## {group.name}', '',
            '| Case | Numerical pass | Admitted pairs | Median ratio | Geometric ratio interval | Rounds R/C | Roots / seeded C | Peak RSS MiB R/C |',
            '|---|---|---:|---:|---|---|---|---|']
        for path in cases:
            d = json.loads(path.read_text())
            blocks = json.loads(path.with_name('blocks.json').read_text())
            planned_pairs = json.loads(path.with_name('manifest.json').read_text())['pairs']
            valid = [b for b in blocks if b['valid']]
            measured = blocks[1:] or blocks
            block = valid[-1] if valid else blocks[-1]
            ratio = f'{d["paired_median_ratio"]:.3f}' if d['paired_median_ratio'] is not None else '—'
            interval = d.get('ratio') or {}
            interval = '–'.join(f'{x:.3f}' for x in interval.get('interval', [])) or '—'
            rounds = [block[s]['validation'].get('work', {}).get('iterations') for s in ['reference','candidate']]
            if not d['numerical_pass']:
                rounds = 'incomplete'
            rss = [max(b[s]['resources']['rss_kib'] for b in measured)/1024 for s in ['reference','candidate']]
            w = block['candidate']['native_work']
            lines.append(f'| {path.parent.name} | {d["numerical_pass"]} | {d["admitted_pairs"]}/{planned_pairs} | '
                f'{ratio} | {interval} | {rounds} | {w["roots"]} / {w["seeded"]} | {rss[0]:.1f} / {rss[1]:.1f} |')
        lines += ['']
    lines += ['## Retained-field comparisons', '',
        '| Case | Maximum principal sine | Maximum frequency difference | Maximum path-gap difference |',
        '|---|---:|---:|---:|']
    for path in sorted(a.directory.glob('*/*/blocks.json')):
        blocks = [b for b in json.loads(path.read_text()) if 'field_comparison' in b]
        if blocks:
            lines.append(f'| {path.parent.parent.name}/{path.parent.name} | '
                f'{max(b["field_comparison"]["maximum_principal_sine"] for b in blocks):.3g} | '
                f'{max(b["max_frequency_difference"] for b in blocks):.3g} | '
                f'{max(b["max_path_gap_difference"] for b in blocks):.3g} |')
    lines += ['', 'Field comparisons use Euclidean spans of complete interior clusters on the same target grid; '
        'the unresolved final public cluster is omitted. These comparisons do not replace fresh residual '
        'and orthogonality checks or the separate f64 audit of stored f32 fields.', '']
    (a.directory / 'tables.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    main()
