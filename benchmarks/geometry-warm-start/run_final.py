"""Focused covering set; each comparison uses the existing paired collector."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--group', choices=['public', 'policy', 'workers', 'controls'], required=True)
    a = p.parse_args()
    common = [sys.executable, str(Path(__file__).with_name('run_paths.py')),
              '--reference', str(a.binary), '--candidate', str(a.binary),
              '--reuse-identical', '--pairs', '3']
    groups = {
        'public': [
            ('contrast', ['--ablation', 'public', '--variant', 'gamma', '--family', 'contrast']),
            ('displacement-fields', ['--ablation', 'public', '--variant', 'gamma', '--family', 'displacement', '--size', '32', '--fields']),
            ('events', ['--ablation', 'public', '--variant', 'gamma', '--family', 'events', '--size', '32']),
            ('precision-f32', ['--ablation', 'public', '--variant', 'gamma', '--size', '24', '--bands', '4', '--precision', 'f32', '--fields']),
        ],
        'policy': [
            ('auto-reference', ['--ablation', 'public', '--variant', 'gamma', '--family', 'contrast', '--reference-auto']),
            ('selected-anchors', ['--ablation', 'public', '--variant', 'anchors', '--filter', 'triangular-TM', '--method', 'copy']),
            ('public-bank', ['--ablation', 'public', '--variant', 'path', '--filter', 'square-TE']),
        ],
        'workers': [(f'workers-{w}', ['--ablation', 'public', '--variant', 'gamma', '--family', 'contrast', '--threads', str(w), '--tolerance', '1e-13'])
                    for w in [1, 2, 4, 8]],
        'controls': [
            ('donors', ['--ablation', 'donors', '--size', '32']),
            ('guards', ['--ablation', 'guards', '--size', '32']),
            ('order', ['--ablation', 'order', '--variant', 'gamma', '--threads', '1']),
        ],
    }
    for name, options in groups[a.group]:
        command = common + ['--output', str(a.output / name)] + options
        print(' '.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
