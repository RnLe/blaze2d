"""Compare scientific fixtures through Python, the native CLI, and fresh WASM.

Examples retain their public defaults. Numerical comparisons apply the same
explicit convergence profile to each backend and compare gauge invariants.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import blaze

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--native', type=Path, required=True)
parser.add_argument('--wasm', type=Path, required=True)
parser.add_argument('--output', type=Path, help='Keep results and the comparison report')
parser.add_argument('--fixture', action='append', help='Limit to a fixture filename')
args = parser.parse_args()
args.native, args.wasm = args.native.resolve(), args.wasm.resolve()


def compare(reference, candidate):
    assert reference['array_info'] == candidate['array_info'], 'array descriptors'
    for name in ('frequencies', 'eigenvalues', 'k_points', 'k_points_cartesian', 'distances'):
        if name in reference:
            a, b = reference[name], candidate[name]
            if name == 'frequencies':
                # Degenerate starting modes can select different tracked branch
                # labels. Compare the spectrum at each point without altering
                # either returned frequency/eigenvector ordering.
                a, b = np.sort(a, axis=-1), np.sort(b, axis=-1)
            np.testing.assert_allclose(a, b, rtol=5e-8, atol=5e-10, err_msg=name)
    for name in ('band_indices', 'solved_band_indices', 'retained_band_indices', 'remote_band_indices',
                 'label_indices', 'labels', 'multi_index', 'gauge', 'config', 'coordinates', 'quantities',
                 'stencil_execution_order', 'reference_sample_index', 'solver_block_size', 'build'):
        assert reference['metadata'].get(name) == candidate['metadata'].get(name), name
    for result in (reference, candidate):
        certificate = result['metadata'].get('certification')
        if certificate:
            assert certificate['max_residual'] == float(np.max(result['residuals']))
            assert certificate['b_orthogonality_defect'] < 1e-8, certificate
            # Eigenvalue-change stopping does not prescribe a residual. These
            # fixtures have a separate backward-error acceptance bound. Raw
            # per-vector residuals are not invariant under degenerate rotations.
            assert certificate['max_residual'] < 1e-3, certificate
    for name, info in reference['array_info'].items():
        a, b = reference[name], candidate[name]
        assert np.isfinite(b).all(), name
        # A Berry connection includes the derivative of the chosen gauge,
        # so its matrix spectrum is not a gauge invariant.
        if name != 'berry_connection_matrices' and info['dimensions'][-2:] in (['retained_band', 'solved_band'], ['retained_band', 'retained_band'],
                                       ['remote_band', 'retained_band'], ['solved_band', 'solved_band']):
            # Independent unitary changes within the selected spaces preserve
            # singular values, including degenerate eigenspaces.
            np.testing.assert_allclose(np.linalg.svd(a, compute_uv=False), np.linalg.svd(b, compute_uv=False),
                                       rtol=1e-5, atol=1e-6, err_msg=name + ' singular values')
        if name == 'eigenvectors' and a.ndim == 3:
            qa = np.linalg.qr(a.reshape(a.shape[0], -1).T)[0]
            qb = np.linalg.qr(b.reshape(b.shape[0], -1).T)[0]
            overlap = np.linalg.svd(qa.conj().T @ qb, compute_uv=False)
            assert np.sqrt(max(0.0, 1 - float(overlap.min())**2)) < 1e-3, 'solved eigenspace'
    assert len(reference.get('samples', [])) == len(candidate.get('samples', []))
    for a, b in zip(reference.get('samples', []), candidate.get('samples', [])):
        compare(a, b)


fixtures = sorted((root / 'crates/interface/tests/fixtures').glob('*.toml'))
fixtures += sorted((root / 'examples/calculations').glob('*.toml'))
if args.fixture:
    fixtures = [path for path in fixtures if path.name in args.fixture]
    assert fixtures, 'No matching fixtures'
report, seen = [], {}
with nullcontext(args.output) if args.output else tempfile.TemporaryDirectory(prefix='blaze-parity-') as folder:
    directory = Path(folder)
    directory.mkdir(parents=True, exist_ok=True)
    for index, fixture in enumerate(fixtures):
        print(f'Checking {fixture.relative_to(root)}', flush=True)
        entry = {'fixture': str(fixture.relative_to(root))}
        try:
            original = blaze.Config.from_file(fixture)
            normalized = subprocess.check_output([str(args.native), 'config', 'normalize', str(fixture)], text=True, encoding='utf8')
            assert blaze.Config.from_toml(normalized).to_dict() == original.to_dict(), 'default normalization'
            identity = json.dumps(original.to_dict(), sort_keys=True)
            if identity in seen:
                entry.update(status='passed', equivalent_fixture=seen[identity])
                report.append(entry)
                print(f'{fixture.name}: shares a verified calculation', flush=True)
                continue
            settings = original.to_dict()
            settings['eigensolver'].update(tolerance=1e-10, max_iterations=600)
            if settings['task'] == 'operators':
                settings['results']['eigenvectors'] = True
            config = blaze.Config.from_dict(settings)
            source = directory / f'{index}-{fixture.name}'
            source.write_text(config.to_toml(), encoding='utf8')
            reference = blaze.run(config, threads=1)
            assert reference['statistics']['status'] == 'completed', reference['errors']
            blaze.save(reference, directory / f'{index}-python.npz')
            native, browser = directory / f'{index}-native.json', directory / f'{index}-wasm.json'
            subprocess.run([str(args.native), 'run', str(source), '--threads', '1', '-o', str(native)], check=True)
            subprocess.run(['node', str(root / 'scripts/verify-wasm.mjs'), str(args.wasm), str(source), str(browser)], check=True)
            for backend, result in [('native', blaze.load(native)), ('wasm', blaze.load(browser))]:
                entry['backend'] = backend
                assert result['config'] == config.to_dict(), 'run configuration'
                assert len(result['results']) == len(reference['results']), 'result count'
                for a, b in zip(reference['results'], result['results']):
                    compare(a, b)
            entry['status'] = 'passed'
            seen[identity] = entry['fixture']
        except Exception as error:
            entry.update(status='failed', error=str(error))
            print(f'{fixture.name}: {entry.get("backend", "execution")} failed: {error}', flush=True)
        report.append(entry)
        (directory / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf8')
        print(f'{fixture.name}: {entry["status"]}', flush=True)
if any(item['status'] != 'passed' for item in report):
    raise SystemExit(f'{sum(item["status"] == "failed" for item in report)} fixtures failed')
print(f'{len(report)} fixtures agree across Python, native CLI, and WASM.')
