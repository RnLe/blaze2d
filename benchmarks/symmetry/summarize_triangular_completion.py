#!/usr/bin/env python3
"""Convergence and variational checks from retained triangular diagnostics."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for root in args.root:
        for source in sorted(root.glob('[0-9][0-9]-*.json')):
            original = json.loads(source.read_text())
            result = original['result']
            report, reference = result['report'], result['physical_reference']
            subspace = reference.get('field_subspace') or {}
            frequency_error = max(reference['scaled_frequency_errors'])
            angle = subspace.get('max_sin_angle')
            spectrum = result['spectrum']
            scale = max(1, max(map(abs, spectrum)))
            ritz_defect = max((max(0, r-v)/scale for v,r in zip(spectrum,reference['spectrum'])), default=0)
            accepted = report['accepted'] and reference['accepted']
            row = dict(source=str(source), case=result['case'], size=result['grid'][0],
                       polarization=result['polarization'], cutoff=result['cutoff_in_2pi_units'],
                       smoothing=result['material_label'], tolerance=result['eigensolver']['tol'],
                       projected_accepted=report['accepted'], reference_accepted=reference['accepted'],
                       iterations=report['iterations'], reference_iterations=reference['iterations'],
                       reference_attempts=reference.get('attempts'), coefficient_count=report['coefficient_count'],
                       scaled_frequency_error=frequency_error, subspace_sine=angle, prefix=subspace.get('prefix'),
                       approximation_target=accepted and frequency_error <= 1e-3 and angle is not None and angle <= 0.1,
                       max_scaled_ritz_lower_bound_violation=ritz_defect if accepted else None,
                       covariance=result.get('covariance'), spectrum=spectrum)
            rows.append(row)
    # Cutoff refinement is variational only within an unchanged sampled material.
    # Keep distinct reference-construction/tolerance collections separate.
    groups = {}
    for row in rows:
        key = (str(Path(row['source']).parent), row['case'], row['size'],row['polarization'],row['smoothing'])
        groups.setdefault(key,[]).append(row)
    refinements=[]
    for key, group in groups.items():
        group.sort(key=lambda r:r['cutoff'])
        for low,high in zip(group,group[1:]):
            scale=max(1,max(map(abs,low['spectrum'])))
            violation=max(max(0,b-a)/scale for a,b in zip(low['spectrum'],high['spectrum']))
            refinements.append(dict(coarse=low['source'],fine=high['source'],
                accepted=low['projected_accepted'] and high['projected_accepted'],
                max_scaled_monotonicity_violation=violation))
    for row in rows:
        row.pop('spectrum')
        print(Path(row['source']).name, 'finite=',row['projected_accepted'],row['reference_accepted'],
              'error=',round(row['scaled_frequency_error'],7),'angle=',row['subspace_sine'],
              'target=',row['approximation_target'])
    args.output.write_text(json.dumps(dict(records=rows,cutoff_refinements=refinements),indent=2)+'\n')


if __name__ == '__main__':
    main()
