"""Physical and executed-path gates for the rectangular PT/sector study."""
import math

from run_pairs import quality_passes

MODES = ('off', 'pt', 'sectors', 'fourier-sectors', 'pt-sectors')
TIMING_KEYS = {'seconds', 'peak_rss_kib', 'counters', 'stage_counters', 'profile'}


def numerics(value):
    return {k: v for k, v in value.items() if k not in TIMING_KEYS}


def path_issues(value, require_combined=True):
    """A certified whole-solve fallback is still a failed requested path."""
    issues = []
    try:
        mode, case, bands = value['mode'], value['case'], value['bands']
        if mode not in MODES or case not in ('gamma', 'mirror', 'rotation'):
            return ['unknown requested mode/case']
        nx, ny = value['grid']; rows = nx * ny
        physical, policy = value['physical_configuration'], value['policies']
        expected_points = [[0., 0.], [.125, 0.], [.25, 0.], [.5, 0.], [0., 0.]] if case == 'mirror' else [[0., 0.]]
        if (physical['grid'] != value['grid'] or physical['grid_centered'] is not False
                or physical['polarization'] != value['polarization']
                or physical['precision'] != 'f64' or value['precision'] != 'f64'
                or physical['eigensolver']['n_bands'] != bands
                or physical['eigensolver']['tol'] != value['tolerance']
                or physical['eigensolver']['max_iter'] != 500
                or physical['output'] != 'full_spatial_complex_fields'
                or physical['band_tracking'] is not False
                or physical['legacy_gamma_reuse'] is not False
                or physical['k_path'] != expected_points
                or [p['k'] for p in value['points']] != expected_points
                or value['expected_points'] != len(expected_points)):
            issues.append('physical configuration/output contract differs')
        if not quality_passes(value, bands):
            issues.append('fresh physical quality gate failed')
        sectors = 'sectors' in mode
        expected_arithmetic = ('pt_real_te' if mode in ('pt', 'pt-sectors')
                               else 'fourier_complex' if mode == 'fourier-sectors' else 'spatial_complex')
        if (policy['arithmetic'] != expected_arithmetic
                or policy['sectors'] != ('adaptive_group' if sectors else 'off')
                or policy['plane_group'] != {'gamma': 'p4m', 'rotation': 'p4', 'mirror': 'pm'}[case]
                or policy['sector_guard_budget'] != (2 if sectors else None)
                or policy['reconstruct_partners'] is not (sectors and case == 'gamma')
                or policy['sector_storage'] != 'full_grid' or policy['sector_kernel'] != 'full_grid'
                or policy['correction_projection_interval'] != 1
                or policy['k_reuse'] != 'off' or policy['share_sector_workspace'] is not False
                or policy['max_sector_workspace_bytes'] != 512*1024*1024):
            issues.append('requested policy differs')
        for index, (point, spectrum) in enumerate(zip(value['points'], value['spectrum'])):
            if any(y < x-1e-9 for x, y in zip(spectrum, spectrum[1:])):
                issues.append(f'point {index}: unordered spectrum')
            bits = point['field_bits']
            if (len(bits) != bands or any(len(f) != rows for f in bits)
                    or any(len(z) != 2 or any(type(b) is not int or not 0 <= b < 2**64
                               or (b >> 52) & 0x7ff == 0x7ff for b in z) for f in bits for z in f)):
                issues.append(f'point {index}: incomplete/nonfinite field bits')
            report = point['symmetry']; attempts = report['solver_attempts']
            if (report['source_index'] is not None or len(attempts) != 1
                    or attempts[0]['method'] != ('sectors' if sectors else 'fourier' if mode == 'pt' else 'unrestricted')
                    or attempts[0]['retained'] is not True or attempts[0]['rejection'] is not None):
                issues.append(f'point {index}: requested solve was not retained alone')
            total = sum(a['iterations'] for a in attempts)
            if total != point['iterations'] or total != report['attempted_iterations'] or total != report['retained_iterations']:
                issues.append(f'point {index}: attempted-work ledger differs')
            if not sectors:
                if report['sectors'] is not None:
                    issues.append(f'point {index}: unexpected sector result')
                c = report['coordinates']
                if mode == 'pt':
                    if (not c or c['representation'] != 'pt_real_te'
                            or c['bytes_per_column'] != ((rows+1)//2)*16
                            or c['qualification']['authorized'] is not True):
                        issues.append(f'point {index}: requested PT was not used')
                elif c is not None:
                    issues.append(f'point {index}: unexpected coordinate solve')
                continue
            s = report['sectors']
            if not s:
                issues.append(f'point {index}: no retained sector output'); continue
            if (s['finite_dimension'] != rows or s['sum_sector_dimensions'] != rows
                    or s['kernel_owners'] != int(point['k'] == [0., 0.])
                    or s['requested'] != bands or s['computed'] < bands
                    or sum(r['dimension']*r['partner_rows'] for r in s['sectors']) != rows):
                issues.append(f'point {index}: sector dimension/multiplicity/kernel mismatch')
            memory = s['workspace']
            if (not memory or memory['budget_bytes'] != policy['max_sector_workspace_bytes']
                    or memory['peak_estimated_payload_bytes'] > memory['budget_bytes']):
                issues.append(f'point {index}: missing/invalid memory admission')
            real = complex_fallbacks = 0
            for r in s['sectors']:
                if (not 0 < r['prefix'] <= r['dimension'] or r['partner_rows'] < 1
                        or not math.isfinite(r['leakage']) or not 0 <= r['leakage'] <= 1e-8):
                    issues.append(f'point {index}: invalid sector prefix/leakage')
                c = r.get('coordinates')
                if mode == 'sectors':
                    if c is not None or r['vector_column_bytes'] != rows*16:
                        issues.append(f'point {index}: spatial sector coordinates changed')
                elif require_combined:
                    if not c:
                        issues.append(f'point {index}: Fourier sector request was ignored'); continue
                    representation = c['representation']
                    if representation == 'pt_real_te':
                        real += 1
                        if (mode != 'pt-sectors' or c['bytes_per_column'] != ((rows+1)//2)*16
                                or c['qualification']['authorized'] is not True
                                or r.get('coordinate_rejection') is not None
                                or not 0 <= r['pt_projector_defect'] <= 1e-10):
                            issues.append(f'point {index}: invalid packed sector authorization')
                    elif representation == 'fourier_complex':
                        if c['bytes_per_column'] != rows*16:
                            issues.append(f'point {index}: invalid complex sector size')
                        if mode == 'pt-sectors':
                            complex_fallbacks += 1
                            if not r.get('coordinate_rejection') or not r.get('pt_projector_defect', 0) > 1e-10:
                                issues.append(f'point {index}: unexplained complex fallback')
                    else:
                        issues.append(f'point {index}: wrong coordinate representation')
                    if (r['vector_column_bytes'] != c['bytes_per_column']
                            or not math.isfinite(c['final_sector_projection_defect'])
                            or c['maximum_real_subspace_leakage'] > 1e-10):
                        issues.append(f'point {index}: invalid coordinate accounting/leakage')
            if require_combined and mode == 'pt-sectors':
                if real == 0 or (case == 'rotation' and complex_fallbacks != 2):
                    issues.append(f'point {index}: PT/complex sector selection differs')
    except (KeyError, TypeError, ValueError, IndexError) as error:
        issues.append('malformed result: '+str(error))
    return issues


def same_pencil(a, b):
    try:
        return (a['physical_configuration'] == b['physical_configuration']
                and len(a['points']) == len(b['points'])
                and all(x['operator_identity'] == y['operator_identity'] and x['k'] == y['k']
                        for x, y in zip(a['points'], b['points'])))
    except (KeyError, TypeError):
        return False


def spectrum_error(a, b):
    return max(abs(x-y)/max(1.0, abs(x), abs(y))
               for xrow, yrow in zip(a['spectrum'], b['spectrum']) for x, y in zip(xrow, yrow))


def independent(value, reference=None):
    """Run only after timing has finished; all complete stored fields are used."""
    from support_contract import independent_certificate, certificate_issues, compare
    checks, comparisons, issues = [], [], []
    for i, point in enumerate(value['points']):
        n = dict(operator_identity=point['operator_identity'], grid=value['grid'], polarization=value['polarization'])
        fields = dict(field_bits=point['field_bits'], spectrum=[x*x for x in value['spectrum'][i]])
        c = independent_certificate(n, fields, projected=False)
        checks.append(c)
        issues += [f'point {i}: '+p for p in certificate_issues(c, value['bands'], point['k'] == [0., 0.], 'independent FFT')]
        if reference is not None:
            other = dict(field_bits=reference['points'][i]['field_bits'], spectrum=[x*x for x in reference['spectrum'][i]])
            compared = compare(n, fields, other); comparisons.append(compared)
            if (compared['max_scaled_frequency_error'] > 1e-6 or compared['guarded_prefix'] <= 0
                    or compared['max_sin_angle'] > 1e-3):
                issues.append(f'point {i}: frequency or guarded physical subspace mismatch')
    return dict(checks=checks, comparisons=comparisons, issues=issues)
