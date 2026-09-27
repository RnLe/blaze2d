"""Only coarse initialization may differ in a public-workflow comparison."""
from interface_contract import physical_configuration


def configuration(result):
    normalized = physical_configuration(result)
    if normalized is None:
        return None
    requested = result['configuration']['bands']['path'].get('points')
    if (not requested or len(requested) != result['expected_points']
            or [point['k'] for point in result['points']] != requested):
        return None
    # Unlike the older PT/reuse factorial, these policies are fixed controls.
    original = result['configuration']['bands']['symmetry']
    symmetry = normalized['bands']['symmetry']
    for key in ('arithmetic', 'k_reuse'):
        symmetry[key] = original[key]
    symmetry.pop('coarse_start', None)
    return normalized


def same_jobs(a, b):
    ca, cb = configuration(a), configuration(b)
    return (ca is not None and ca == cb
            and [p['k'] for p in a['points']] == [p['k'] for p in b['points']])


def path_issues(result):
    issues = []
    try:
        enabled = result['mode'] in ('coarse', 'pt-coarse')
        if result['mode'] not in ('off', 'pt', 'coarse', 'pt-coarse'):
            return ['undeclared initialization control']
        config = result['configuration']['bands']['symmetry'].get('coarse_start')
        n = result['grid'][0]
        half = (n - 1) // 2
        coarse = half - int(half % 2 == 0)
        expected = {'resolution': [coarse, coarse], 'max_seed_field_bytes': 67108864}
        if (config != expected if enabled else config is not None):
            issues.append('coarse configuration differs from frozen policy')
        for index, point in enumerate(result['points']):
            report = point['symmetry']
            attempts = report['solver_attempts']
            if (point['iterations'] != sum(a['iterations'] for a in attempts)
                    or report['attempted_iterations'] != point['iterations']
                    or sum(a['retained'] is True for a in attempts) != 1
                    or report['retained_iterations'] != sum(a['iterations'] for a in attempts if a['retained'])):
                issues.append(f'{index}: incomplete solve-work ledger')
            seed = report.get('coarse_start')
            sources = [a for a in attempts if a['method'] == 'coarse_initializer']
            if enabled and index == 0:
                if (seed is None or seed['used_as_seed'] is not True or seed['rejection'] is not None
                        or seed['resolution'] != expected['resolution']
                        or seed['max_seed_field_bytes'] != expected['max_seed_field_bytes']
                        or seed['reserved_seed_field_bytes'] != 16*result['bands']*(coarse*coarse+2*n*n)
                        or not 0 < seed['reserved_seed_field_bytes'] <= seed['max_seed_field_bytes']
                        or seed['iterations'] <= 0 or len(sources) != 1
                        or sources[0]['iterations'] != seed['iterations'] or sources[0]['retained']):
                    issues.append('first point did not use the declared accounted initializer')
            elif seed is not None or sources:
                issues.append(f'{index}: unexpected coarse preparation')
    except (KeyError, TypeError, ValueError):
        issues.append('malformed coarse workflow report')
    return issues
