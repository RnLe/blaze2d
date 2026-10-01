"""Shared process control, provenance and output reduction for preconditioner studies.

One case runs as `blaze2d run case.toml --threads 1 --quiet -o out.json` with all
library thread pools at one, optionally under nice and taskset (both exec, so the
reaped pid is the solver and wait4 reports its own resource usage). The output is
archived losslessly under objects/<sha256>.json.gz and reduced to a compact record.
"""
from __future__ import annotations

import datetime
import functools
import gzip
import hashlib
import json
import math
import os
import platform
import queue
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Iterable

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / 'results'
FROZEN = HERE / 'bin' / 'e054876'
sys.path.insert(0, str(REPO / 'benchmarks' / 'symmetry'))

import cases  # noqa: E402

RECORD_SCHEMA = 'blaze2d-lab/preconditioner/record/1'
MANIFEST_SCHEMA = 'blaze2d-lab/preconditioner/manifest/1'
THREAD_ENV = dict.fromkeys(
    ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'RAYON_NUM_THREADS'), '1')
RELATIVE_LIMIT = 1e-5
ABSOLUTE_LIMIT = 1e-7
ORTHOGONALITY_LIMIT = 1e-8
ZERO_EIGENVALUE = 1e-7
# Failures that void a run; the remaining flags are accuracy certificates of the variant.
HARD_FLAGS = frozenset({'timeout', 'exit', 'no_output', 'status', 'job_error', 'nonconverged',
                        'no_points', 'no_frequencies'})
OPTIONAL_METADATA = ('point_seconds', 'work', 'preconditioner', 'eigensolver')
BINARIES = {
    'frozen': ((FROZEN / 'blaze2d', REPO / 'target/release/blaze2d'),
               (FROZEN / 'blaze2d-profiling', REPO / 'target-profiling/release/blaze2d')),
    'current': ((REPO / 'target/release/blaze2d',),
                (REPO / 'target-profiling/release/blaze2d',)),
}

_ACTIVE: set[subprocess.Popen] = set()
_ACTIVE_LOCK = threading.Lock()
INTERRUPTED = threading.Event()
TIME_AVAILABLE = Path('/usr/bin/time').is_file()


def raw_clock() -> float:
    return time.clock_gettime(time.CLOCK_MONOTONIC_RAW)


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def resolve_binary(spec: str | None, profiling: bool = False) -> Path:
    """'frozen' prefers bin/e054876 and falls back to the workspace build; else a path."""
    spec = spec or 'current'
    if spec not in BINARIES:
        path = Path(spec).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f'binary not found: {path}')
        return path
    candidates = BINARIES[spec][profiling]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f'no {spec} binary at ' + ', '.join(map(str, candidates)))


@functools.lru_cache(maxsize=None)
def _file_digest(path: str, size: int, mtime: int) -> str:
    with open(path, 'rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def file_digest(path: Path) -> str:
    stat = Path(path).stat()
    return _file_digest(str(path), stat.st_size, stat.st_mtime_ns)


def binary_info(path: Path, spec: str | None = None) -> dict[str, Any]:
    path = Path(path).resolve()
    try:
        name = str(path.relative_to(REPO))
    except ValueError:
        name = str(path)
    info = dict(name=name, path=str(path), sha256=file_digest(path), spec=spec, revision=None)
    if spec in BINARIES:
        info['fallback'] = path not in (BINARIES[spec][0][0], BINARIES[spec][1][0])
    sidecar = path.with_name(path.name + '.provenance.json')
    if sidecar.is_file():
        provenance = json.loads(sidecar.read_text())
        if provenance.get('binary_sha256') not in (None, info['sha256']):
            raise ValueError(f'{path} differs from its provenance record {sidecar.name}')
        info['revision'] = provenance.get('source_revision') or provenance.get('revision')
    return info


def variant_binary(variant: dict[str, Any], override: str | None = None,
                   profiling: bool = False) -> dict[str, Any]:
    spec = override or variant.get('binary') or 'current'
    return binary_info(resolve_binary(spec, profiling), spec)


def git_state() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True,
                              text=True, check=False).stdout.strip()
    return dict(branch=git('rev-parse', '--abbrev-ref', 'HEAD'), head=git('rev-parse', 'HEAD'),
                dirty=bool(git('status', '--porcelain', '--untracked-files=no')))


def host_info() -> dict[str, Any]:
    model = next((line.split(':', 1)[1].strip() for line in
                  Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), None)
    memory = next(int(line.split()[1]) * 1024 for line in
                  Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemTotal'))
    return dict(node=platform.node(), kernel=platform.release(), cpu_model=model,
                logical_cpus=os.cpu_count(), memory_bytes=memory, python=platform.python_version(),
                load_average=os.getloadavg())


def manifest(study: str, kind: str, stage: str, subset: str, variants: dict[str, dict],
             binaries: dict[str, dict], args: dict[str, Any]) -> dict[str, Any]:
    return dict(schema=MANIFEST_SCHEMA, study=study, kind=kind, stage=stage, subset=subset,
                variants=variants, binaries=binaries, git=git_state(), host=host_info(),
                args=args, started=utc_now(), ended=None, status='running', pid=os.getpid())


def cpu_list(text: str | None, count: int) -> list[int]:
    """Parse '16-23,30' or choose one CPU per virtual core away from CPUs 0-3."""
    if text:
        cpus: list[int] = []
        for part in text.split(','):
            low, _, high = part.partition('-')
            cpus.extend(range(int(low), int(high or low) + 1))
        return cpus
    total = os.cpu_count() or 1
    order = [*range(16, total, 2), *range(4, 16, 2), *range(17, total, 2), *range(5, 16, 2), 0, 1, 2, 3]
    order = list(dict.fromkeys(c for c in order if c < total))
    if count > len(order):
        raise ValueError(f'{count} workers exceed {len(order)} CPUs')
    return order[:count]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not Path(path).is_file():
        return []
    rows = []
    for line in Path(path).read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            continue  # A torn final line from an interrupted writer.
    return rows


class JsonlWriter:
    def __init__(self, path: Path):
        self.path, self.lock = Path(path), threading.Lock()

    def write(self, row: dict[str, Any]) -> None:
        line = json.dumps(row, separators=(',', ':')) + '\n'
        with self.lock, self.path.open('a') as stream:
            stream.write(line)


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(data, indent=1) + '\n')
    temporary.replace(path)


def remove_scratch(root: Path) -> None:
    """Drop the per-run scratch directory once no run uses it (kept if anything remains)."""
    try:
        (Path(root) / 'tmp').rmdir()
    except OSError:
        pass


def archive(objects: Path, data: bytes, suffix: str = '.json.gz') -> str:
    digest = hashlib.sha256(data).hexdigest()
    target = Path(objects) / f'{digest}{suffix}'
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f'.{target.name}.{threading.get_ident()}.tmp')
        temporary.write_bytes(gzip.compress(data, mtime=0))
        temporary.replace(target)
    return digest


def proc_cpu_seconds(pid: int) -> float:
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    except (OSError, IndexError):
        return 0.0
    return (int(fields[11]) + int(fields[12])) / os.sysconf('SC_CLK_TCK')


def owned_snapshot(host: Any, owned_pids: Iterable[int] = ()) -> dict[str, Any]:
    """QuietHost snapshot that also owns the CPU of live helper processes (the sampler)."""
    snapshot = host.snapshot()
    snapshot['owned_cpu_seconds'] += sum(proc_cpu_seconds(pid) for pid in owned_pids)
    return snapshot


def _kill_group(process: subprocess.Popen) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def terminate_active() -> None:
    """Kill live solver processes; their callers must not record the killed runs."""
    INTERRUPTED.set()
    with _ACTIVE_LOCK:
        for process in list(_ACTIVE):
            _kill_group(process)


def run_process(command: list[Any], stderr_path: Path, *, cpu: int | None = None, nice: int = 0,
                timeout: float = 1200, host: Any = None, owned_pids: Iterable[int] = ()) -> dict[str, Any]:
    """Run in its own process group; GNU time reports the solver's peak RSS without the
    forking Python's pages, which wait4 would otherwise include."""
    rss_path = stderr_path.with_suffix('.maxrss')
    prefix = ['/usr/bin/time', '-f', '%M', '-o', str(rss_path)] if TIME_AVAILABLE else []
    prefix += (['nice', '-n', str(nice)] if nice else []) + (['taskset', '-c', str(cpu)] if cpu is not None else [])
    full = prefix + [str(part) for part in command]
    owned_pids = tuple(owned_pids)
    before = owned_snapshot(host, owned_pids) if host else None
    expired = threading.Event()
    with stderr_path.open('wb') as stderr:
        start = raw_clock()
        process = subprocess.Popen(full, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=stderr,
                                   env=os.environ | THREAD_ENV, start_new_session=True)
        with _ACTIVE_LOCK:
            _ACTIVE.add(process)
        timer = threading.Timer(timeout, lambda: (expired.set(), _kill_group(process)))
        timer.daemon = True
        timer.start()
        try:
            _, status, usage = os.wait4(process.pid, 0)
        except BaseException:
            _kill_group(process)
            process.wait()
            raise
        finally:
            timer.cancel()
            with _ACTIVE_LOCK:
                _ACTIVE.discard(process)
        end = raw_clock()
    process.returncode = os.waitstatus_to_exitcode(status)
    after = owned_snapshot(host, owned_pids) if host else None
    try:
        max_rss, source = int(rss_path.read_text().split()[-1]), 'gnu_time'
    except (OSError, ValueError, IndexError):
        max_rss, source = usage.ru_maxrss, 'wait4_including_fork'
    rss_path.unlink(missing_ok=True)
    return dict(command=full, returncode=process.returncode, timed_out=expired.is_set(),
                raw_start=start, raw_end=end, wall_seconds=end - start,
                rusage=dict(user_seconds=usage.ru_utime, system_seconds=usage.ru_stime,
                            max_rss_kib=max_rss, max_rss_source=source, minor_faults=usage.ru_minflt,
                            major_faults=usage.ru_majflt, voluntary_switches=usage.ru_nvcsw,
                            involuntary_switches=usage.ru_nivcsw),
                snapshots=[before, after] if host else None)


def array_digest(array: dict[str, Any]) -> str:
    header = json.dumps({'shape': array['shape'], 'dtype': array['dtype']}, separators=(',', ':'))
    data = array['data']
    return hashlib.sha256(header.encode() + struct.pack(f'<{len(data)}d', *data)).hexdigest()


def _rows(array: dict[str, Any] | None) -> list[list[float]] | None:
    if not array or len(array.get('shape', ())) != 2:
        return None
    rows, width = array['shape']
    data = array['data']
    return [data[i * width:(i + 1) * width] for i in range(rows)]


def _symmetry_summary(points: list[dict[str, Any]]) -> dict[str, Any]:
    methods: Counter[str] = Counter()
    representations: Counter[str] = Counter()
    rejected = candidates = qualification = reused = 0
    for point in points:
        attempts = point.get('solver_attempts') or []
        methods.update(a['method'] for a in attempts if a.get('retained'))
        rejected += sum(not a.get('retained') for a in attempts)
        candidates += len(point.get('rejected') or [])
        qualification += point.get('qualification_attempts') or 0
        if point.get('source_index') is not None and not attempts:
            reused += 1
        representation = (point.get('coordinates') or {}).get('representation')
        if representation:
            representations[representation] += 1
    return dict(methods=dict(methods), rejected_attempts=rejected, rejected_candidates=candidates,
                qualification_attempts=qualification, reused_points=reused,
                representations=dict(representations))


def _residual_summary(arrays: dict[str, Any]) -> dict[str, Any]:
    frequencies, relative, absolute, k = (
        _rows(arrays.get(n)) for n in ('frequencies', 'residuals', 'absolute_residuals', 'k_points'))
    if not (frequencies and relative and absolute):
        return dict(max_relative_nonzero=None, max_absolute_zero_modes=None, zero_modes=0, finite=False)
    worst_relative = worst_absolute = 0.0
    zero_modes, finite = 0, True
    for row, (fs, rs, rabs) in enumerate(zip(frequencies, relative, absolute, strict=True)):
        gamma = k is None or all(v == 0.0 for v in k[row])
        for f, r, a in zip(fs, rs, rabs, strict=True):
            finite &= all(map(math.isfinite, (f, r, a)))
            if gamma and (math.tau * f) ** 2 <= ZERO_EIGENVALUE:
                zero_modes += 1
                worst_absolute = max(worst_absolute, a)
            else:
                worst_relative = max(worst_relative, r)
    return dict(max_relative_nonzero=worst_relative, max_absolute_zero_modes=worst_absolute,
                zero_modes=zero_modes, finite=finite)


def _optional(metadata: dict[str, Any]) -> dict[str, Any]:
    found = {key: metadata[key] for key in OPTIONAL_METADATA if metadata.get(key) is not None}
    if isinstance(found.get('work'), list):
        totals: Counter[str] = Counter()
        for item in found['work']:
            totals.update({k: v for k, v in item.items() if isinstance(v, (int, float))})
        found['work'] = dict(totals)
    trace = metadata.get('trace')
    if trace is not None:
        found['trace'] = dict(present=True, length=len(trace) if isinstance(trace, (list, dict)) else None)
    return found


def reduce_output(document: dict[str, Any] | None, include_frequencies: bool = True) -> dict[str, Any]:
    """Compact, schema-tolerant summary of one single-job run document."""
    if not document:
        return dict(status=None, points=0, iterations=[], iterations_total=None, converged=False)
    statistics = document.get('statistics') or {}
    errors = document.get('errors') or []
    results = document.get('results') or []
    result = results[0] if results else next(
        (e['partial_result'] for e in errors if e.get('partial_result')), None)
    reduced: dict[str, Any] = dict(
        status=statistics.get('status'), study_seconds=statistics.get('elapsed_seconds'),
        job_error='; '.join(str(e.get('diagnostic')) for e in errors) or None)
    if result is None:
        return dict(reduced, points=0, iterations=[], iterations_total=None, converged=False)
    metadata, arrays = result.get('metadata') or {}, result.get('arrays') or {}
    iterations = metadata.get('iterations') or []
    converged = metadata.get('converged_per_point') or []
    certification = metadata.get('certification') or {}
    execution = metadata.get('execution') or {}
    frequencies = arrays.get('frequencies')
    reduced.update(
        source_revision=(metadata.get('build') or {}).get('source_revision'),
        job_seconds=metadata.get('elapsed_seconds'),
        points=len(iterations), iterations=iterations, iterations_total=sum(iterations),
        converged=bool(metadata.get('converged')),
        nonconverged_points=[i for i, ok in enumerate(converged) if not ok],
        stopping_criterion=metadata.get('stopping_criterion'),
        solver_block_size=metadata.get('solver_block_size'),
        certification={k: certification.get(k) for k in
                       ('max_residual', 'max_absolute_residual', 'b_orthogonality_defect')},
        execution=dict(
            attempted_iterations=execution.get('attempted_iterations'),
            retained_iterations=execution.get('retained_iterations'),
            rejected_solver_attempts=execution.get('rejected_solver_attempts'),
            reused_points=execution.get('reused_points'),
            fallback_or_declined_points=execution.get('fallback_or_declined_points'),
            stopping_converged=execution.get('stopping_converged'),
            resolved_arithmetic=(execution.get('resolved') or {}).get('arithmetic'),
            retained_solve_methods=execution.get('retained_solve_methods'),
            text=execution.get('text')),
        symmetry=_symmetry_summary((metadata.get('symmetry') or {}).get('points') or []),
        residuals=_residual_summary(arrays),
        frequencies_sha256=array_digest(frequencies) if frequencies else None,
        optional=_optional(metadata))
    if include_frequencies:
        reduced['frequencies'] = _rows(frequencies)
    return reduced


def validate(record: dict[str, Any]) -> dict[str, Any]:
    flags = []
    if record.get('timed_out'):
        flags.append('timeout')
    if record.get('returncode'):
        flags.append(f"exit:{record['returncode']}")
    if not record.get('output_sha256'):
        flags.append('no_output')
    elif record.get('status') != 'completed':
        flags.append(f"status:{record.get('status')}")
    if record.get('job_error'):
        flags.append('job_error')
    if record.get('points'):
        if record.get('nonconverged_points'):
            flags.append(f"nonconverged:{len(record['nonconverged_points'])}")
        residuals = record.get('residuals') or {}
        if not residuals.get('finite'):
            flags.append('nonfinite_or_missing_residuals')
        if (residuals.get('max_relative_nonzero') or 0) > RELATIVE_LIMIT:
            flags.append('residual>1e-5')
        if (residuals.get('max_absolute_zero_modes') or 0) > ABSOLUTE_LIMIT:
            flags.append('zero_mode_absolute>1e-7')
        defect = (record.get('certification') or {}).get('b_orthogonality_defect')
        if defect is None or not math.isfinite(defect) or defect > ORTHOGONALITY_LIMIT:
            flags.append('b_orthogonality>1e-8')
        if not record.get('frequencies_sha256'):
            flags.append('no_frequencies')
    elif record.get('output_sha256'):
        flags.append('no_points')
    hard = [f for f in flags if f.split(':')[0] in HARD_FLAGS]
    return dict(ok=not flags, completed=not hard, flags=flags)


def execute_case(case: cases.Case, variant_name: str, variant: dict[str, Any],
                 binary: dict[str, Any], workdir: Path, objects: Path, *, cpu: int | None = None,
                 nice: int = 19, timeout: float = 1200, host: Any = None,
                 owned_pids: Iterable[int] = (), profile: bool = False,
                 include_frequencies: bool = True, tag: str = 'work',
                 ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    """Run one case; return (record, process run, parsed profile or None)."""
    workdir.mkdir(parents=True, exist_ok=True)
    text = cases.render(case, variant)
    stem = workdir / f'{case.label}.{variant_name}.{threading.get_ident()}'
    paths = {k: Path(f'{stem}.{k}') for k in ('toml', 'json', 'stderr', 'profile.json')}
    paths['toml'].write_text(text)
    command = [binary['path'], 'run', paths['toml'], '--threads', '1', '--quiet', '-o', paths['json'],
               *variant.get('cli_args', [])]
    if profile:
        command += ['--profile-output', paths['profile.json']]
    started = utc_now()
    run = run_process(command, paths['stderr'], cpu=cpu, nice=nice, timeout=timeout, host=host,
                      owned_pids=owned_pids)
    raw = paths['json'].read_bytes() if paths['json'].exists() else b''
    try:
        document = json.loads(raw) if raw else None
    except ValueError:
        document = None
    stderr = paths['stderr'].read_text(errors='replace')[-4000:]
    profile_document = None
    if profile and paths['profile.json'].exists():
        profile_bytes = paths['profile.json'].read_bytes()
        profile_document = json.loads(profile_bytes)
        profile_document['object_sha256'] = archive(objects, profile_bytes)
    for path in paths.values():
        path.unlink(missing_ok=True)
    record: dict[str, Any] = dict(
        schema=RECORD_SCHEMA, case_id=case.id, label=case.label, params=case.params,
        variant=variant_name, tag=tag,
        binary={k: binary.get(k) for k in ('name', 'sha256', 'revision')},
        config_sha256=archive(objects, text.encode(), '.toml.gz'), cpu=cpu, started=started,
        returncode=run['returncode'], timed_out=run['timed_out'], wall_seconds=run['wall_seconds'],
        raw_start=run['raw_start'], raw_end=run['raw_end'], rusage=run['rusage'],
        output_sha256=archive(objects, raw) if raw else None,
        stderr_tail=stderr if run['returncode'] or document is None else None)
    record.update(reduce_output(document, include_frequencies))
    if record.get('source_revision') and not record['binary']['revision']:
        record['binary']['revision'] = record['source_revision']
    record['validation'] = validate(record)
    return record, run, profile_document


def run_pool(tasks: list[Any], cpus: list[int], work: Callable[[Any, int], None]) -> None:
    """Run tasks on one pinned worker per CPU; Ctrl-C kills live solver processes."""
    pending: queue.SimpleQueue = queue.SimpleQueue()
    for task in tasks:
        pending.put(task)
    stop = threading.Event()

    def worker(cpu: int) -> None:
        while not stop.is_set():
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            try:
                work(task, cpu)
            except Exception:  # Keep the pool alive; the missing record makes the task resumable.
                traceback.print_exc()

    threads = [threading.Thread(target=worker, args=(cpu,), daemon=True) for cpu in cpus]
    for thread in threads:
        thread.start()
    try:
        while any(thread.is_alive() for thread in threads):
            for thread in threads:
                thread.join(0.5)
    except KeyboardInterrupt:
        stop.set()
        terminate_active()
        raise
