#!/usr/bin/env python3
"""Concurrent host-load sampler and per-run contamination verdicts for timing studies.

`sample` appends JSONL readings on CLOCK_MONOTONIC_RAW: Linux /proc counters (total and
per-CPU busy ticks, CPU and memory pressure, load, swap, MemAvailable, the sampler's own
CPU) every period and Windows GetSystemTimes through the persistent host_clock.ps1
collector every second. `verdict()` admits one run window from those readings, with
quiet_host.assess on bracketing snapshots when the run spans fewer than two readings.
Background CPU excludes the benchmark CPUs (or the child's CPU time) and the sampler.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import runlib
from quiet_host import QuietHost, assess

LIMITS = dict(linux_mean=0.5, linux_peak=1.5, windows_mean=1.5, windows_peak=3.0,
              psi_cpu=0.02, on_cpu=0.98, memory_available=2 * 1024**3)
BACKGROUND_REASONS = frozenset({'linux_mean', 'linux_peak', 'windows_mean', 'windows_peak', 'psi_cpu', 'swap'})
# Stray swap traffic of other processes (WSL pages idle processes in and out): up to this
# many pages it is background noise like the CPU limits; more, or any full memory stall,
# stays a hard failure.
SWAP_TOLERANCE_PAGES = 1024
TICKS = os.sysconf('SC_CLK_TCK')
HERE = Path(__file__).resolve().parent


def _pressure(path: str) -> dict[str, int]:
    values = {}
    for line in Path(path).read_text().splitlines():
        kind, *fields = line.split()
        values[kind] = int(next(f for f in fields if f.startswith('total=')).split('=')[1])
    return values


def linux_reading() -> dict[str, Any]:
    stat = Path('/proc/stat').read_text()
    t = runlib.raw_clock()
    busy = []
    for line in stat.splitlines():
        if not line.startswith('cpu'):
            break
        fields = list(map(int, line.split()[1:9]))
        busy.append(sum(fields) - fields[3] - fields[4])
    vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
    available = next(int(line.split()[1]) * 1024 for line in Path('/proc/meminfo').read_text().splitlines()
                     if line.startswith('MemAvailable:'))
    cpu, memory = _pressure('/proc/pressure/cpu'), _pressure('/proc/pressure/memory')
    own = resource.getrusage(resource.RUSAGE_SELF)
    return dict(kind='linux', t=t, total_busy=busy[0], cpu_busy=busy[1:],
                psi_cpu_some_us=cpu['some'], psi_memory_some_us=memory['some'],
                psi_memory_full_us=memory['full'], loadavg=os.getloadavg(),
                pswpin=int(vm['pswpin']), pswpout=int(vm['pswpout']), mem_available=available,
                self_cpu=own.ru_utime + own.ru_stime)


def windows_reading(host: QuietHost) -> dict[str, Any]:
    before = runlib.raw_clock()
    host.process.stdin.write('\n')
    host.process.stdin.flush()
    line = host.process.stdout.readline()
    after = runlib.raw_clock()
    if not line:
        raise RuntimeError('Windows clock unavailable: ' + host.process.stderr.read()[-2000:])
    value = json.loads(line)
    return dict(kind='windows', t=(before + after) / 2, latency=after - before,
                busy=value['busy_seconds'], collector=value['collector_cpu_seconds'],
                qpc=value['qpc_ticks'], qpc_frequency=value['qpc_frequency'])


def sample(out: Path, period: float = 0.25, windows_period: float = 1.0,
           duration: float | None = None, windows: bool = True) -> None:
    stopping = []
    for number in (signal.SIGTERM, signal.SIGINT):
        signal.signal(number, lambda *_: stopping.append(True))
    host = QuietHost() if windows else None
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('a') as stream:
        def emit(row: dict[str, Any]) -> None:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
            stream.flush()

        start = runlib.raw_clock()
        emit(dict(kind='header', t=start, pid=os.getpid(), period=period, windows_period=windows_period,
                  ticks=TICKS, cpus=os.cpu_count(), affinity=sorted(os.sched_getaffinity(0)),
                  windows=windows, started=runlib.utc_now()))
        due = next_windows = start
        while not stopping and (duration is None or runlib.raw_clock() - start < duration):
            reading = linux_reading()
            emit(reading)
            if host and reading['t'] >= next_windows:
                emit(windows_reading(host))
                next_windows = max(next_windows + windows_period, runlib.raw_clock())
            due = max(due + period, runlib.raw_clock())
            time.sleep(max(0.0, due - runlib.raw_clock()))
    if host:
        host.close()


class SampleReader:
    """Incremental reader of a growing sampler file."""

    def __init__(self, path: Path):
        self.path, self.offset, self.samples = Path(path), 0, []

    def poll(self) -> list[dict[str, Any]]:
        if self.path.exists():
            with self.path.open('rb') as stream:
                stream.seek(self.offset)
                data = stream.read()
            complete = data[:data.rfind(b'\n') + 1]
            self.offset += len(complete)
            self.samples.extend(json.loads(line) for line in complete.splitlines() if line.strip())
        return self.samples

    def prune(self, before: float) -> None:
        self.samples = [s for s in self.samples if s['kind'] == 'header' or s['t'] >= before]

    def wait_for(self, kind: str, after: float, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            if any(s['kind'] == kind and s['t'] >= after for s in reversed(self.poll()[-64:])):
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.05)


def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    return max(0.0, min(a1, b1) - max(a0, b0))


def _nearest(series: list[dict[str, Any]], t: float) -> dict[str, Any]:
    return min(series, key=lambda s: abs(s['t'] - t))


def _owned_linux(a: dict, b: dict, bench: set[int] | None, child: tuple | None, self_owned: bool) -> float:
    """Linux CPU seconds between readings that belong to the measurement, not background."""
    if bench:
        owned = sum(b['cpu_busy'][i] - a['cpu_busy'][i] for i in bench) / TICKS
    elif child:
        cpu, t0, t1 = child
        owned = cpu * _overlap(a['t'], b['t'], t0, t1) / max(t1 - t0, 1e-9)
    else:
        owned = 0.0
    return owned + (b['self_cpu'] - a['self_cpu'] if self_owned else 0.0)


def _linux_background(a: dict, b: dict, bench: set[int] | None, child: tuple | None, self_owned: bool) -> float:
    if bench:
        busy = sum(b['cpu_busy'][i] - a['cpu_busy'][i] for i in range(len(a['cpu_busy'])) if i not in bench) / TICKS
        busy -= b['self_cpu'] - a['self_cpu'] if self_owned else 0.0
    else:
        busy = (b['total_busy'] - a['total_busy']) / TICKS - _owned_linux(a, b, None, child, self_owned)
    return busy / (b['t'] - a['t'])


def _peak(series: list[dict[str, Any]], rate) -> float:
    """Largest rate over windows of at least one second (the whole span if shorter)."""
    rates, j = [], 0
    for i, start in enumerate(series):
        j = max(j, i + 1)
        while j < len(series) and series[j]['t'] - start['t'] < 0.999:
            j += 1
        if j == len(series):
            break
        rates.append(rate(start, series[j]))
    return max(rates) if rates else rate(series[0], series[-1])


def _fallback(snapshots, t0: float, t1: float, child_cpu: float | None, limits: dict) -> dict[str, Any]:
    if not snapshots or None in snapshots:
        return dict(verdict='unknown', reasons=['evidence: run spans fewer than two readings and no snapshots'],
                    metrics=dict(method='none'))
    result = assess(*snapshots, limit=limits['linux_mean'], windows_limit=limits['windows_mean'])
    metrics = dict(method='assess', elapsed=t1 - t0,
                   linux_background_mean=result.get('linux_background_cores'),
                   windows_background_mean=result.get('windows_background_cores'))
    if 'linux_background_cores' not in result:
        return dict(verdict='unknown', reasons=['evidence: ' + i for i in result['issues']], metrics=metrics)
    codes = {'linux': 'linux_mean', 'windows': 'windows_mean'}
    reasons = [f"{codes.get(i.split()[0], 'memory')}: {i}" for i in result['issues']]
    if child_cpu is not None:
        metrics['on_cpu_ratio'] = child_cpu / (t1 - t0)
        if metrics['on_cpu_ratio'] < limits['on_cpu']:
            reasons.append(f"on_cpu: {metrics['on_cpu_ratio']:.3f} < {limits['on_cpu']}")
    return dict(verdict='contaminated' if reasons else 'clean', reasons=reasons, metrics=metrics)


def verdict(samples: list[dict[str, Any]], t0: float, t1: float, child_cpu_seconds: float | None = None,
            benchmark_cpus: Iterable[int] | None = None, snapshots=None,
            limits: dict[str, float] = LIMITS) -> dict[str, Any]:
    """Admit the raw-clock window [t0, t1]; returns verdict clean/contaminated/unknown."""
    bench = set(benchmark_cpus) if benchmark_cpus is not None else None
    linux = [s for s in samples if s['kind'] == 'linux']
    inside = [s for s in linux if t0 <= s['t'] <= t1]
    if len(inside) < 2:
        return _fallback(snapshots, t0, t1, child_cpu_seconds, limits)
    header = next((s for s in samples if s['kind'] == 'header'), {})
    self_owned = not (bench and set(header.get('affinity', ())) <= bench)
    child = (child_cpu_seconds, t0, t1) if child_cpu_seconds is not None else None
    a = max((s for s in linux if s['t'] <= t0), key=lambda s: s['t'], default=inside[0])
    b = min((s for s in linux if s['t'] >= t1), key=lambda s: s['t'], default=inside[-1])
    span = [s for s in linux if a['t'] <= s['t'] <= b['t']]
    elapsed = b['t'] - a['t']
    reasons, unknown = [], []
    m: dict[str, Any] = dict(method='sampler', elapsed=elapsed, linux_samples=len(span))
    m['linux_background_mean'] = _linux_background(a, b, bench, child, self_owned)
    m['linux_background_peak_1s'] = _peak(span, lambda x, y: _linux_background(x, y, bench, child, self_owned))
    m['sampler_cores'] = (b['self_cpu'] - a['self_cpu']) / elapsed
    m['psi_cpu_fraction'] = (b['psi_cpu_some_us'] - a['psi_cpu_some_us']) / 1e6 / elapsed
    m['swap_in'], m['swap_out'] = b['pswpin'] - a['pswpin'], b['pswpout'] - a['pswpout']
    m['memory_full_stall_us'] = b['psi_memory_full_us'] - a['psi_memory_full_us']
    m['min_mem_available_bytes'] = min(s['mem_available'] for s in span)
    if bench and child_cpu_seconds is not None:
        m['benchmark_foreign_cores'] = (_owned_linux(a, b, bench, None, False) - child_cpu_seconds) / elapsed
    for code, value, bound in [('linux_mean', m['linux_background_mean'], limits['linux_mean']),
                               ('linux_peak', m['linux_background_peak_1s'], limits['linux_peak']),
                               ('psi_cpu', m['psi_cpu_fraction'], limits['psi_cpu'])]:
        if value > bound:
            reasons.append(f'{code}: {value:.3f} > {bound}')
    for key in ('swap_in', 'swap_out', 'memory_full_stall_us'):
        if m[key]:
            code = 'swap' if key != 'memory_full_stall_us' and m[key] <= SWAP_TOLERANCE_PAGES else 'memory'
            reasons.append(f'{code}: {key} increased by {m[key]}')
    if m['min_mem_available_bytes'] < limits['memory_available']:
        reasons.append(f"memory: MemAvailable {m['min_mem_available_bytes'] / 1024**3:.2f} GiB < 2 GiB")
    if child_cpu_seconds is not None:
        m['on_cpu_ratio'] = child_cpu_seconds / (t1 - t0)
        if m['on_cpu_ratio'] < limits['on_cpu']:
            reasons.append(f"on_cpu: {m['on_cpu_ratio']:.3f} < {limits['on_cpu']}")
    windows = [s for s in samples if s['kind'] == 'windows']
    w0 = max((s for s in windows if s['t'] <= t0), key=lambda s: s['t'], default=None)
    w1 = min((s for s in windows if s['t'] >= t1), key=lambda s: s['t'], default=None)
    if w0 is None or w1 is None:
        unknown.append('evidence: no Windows readings bracket the run')
    else:
        def windows_rate(x: dict, y: dict) -> float:
            owned = _owned_linux(_nearest(linux, x['t']), _nearest(linux, y['t']), bench, child, self_owned)
            qpc = (y['qpc'] - x['qpc']) / x['qpc_frequency']
            return (y['busy'] - x['busy'] - (y['collector'] - x['collector']) - owned) / qpc
        wspan = [s for s in windows if w0['t'] <= s['t'] <= w1['t']]
        qpc = (w1['qpc'] - w0['qpc']) / w0['qpc_frequency']
        raw = w1['t'] - w0['t']
        m['windows_samples'] = len(wspan)
        m['windows_background_mean'] = windows_rate(w0, w1)
        m['windows_background_peak_1s'] = max(windows_rate(x, y) for x, y in zip(wspan, wspan[1:]))
        m['collector_cores'] = (w1['collector'] - w0['collector']) / qpc
        tolerance = .010 + .005 * raw + w0['latency'] + w1['latency']
        if w1['qpc_frequency'] != w0['qpc_frequency'] or abs(raw - qpc) > tolerance:
            unknown.append(f'evidence: Windows QPC {qpc:.4f}s and raw {raw:.4f}s disagree')
        for code, value, bound in [('windows_mean', m['windows_background_mean'], limits['windows_mean']),
                                   ('windows_peak', m['windows_background_peak_1s'], limits['windows_peak'])]:
            if value > bound:
                reasons.append(f'{code}: {value:.3f} > {bound}')
    state = 'contaminated' if reasons else 'unknown' if unknown else 'clean'
    return dict(verdict=state, reasons=reasons + unknown, metrics=m)


def background_only(result: dict[str, Any]) -> bool:
    """True when every failed check is a background-load limit (paired-matched candidates)."""
    return result['verdict'] != 'unknown' and all(r.split(':')[0] in BACKGROUND_REASONS for r in result['reasons'])


def _interval(host: QuietHost, seconds: float, linux: float, windows: float, owned_pids: Iterable[int]):
    start = runlib.owned_snapshot(host, owned_pids)
    deadline = start['monotonic_raw'] + seconds
    while (remaining := deadline - runlib.raw_clock()) > 0:
        time.sleep(remaining)
    end = runlib.owned_snapshot(host, owned_pids)
    return dict(start=start, end=end, **assess(start, end, linux, windows))


def preflight(host: QuietHost, wait: float = 120, seconds: float = 2.0, linux: float = 0.25,
              windows: float = 1.5, owned_pids: Iterable[int] = ()) -> dict[str, Any]:
    """Two consecutive quiet windows, polling until `wait` seconds have passed."""
    owned_pids, windows_seen, streak = tuple(owned_pids), [], 0
    deadline = time.monotonic() + wait
    while True:
        result = _interval(host, seconds, linux, windows, owned_pids)
        windows_seen.append(result)
        streak = streak + 1 if result['accepted'] else 0
        if streak == 2 or time.monotonic() > deadline:
            break
    brief = [dict(accepted=w['accepted'], issues=w['issues'],
                  linux=w.get('linux_background_cores'), windows=w.get('windows_background_cores'))
             for w in windows_seen]
    return dict(passed=streak == 2, limits=dict(seconds=seconds, linux=linux, windows=windows),
                attempts=len(brief), windows=brief)


def start_sampler(out: Path, cpu: int | None = None, period: float = 0.25) -> subprocess.Popen:
    command = [sys.executable, str(HERE / 'loadmon.py'), 'sample', '--out', str(out), '--period', str(period)]
    if cpu is not None:
        command = ['taskset', '-c', str(cpu)] + command
    return subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def stop_sampler(process: subprocess.Popen) -> None:
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def calibrate(out: Path, idle: float, busy: float, busy_cpu: int, bench_cpu: int, sampler_cpu: int) -> dict:
    path = out.with_suffix('.samples.jsonl')
    sampler = start_sampler(path, sampler_cpu)
    reader = SampleReader(path)
    try:
        if not reader.wait_for('windows', runlib.raw_clock(), 60):
            raise SystemExit('sampler produced no Windows reading within 60 s')
        t0 = runlib.raw_clock()
        time.sleep(idle)
        t1 = runlib.raw_clock()
        reader.wait_for('windows', t1, 5)
        quiet = verdict(reader.poll(), t0, t1, benchmark_cpus=[bench_cpu])
        loop = f'import time\nend = time.monotonic() + {busy}\nwhile time.monotonic() < end: pass'
        b0 = runlib.raw_clock()
        subprocess.run(['taskset', '-c', str(busy_cpu), sys.executable, '-c', loop], check=True)
        b1 = runlib.raw_clock()
        reader.wait_for('windows', b1, 5)
        loaded = verdict(reader.poll(), b0, b1, benchmark_cpus=[bench_cpu])
    finally:
        stop_sampler(sampler)
    report = dict(schema='blaze2d-lab/preconditioner/loadmon-calibration/1', created=runlib.utc_now(),
                  idle_seconds=idle, busy_seconds=busy, busy_cpu=busy_cpu, benchmark_cpu=bench_cpu,
                  sampler_cpu=sampler_cpu, idle=quiet, injected=loaded,
                  sampler_cores=quiet['metrics'].get('sampler_cores'),
                  collector_cores=quiet['metrics'].get('collector_cores'),
                  passed=quiet['verdict'] == 'clean' and loaded['verdict'] == 'contaminated')
    runlib.write_json(out, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('sample')
    command.add_argument('--out', type=Path, required=True)
    command.add_argument('--period', type=float, default=0.25)
    command.add_argument('--windows-period', type=float, default=1.0)
    command.add_argument('--duration', type=float)
    command.add_argument('--no-windows', action='store_true')
    command = commands.add_parser('verdict')
    command.add_argument('--samples', type=Path, required=True)
    command.add_argument('--t0', type=float, help='raw clock start (default: first Windows reading)')
    command.add_argument('--t1', type=float, help='raw clock end (default: last Windows reading)')
    command.add_argument('--child-cpu', type=float)
    command.add_argument('--cpus', help='benchmark CPUs, e.g. 2')
    command = commands.add_parser('preflight')
    command.add_argument('--wait', type=float, default=120)
    command.add_argument('--seconds', type=float, default=2.0)
    command.add_argument('--linux', type=float, default=0.25)
    command.add_argument('--windows', type=float, default=1.5)
    command = commands.add_parser('calibrate')
    command.add_argument('--idle', type=float, default=60)
    command.add_argument('--busy', type=float, default=5)
    command.add_argument('--busy-cpu', type=int, default=8)
    command.add_argument('--bench-cpu', type=int, default=2)
    command.add_argument('--sampler-cpu', type=int, default=30)
    command.add_argument('--out', type=Path, default=runlib.RESULTS / 'loadmon' / 'calibration.json')
    args = parser.parse_args()
    if args.command == 'sample':
        sample(args.out, args.period, args.windows_period, args.duration, not args.no_windows)
    elif args.command == 'verdict':
        samples = SampleReader(args.samples).poll()
        kind = 'windows' if any(s['kind'] == 'windows' for s in samples) else 'linux'
        times = [s['t'] for s in samples if s['kind'] == kind]
        t0 = args.t0 if args.t0 is not None else times[0]
        t1 = args.t1 if args.t1 is not None else times[-1]
        cpus = runlib.cpu_list(args.cpus, 0) if args.cpus else None
        print(json.dumps(verdict(samples, t0, t1, args.child_cpu, cpus), indent=1))
    elif args.command == 'preflight':
        with QuietHost() as host:
            result = preflight(host, args.wait, args.seconds, args.linux, args.windows)
        print(json.dumps(result, indent=1))
        raise SystemExit(0 if result['passed'] else 3)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        report = calibrate(args.out, args.idle, args.busy, args.busy_cpu, args.bench_cpu, args.sampler_cpu)
        print(json.dumps({k: report[k] for k in ('passed', 'sampler_cores', 'collector_cores')}
                         | {'idle': report['idle']['verdict'], 'injected': report['injected']['verdict'],
                            'injected_reasons': report['injected']['reasons']}, indent=1))


if __name__ == '__main__':
    main()
