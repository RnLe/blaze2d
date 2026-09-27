"""Read-only WSL/Windows activity admission; does not reserve hardware resources."""
import json
import math
import os
from pathlib import Path
import resource
import subprocess
import time

MEASUREMENT_PROTOCOL = 'raw-clock-v4'


def elapsed_clocks(start, end):
    """Keep CPU-clock denominators independent of the adjusted guest clock."""
    try:
        raw = end['monotonic_raw'] - start['monotonic_raw']
        adjusted = end['monotonic'] - start['monotonic']
        frequency = start['windows']['qpc_frequency']
        if frequency <= 0 or end['windows']['qpc_frequency'] != frequency:
            raise ValueError('Windows QPC frequency changed or is invalid')
        windows = (end['windows']['qpc_ticks'] - start['windows']['qpc_ticks']) / frequency
    except KeyError as error:
        raise ValueError('Missing independent raw/QPC clock evidence') from error
    if not all(math.isfinite(v) and v > 0 for v in (raw, adjusted, windows)):
        raise ValueError('Invalid measurement interval')
    if abs(raw-windows) > .010 + .005*raw:
        raise ValueError('Windows QPC and Linux raw elapsed clocks disagree')
    return dict(raw_seconds=raw, windows_seconds=windows, adjusted_seconds=adjusted,
                adjusted_to_raw_scale=raw/adjusted)


def assess(start, end, limit=0.5, windows_limit=1.5):
    try:
        clocks = elapsed_clocks(start, end)
    except ValueError as error:
        return {'accepted': False, 'issues': [str(error)],
                'measurement_protocol': MEASUREMENT_PROTOCOL}
    elapsed = clocks['raw_seconds']
    own = end['owned_cpu_seconds'] - start['owned_cpu_seconds']
    linux = (end['linux_busy_seconds'] - start['linux_busy_seconds'] - own) / elapsed
    win_own = end['windows']['collector_cpu_seconds'] - start['windows']['collector_cpu_seconds']
    windows = (end['windows']['busy_seconds'] - start['windows']['busy_seconds'] - own - win_own) / clocks['windows_seconds']
    issues = []
    for host, value, bound in [('linux', linux, limit), ('windows', windows, windows_limit)]:
        if value > bound:
            issues.append(f'{host} background CPU {value:.3f} cores exceeds {bound}')
    for key in ['swap_in', 'swap_out', 'memory_stall_us']:
        if end[key] != start[key]:
            issues.append(key + ' changed')
    if min(start['memory_available_bytes'], end['memory_available_bytes']) < 2 * 1024**3:
        issues.append('less than 2 GiB available memory')
    return dict(accepted=not issues, issues=issues, elapsed_seconds=elapsed,
                clocks=clocks, measurement_protocol=MEASUREMENT_PROTOCOL,
                linux_background_cores=linux, windows_background_cores=windows,
                owned_cpu_seconds=own, linux_limit_background_cores=limit,
                windows_limit_background_cores=windows_limit)


class QuietHost:
    def __init__(self):
        script = Path(__file__).with_name('host_clock.ps1').resolve()
        self.process = subprocess.Popen([
            '/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe',
            '-NoProfile', '-NonInteractive', '-Command', script.read_text()],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self.ticks = os.sysconf('SC_CLK_TCK')
        self.snapshot()  # Complete Windows initialization outside all measurement windows.

    def snapshot(self):
        self.process.stdin.write('\n'); self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError('Windows clock unavailable: ' + self.process.stderr.read()[-2000:])
        windows = json.loads(line)
        cpu = list(map(int, Path('/proc/stat').read_text().splitlines()[0].split()[1:9]))
        vm = dict(line.split() for line in Path('/proc/vmstat').read_text().splitlines())
        mem = dict((k, int(v.split()[0]) * 1024)
                   for k, v in (line.split(':') for line in Path('/proc/meminfo').read_text().splitlines()))
        pressure = Path('/proc/pressure/memory').read_text().splitlines()
        stall = sum(int(line.split('total=')[1]) for line in pressure if line.startswith('full '))
        own = resource.getrusage(resource.RUSAGE_SELF)
        children = resource.getrusage(resource.RUSAGE_CHILDREN)
        # Bracket the raw read to quantify the local clock-pair alignment.
        before = time.monotonic_ns()
        raw = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        after = time.monotonic_ns()
        return dict(monotonic=(before+after)/2e9, monotonic_raw=raw/1e9,
                    local_clock_alignment_seconds=(after-before)/2e9,
                    utc=time.time(), windows=windows,
                    linux_busy_seconds=(sum(cpu) - cpu[3] - cpu[4]) / self.ticks,
                    owned_cpu_seconds=own.ru_utime + own.ru_stime + children.ru_utime + children.ru_stime,
                    swap_in=int(vm['pswpin']), swap_out=int(vm['pswpout']), memory_stall_us=stall,
                    memory_available_bytes=mem['MemAvailable'], load_average=os.getloadavg())

    def interval(self, duration=2, limit=0.25, windows_limit=1.5):
        start = self.snapshot()
        deadline = start['monotonic_raw'] + duration
        while (remaining := deadline-time.clock_gettime(time.CLOCK_MONOTONIC_RAW)) > 0:
            time.sleep(max(.001, remaining))
        end = self.snapshot()
        return dict(start=start, end=end, **assess(start, end, limit, windows_limit))

    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=10)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
