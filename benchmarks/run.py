"""Measure a configured study without modifying its scientific settings."""
import argparse
from datetime import datetime, timezone
import json
import platform
from pathlib import Path
import statistics
import time
import blaze


def peak_memory_bytes():
    if platform.system() != 'Windows':
        import resource
        value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(value if platform.system() == 'Darwin' else value * 1024)
    import ctypes
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage','PagefileUsage','PeakPagefileUsage')]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    function = ctypes.WinDLL('psapi', use_last_error=True).GetProcessMemoryInfo
    function.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
    if not function(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    return counters.PeakWorkingSetSize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file', type=Path)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--warmups', type=int, default=2)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or args.warmups < 0: parser.error('Use positive repeats and nonnegative warmups')
    config = blaze.Config.from_file(args.file)
    elapsed = []
    for index in range(args.warmups + args.repeats):
        start = time.perf_counter()
        study = blaze.run(config, threads=args.threads)
        seconds = time.perf_counter() - start
        if study['statistics']['status'] != 'completed':
            raise RuntimeError(study['errors'])
        if index >= args.warmups: elapsed.append(seconds)
        last = study['statistics']
        del study
    report = {'schema':'blaze2d/benchmark/1', 'recorded_at':datetime.now(timezone.utc).isoformat(),
              'build':blaze.build_info(), 'config':config.to_dict(), 'run':last,
              'host':{'platform':platform.platform(), 'machine':platform.machine(), 'processor':platform.processor(),
                      'python':platform.python_version()},
              'warmups':args.warmups, 'seconds':elapsed, 'median_seconds':statistics.median(elapsed),
              'peak_process_bytes':peak_memory_bytes()}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(f"Median {report['median_seconds']:.6f} s; peak process {report['peak_process_bytes']/2**20:.1f} MiB")

if __name__ == '__main__': main()
