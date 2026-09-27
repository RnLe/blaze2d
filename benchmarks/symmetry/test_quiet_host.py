import copy
import unittest
from quiet_host import assess


class AdmissionTests(unittest.TestCase):
    def bounds(self):
        a = dict(monotonic=10, monotonic_raw=10, owned_cpu_seconds=0, linux_busy_seconds=0,
                 windows=dict(busy_seconds=0, collector_cpu_seconds=0,
                              qpc_ticks=10000, qpc_frequency=1000),
                 swap_in=0, swap_out=0, memory_stall_us=0,
                 memory_available_bytes=4*1024**3)
        b = copy.deepcopy(a)
        b.update(monotonic=12, monotonic_raw=12, owned_cpu_seconds=6, linux_busy_seconds=6.1)
        b['windows'].update(busy_seconds=6.5, collector_cpu_seconds=0.1, qpc_ticks=12000)
        return a, b

    def test_owned_parallel_work_is_not_background(self):
        self.assertTrue(assess(*self.bounds())['accepted'])

    def test_separate_windows_load_is_visible(self):
        a, b = self.bounds(); b['windows']['busy_seconds'] += 4
        self.assertFalse(assess(a, b)['accepted'])

    def test_separate_linux_load_is_visible(self):
        a, b = self.bounds(); b['linux_busy_seconds'] += 2
        self.assertFalse(assess(a, b)['accepted'])

    def test_paging_and_stalls_reject_quiet_cpu(self):
        for key in ['swap_in', 'swap_out', 'memory_stall_us']:
            a, b = self.bounds(); b[key] += 1
            self.assertFalse(assess(a, b)['accepted'])

    def test_available_memory_gate(self):
        a, b = self.bounds(); b['memory_available_bytes'] = 1024**3
        self.assertFalse(assess(a, b)['accepted'])

    def test_invalid_window(self):
        a, b = self.bounds(); b['monotonic'] = a['monotonic']
        self.assertFalse(assess(a, b)['accepted'])

    def test_fast_adjusted_clock_cannot_hide_host_load(self):
        a, b = self.bounds()
        b['monotonic'] = 12.2
        b['windows']['busy_seconds'] = 9.3
        result = assess(a, b)
        self.assertFalse(result['accepted'])
        self.assertAlmostEqual(result['windows_background_cores'], 1.6)
        self.assertAlmostEqual(result['clocks']['adjusted_to_raw_scale'], 1/1.1)

    def test_missing_or_disagreeing_independent_clocks_fail_closed(self):
        a, b = self.bounds(); del b['monotonic_raw']
        self.assertFalse(assess(a,b)['accepted'])
        a, b = self.bounds(); b['windows']['qpc_ticks'] = 13000
        self.assertFalse(assess(a,b)['accepted'])


if __name__ == '__main__':
    unittest.main()
