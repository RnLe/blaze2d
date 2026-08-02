#import "../figures.typ": plot

= Parallel Scaling <sec_parallel_scaling>

The two solvers parallelize in different ways. MPB splits the work inside a
single solve across threads, while Blaze runs many independent solves at the
same time, one per thread, which suits the parameter sweeps it is built for.
@fig_throughput_scaling sweeps the thread count from 1 to 16 and measures
throughput in solves per second, for a small problem ($N = 16$) and a large one
($N = 128$).

For the small problem, Blaze scales almost linearly, from 28 to 220 solves
per second across 16 threads. MPB's threaded mode does not benefit: it stays
near 14 solves per second and even drops slightly, because the cost of
coordinating threads within a single solve outweighs the gain. Running MPB as
separate processes, one solve per core, does scale, but reaches only about half
of Blaze's throughput.

For the large problem the absolute numbers are smaller for every solver, but the
pattern is the same. Blaze scales by about 3× across the sweep
(@fig_speedup_scaling), while MPB's threaded mode stays flat. The practical
conclusion is that *parallelizing across independent jobs works better than
parallelizing within a single solve*.

#figure(
  plot("throughput-scaling"),
  caption: [
    Throughput in solves per second against thread count, for a small problem
    ($N = 16$, left) and a large one ($N = 128$, right).
  ],
) <fig_throughput_scaling>

#figure(
  plot("speedup-scaling"),
  caption: [
    The same sweep expressed as speedup over the single-threaded result. Blaze
    tracks the ideal line closely at $N = 16$; MPB's threaded mode is flat in
    both panels.
  ],
) <fig_speedup_scaling>
