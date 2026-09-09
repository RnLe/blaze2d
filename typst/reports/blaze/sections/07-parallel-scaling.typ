#import "../figures.typ": plot

= Parallel Scaling <sec_parallel_scaling>

The two solvers parallelize in different ways. MPB splits the work inside a
single solve across threads, while Blaze runs many independent solves at the
same time, one per thread, which suits the parameter sweeps it is built for.
@fig_throughput_scaling sweeps the thread count from 1 to 16 and measures
throughput in solves per second, for a small problem ($N = 16$) and a large one
($N = 128$).

For the recorded $N=16$ problem, Blaze throughput increases from about `28` to `220` solves per second across 16 threads. MPB threaded throughput stays near `14` solves per second. Separate MPB processes scale better than its threaded mode in this experiment.

For $N=128$, Blaze throughput increases by about `3×` across the thread-count sweep. This supports scheduling independent configurations for these workloads; it does not establish a general limit on MPB parallelism.

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
