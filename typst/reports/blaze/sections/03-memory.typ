#import "../figures.typ": plot

= Memory Efficiency <sec_memory>

High-performance computing is increasingly defined by data movement
@williams2009. A major limitation of MPB is its static memory management;
benchmarks reveal that the legacy solver reserves a large, fixed memory block
(approx. 190 MB) *_regardless_* of the problem size.

Blaze adopts a dynamic allocation strategy. As @fig_memory_resolution shows,
this results in a dramatic reduction in peak memory usage for standard
resolutions. This reduction is critical: by keeping the working set small, Blaze
allows the CPU to operate almost entirely within its high-speed L3 cache,
avoiding the latency penalty of fetching data from main RAM.

Fundamentally, the storage requirements for FFTs and operator workspaces scale
directly with the grid resolution ($N$). Therefore, analyzing memory growth
against resolution provides the most critical insight into the architectural
efficiency.

#figure(
  plot("memory-resolution"),
  caption: [
    Peak memory against grid resolution, for both polarizations. MPB holds a
    fixed footprint of roughly 190 MB at every resolution, while Blaze
    allocates what the problem actually requires.
  ],
) <fig_memory_resolution>

This efficiency extends to the dimensionality of the search space. In the LOBPCG
algorithm, the search space size is determined by the number of bands ($3n$)
@knyazev2001. While one might expect memory usage to scale with this complexity,
both solvers maintain a constant footprint even as the number of bands increases
(@fig_memory_bands).

#figure(
  plot("memory-bands"),
  caption: [
    Peak memory against the number of requested bands. Neither solver's
    footprint grows with the size of the LOBPCG search block.
  ],
) <fig_memory_bands>

== Memory Scaling Laws <sec_memory_scaling>

To understand the limits of this efficiency, we analyzed how the relative
advantage evolves (@fig_memory_ratio). At low resolutions, MPB is dominated by
its static overhead, giving Blaze a 20× advantage. As the resolution
increases, the physical storage requirements for the grid naturally grow, and
the ratio asymptotically approaches 1×. As mentioned, for the number of bands
sweep, both solvers maintain constant memory usage, resulting in a flat ratio.

#figure(
  plot("memory-ratio"),
  caption: [
    Ratio of MPB's peak memory to Blaze's, swept over resolution (left) and band
    count (right). The advantage is largest where MPB's fixed allocation
    dominates, and decays as the grid itself becomes the cost.
  ],
) <fig_memory_ratio>

For varying resolutions, MPB's memory usage is effectively constant
($N^(0.06)$), confirming the pre-allocation hypothesis. In contrast, Blaze
follows a near-linear trend ($N^(1.09)$), scaling predictably with the problem
size. Notably, this footprint is identical for both TM and TE polarizations,
proving that the storage cost in Blaze is determined strictly by grid topology,
independent of the operator's computational complexity.

#figure(
  plot("memory-scaling"),
  caption: [
    Peak memory against resolution on log-log axes, with power-law fits. MPB is
    flat at $N^(0.06)$; Blaze follows $N^(1.09)$, well below the $O(N^2)$
    reference.
  ],
) <fig_memory_scaling>
