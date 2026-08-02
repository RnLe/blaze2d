#import "../figures.typ": plot
#import "/typst/lib/macros.typ": callout-box
#import "/typst/lib/theme.typ": col-steel-blue

= Multi-Core Performance <sec_multi_core>

The architectural age of legacy solvers is most evident in parallel execution
@hennessy2011 @williams2009. MPB attempts to parallelize individual operations
*_within_* an iteration. On modern hardware, where small-to-medium lattice
problems fit entirely within the CPU cache, this fine-grained threading
introduces synchronization overhead that outweighs the computational gains,
causing performance to regress as threads are added.#footnote[
  Figure 2.2 of @hennessy2011 illustrates the dramatic divergence in performance
  trends between processor speed and memory bandwidth, showing how memory access
  has become the dominant bottleneck in modern computing systems.
]

Blaze avoids this overhead by parallelizing entire jobs rather than individual
operations, a strategy optimized for large-scale parameter sweeps.

@fig_multi_core reveals a clear three-tier performance hierarchy. The legacy
solver struggles with thread overhead. Blaze in Full Precision scales efficiently
but remains sensitive to the higher algebraic load of TE modes (approx.
240–350 ms). In contrast, the Mixed Precision mode hits a "hard floor" at
roughly 160 ms. By halving the memory requirement for state vectors, Blaze
masks much of the computational complexity of the TE mode, consistent with a
solver that has entered a memory-bandwidth-limited regime on a given machine
@williams2009.

#figure(
  plot("multi-core"),
  caption: [
    Blaze achieves 11.9× average speedup over MPB on 16-thread workloads.
    Mixed precision (`f32/f64`) shown alongside full precision (`f64`).
  ],
) <fig_multi_core>

#callout-box(color: col-steel-blue)[
  All subsequent benchmarks for Blaze are performed in Mixed Precision mode,
  unless otherwise specified.
]
