#import "../figures.typ": plot
#import "/typst/lib/macros.typ": callout-box
#import "/typst/lib/theme.typ": col-steel-blue

= Multi-Core Performance <sec_multi_core>

The benchmark compares MPB threading within individual solves with Blaze scheduling independent configurations. The cost of synchronization and data movement can limit scaling for small problems.

Blaze schedules independent configurations across workers. In these measurements,
full precision TE solves take approximately 240–350 ms and mixed precision runs
cluster near 160 ms. The timing data alone do not identify a bandwidth limit or
separate allocation, synchronization, and arithmetic costs @williams2009.

#figure(
  plot("multi-core"),
  caption: [
    The recorded 16-thread workloads have an average MPB-to-Blaze timing ratio of 11.9.
    Mixed precision (`f32/f64`) shown alongside full precision (`f64`).
  ],
) <fig_multi_core>

#callout-box(color: col-steel-blue)[
  All subsequent benchmarks for Blaze are performed in Mixed Precision mode,
  unless otherwise specified.
]
