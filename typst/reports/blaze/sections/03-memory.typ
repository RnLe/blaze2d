#import "../figures.typ": plot

= Memory Efficiency <sec_memory>

The recorded process-memory measurements are approximately `190 MB` for MPB over much of this sweep. Process memory includes runtime and library allocations, so this observation does not establish that MPB reserves a fixed solver workspace.

Blaze uses less measured peak memory in these runs. A smaller working set can reduce data movement, but cache residency requires direct profiling and cannot be inferred from the memory ratio alone.

For a square grid with side length $N$, field storage grows as $N^2$ per band. Eigensolver workspaces also depend on band count, precision, and retained outputs. Process-level measurements may conceal that growth over a limited range.

#figure(
  plot("memory-resolution"),
  caption: [
    Peak memory against grid resolution, for both polarizations. MPB holds a
    fixed footprint of roughly 190 MB at every resolution, while Blaze
    allocates what the problem actually requires.
  ],
) <fig_memory_resolution>

Both recorded process-memory curves remain nearly flat over the tested band-count range. This does not imply that storing more eigenvectors has no memory cost.

#figure(
  plot("memory-bands"),
  caption: [
    Peak memory against the number of requested bands. Neither solver's
    footprint grows with the size of the LOBPCG search block.
  ],
) <fig_memory_bands>

== Memory Scaling Laws <sec_memory_scaling>

The measured MPB-to-Blaze memory ratio is about `20×` at the smallest grids and decreases as the grid grows. These finite-range measurements do not determine an asymptotic memory ratio.

#figure(
  plot("memory-ratio"),
  caption: [
    Ratio of MPB's peak memory to Blaze's, swept over resolution (left) and band
    count (right). The advantage is largest where MPB's fixed allocation
    dominates, and decays as the grid itself becomes the cost.
  ],
) <fig_memory_ratio>

The fitted exponents over this finite range are 0.06 for MPB and 1.09 for Blaze. These are empirical process-memory trends, not asymptotic storage laws or proof of a particular allocation strategy.

#figure(
  plot("memory-scaling"),
  caption: [
    Peak memory against resolution on log-log axes, with power-law fits. MPB is
    flat at $N^(0.06)$; Blaze follows $N^(1.09)$, well below the $O(N^2)$
    reference.
  ],
) <fig_memory_scaling>
