#import "../figures.typ": plot
#import "/typst/lib/macros.typ": callout-box
#import "/typst/lib/theme.typ": col-steel-blue

= Single-Core Performance <sec_single_core>

#callout-box(color: col-steel-blue)[
  Performance benchmarks utilize the canonical square and hexagonal lattice
  configurations from Joannopoulos' seminal 1997 Nature paper @joannopoulos1997.
  Unless otherwise specified, all data reflects the square lattice configuration
  at a resolution of 64, computing the lowest 8 bands with 20 $bold(k)$-points
  per segment between two high-symmetry points.
]

The computational cost of PWE solvers is dominated by Fast Fourier Transforms
(FFTs) @mpb. In this plane-wave formulation, Transverse Electric (TE) modes are
more expensive to solve than Transverse Magnetic (TM) modes, because each
operator application requires six FFTs rather than two for TM.

This complexity penalty is clearly visible in the legacy solver (@fig_single_core).
Blaze, however, mitigates this through algorithmic optimizations. Even in Full
Precision (`f64`), Blaze outperforms MPB. The decisive leap comes from the Mixed
Precision (`f32/f64`) approach, which reduces memory traffic enough to
effectively double the throughput, resulting in a total speedup of approximately
3×.

#figure(
  plot("single-core"),
  caption: [
    Blaze achieves 2.9× average speedup over MPB on single-core workloads.
    Mixed precision (`f32/f64`) shown alongside full precision (`f64`).
  ],
) <fig_single_core>
