#import "../figures.typ": plot

= Scaling with Resolution <sec_resolution_scaling>

Resolution sets the overall cost: the number of plane waves grows as $N^2$ and
the FFTs as $N^2 log N$. @fig_resolution_bar sweeps the grid from $N = 16$ to
$N = 192$ and compares wall-clock time against MPB.

Blaze is faster at each sampled resolution in this dataset. At $N=192$, the recorded times are approximately `8.0 s` versus `39 s` for TM and `12 s` versus `70 s` for TE. MPB timings vary non-monotonically with some grid sizes. FFT factorization and planning are possible contributors, requiring separate profiling.

The log-log plot compares empirical trends over the sampled resolutions. Its fitted slopes should not be extrapolated beyond this range without further measurements.

#figure(
  plot("resolution-bar"),
  caption: [
    Wall-clock solve time against grid resolution for TM (left) and TE (right).
    MPB's cost is uneven across resolutions; Blaze's grows smoothly.
  ],
) <fig_resolution_bar>

#figure(
  plot("resolution-speedup"),
  caption: [
    Blaze's speedup over MPB across the same sweep. The lead widens with
    resolution rather than saturating.
  ],
) <fig_resolution_speedup>

#figure(
  plot("resolution-scaling"),
  caption: [
    The same data on log-log axes with power-law fits, which makes the two
    solvers' scaling exponents directly comparable.
  ],
) <fig_resolution_scaling>
