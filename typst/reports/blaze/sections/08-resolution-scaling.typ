#import "../figures.typ": plot

= Scaling with Resolution <sec_resolution_scaling>

Resolution sets the overall cost: the number of plane waves grows as $N^2$ and
the FFTs as $N^2 log N$. @fig_resolution_bar sweeps the grid from $N = 16$ to
$N = 192$ and compares wall-clock time against MPB.

Blaze is faster at every resolution, and its solve time grows smoothly and
predictably with $N$. MPB behaves differently. Its timings are uneven across
resolutions, performing noticeably better at powers of two (16, 32, 64, 128) and
jumping in cost at the values in between. At $N = 96$ MPB is already almost as
slow as at $N = 128$, and the step up to $N = 192$ is large. The exact cause is
not certain, but it most likely comes down to how the FFT plans and operator
tiling handle sizes that are not powers of two. @fig_resolution_speedup shows
the combined effect: Blaze's lead grows toward higher $N$, reaching roughly 5×
at $N = 192$ (8.0 s against 39 s for TM, 12 s against 70 s for TE).

The log-log scaling plot in @fig_resolution_scaling makes the trends comparable.
Blaze follows a clean power law, while MPB's curve is jumpy and trends toward a
steeper slope at high resolution.

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
