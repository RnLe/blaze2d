#import "../figures.typ": plot

= Convergence and Iteration Count <sec_iterations>

Beyond the cost of the _entire_ single iteration, the number of iterations
needed to converge each $bold(k)$-point matters.

Blaze converges in fewer iterations than MPB on average, about 4.5 per
$bold(k)$-point for TM and 5.6 for TE, against 7.5 and 11.9 for MPB
(@fig_iteration_bar). Both solvers *warm-start* each $bold(k)$-point from the
converged solution of the previous one (Blaze adopts this technique from MPB),
so the warm start cannot explain the difference. The most likely explanation is
a combination of soft-locking, where converged bands are kept in the working
space but no longer refined, and the different convergence criteria the two
solvers use: Blaze stops once the eigenvalues have stabilized, while MPB
monitors the residual. There may also be a general efficiency difference in the
LOBPCG variant.

MPB's iteration count spikes at certain $bold(k)$-points, most often around
high-symmetry points, where it reaches 39 iterations for TM and 85 for TE.
Blaze has occasional spikes as well, but they are smaller (up to 12 and 24)
and tend to fall at _different_ $bold(k)$-points (@fig_iteration_distribution).
That the spikes occur in different places is consistent with the two solvers
using different stopping criteria, which become sensitive under different
conditions rather than at the same points. The polarization gap is visible
throughout: TM, whose operator is diagonal and easy to precondition, converges
in fewer iterations than TE for both solvers.

#figure(
  plot("iteration-bar"),
  caption: [
    Mean iteration count per $bold(k)$-point for TM (left) and TE (right).
  ],
) <fig_iteration_bar>

#figure(
  plot("iteration-time"),
  caption: [
    Time per iteration for the same runs. Blaze's per-iteration cost is lower as
    well, so the iteration-count advantage compounds.
  ],
) <fig_iteration_time>

#figure(
  plot("iteration-distribution"),
  caption: [
    Distribution of iteration counts over all $bold(k)$-points. MPB's tail
    reaches 39 iterations for TM and 85 for TE; Blaze's stays short.
  ],
) <fig_iteration_distribution>
