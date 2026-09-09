#import "../figures.typ": plot

= Convergence and Iteration Count <sec_iterations>

Beyond the cost of the _entire_ single iteration, the number of iterations
needed to converge each $bold(k)$-point matters.

Mean iteration counts are approximately 4.5 for Blaze TM and 5.6 for TE, compared with 7.5 and 11.9 for MPB. The recorded tolerances are `1e-4` for Blaze and `1e-7` for MPB. Both use eigenvalue changes in their stopping criteria; #link("https://mpb.readthedocs.io/en/latest/Python_User_Interface/")[MPB documents this explicitly]. The iteration counts therefore do not constitute an equal-tolerance comparison. Fresh residuals must be checked separately.

Maximum recorded iteration counts are `39` for MPB TM and `85` for MPB TE, compared with `12` and `24` for Blaze. Their locations vary along the path. The measurements do not isolate whether preconditioning, subspace updates, or stopping settings cause the difference.

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
