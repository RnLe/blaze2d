#import "../figures.typ": plot-row

= Accuracy Validation <sec_accuracy>

The reduction in precision and memory footprint raises a direct question: do we
gain speed at the cost of accuracy? To answer this question, we compared the
eigenfrequencies from Blaze against a high-precision MPB reference along the
full $Gamma -> X -> M -> Gamma$ path. In @fig_band_comparison, the MPB bands are
drawn as lines and the Blaze eigenvalues as markers.

The comparison computes 20 bands and displays the lowest 10. Frequencies are compared as sorted eigenvalue sets at each wavevector, avoiding sensitivity to band labels near crossings and the upper edge of the solved window. This historical plotting convention differs from the current API, which preserves the tracked path returned by the solver.

#figure(
  plot-row("bands-tm", "bands-te"),
  caption: [
    Band structures along $Gamma -> X -> M -> Gamma$ for TM (left) and TE
    (right). MPB is drawn as lines, Blaze in full and mixed precision as
    markers. The three curves are indistinguishable at this scale.
  ],
) <fig_band_comparison>

#figure(
  plot-row("deviation-tm", "deviation-te"),
  caption: [
    Distribution of the relative deviation across all bands and
    $bold(k)$-points, for TM (left) and TE (right). The first two boxes measure
    each Blaze run against MPB; the third measures the two Blaze runs against
    each other.
  ],
) <fig_deviation>
