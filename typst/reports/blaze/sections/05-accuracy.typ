#import "../figures.typ": plot-row

= Accuracy Validation <sec_accuracy>

The reduction in precision and memory footprint raises a direct question: do we
gain speed at the cost of accuracy? To answer this question, we compared the
eigenfrequencies from Blaze against a high-precision MPB reference along the
full $Gamma -> X -> M -> Gamma$ path. In @fig_band_comparison, the MPB bands are
drawn as lines and the Blaze eigenvalues as markers.

A short note on methodology. MPB tracks each band _adiabatically_ across avoided
crossings, while Blaze reports the lowest eigenvalues at each $bold(k)$-point
without assigning band identities. The two conventions can disagree on the
highest band of the computed set, where a crossing may exchange a band with the
next one just outside the set. To keep the comparison clean we compute the
lowest 20 bands with both solvers but show only the lowest 10, and for Blaze we
plot the lowest eigenvalues directly with no band matching. Every displayed band
is then well inside the computed window, so both solvers report the same set of
values and the band-tracking difference does not appear here.

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
