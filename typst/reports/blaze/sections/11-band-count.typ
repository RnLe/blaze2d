#import "../figures.typ": plot

= Scaling with Band Count <sec_band_count>

The sweep in @fig_bands_bar varies the number of bands requested, from 4 to 20.
The cost grows steadily with band count for both solvers, which is expected:
more bands mean a larger LOBPCG search block and more vectors to
orthogonalize each iteration. Blaze keeps a roughly 2× lead across the whole
range, solving the 20-band TM problem in 1.9 s against MPB's 3.9 s. As
@sec_memory showed, this added work does not increase the memory footprint, so
requesting more bands costs time but not space.

#figure(
  plot("bands-bar"),
  caption: [
    Solve time against the number of requested bands for TM (left) and TE
    (right).
  ],
) <fig_bands_bar>
