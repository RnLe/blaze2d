#import "../figures.typ": plot

= Resolution and Geometry <sec_resolution_geometry>

Several of the benchmarks below vary the grid resolution, measured in pixels per
unit cell. @fig_epsilon_grids shows what a given resolution looks like for a
circular rod, which helps build intuition for how coarse these grids actually
are. It also shows the subpixel smoothing applied at the rod boundary: instead
of sampling the permittivity on a hard pixel grid, which would produce jagged
staircase edges, the dielectric is smoothed at the corners. Blaze uses the same
subpixel smoothing method as MPB @farjadpour2006. In practice, resolutions of 32
to 64 pixels per unit cell are enough for research-grade band structures, and
higher values are rarely necessary unless the unit cell contains many atoms.

#figure(
  plot("epsilon-grids"),
  caption: [
    Smoothed permittivity map of a circular rod at four grid resolutions. The
    boundary cells take intermediate values from subpixel smoothing rather than
    snapping to $epsilon = 1$ or $epsilon = 13$, which is what keeps a coarse
    grid usable. On the website this figure is an interactive slider.
  ],
) <fig_epsilon_grids>
