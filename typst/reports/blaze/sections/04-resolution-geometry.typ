#import "../figures.typ": plot

= Resolution and Geometry <sec_resolution_geometry>

Grid resolution gives the number of samples along each unit-cell direction.
@fig_epsilon_grids shows the sampled permittivity of a circular rod. Boundary
cells use subpixel averaging, related to the interface treatment described by
@farjadpour2006. This does not establish identical rasterization behavior to MPB.
The required resolution depends on geometry, dielectric contrast, frequency,
and the target observable. Check convergence under grid refinement.

#figure(
  plot("epsilon-grids"),
  caption: [
    Smoothed permittivity map of a circular rod at four grid resolutions. The
    boundary cells take intermediate values from subpixel smoothing rather than
    snapping to $epsilon = 1$ or $epsilon = 13$, which is what keeps a coarse
    grid usable. On the website this figure is an interactive slider.
  ],
) <fig_epsilon_grids>
