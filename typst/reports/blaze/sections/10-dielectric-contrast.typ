#import "../figures.typ": plot

= Dielectric Contrast <sec_dielectric_contrast>

The benchmark in @fig_epsilon_bar varies the rod permittivity from
$epsilon = 2$ to $epsilon = 13$, covering the range where photonic band gaps
open. Blaze's solve time stays essentially flat across the whole range, which is
a useful stability property: the cost does not depend on the contrast of the
crystal. MPB's behavior is more uneven. Its timings vary substantially from one
contrast to the next, with occasional sudden drops and jumps. MPB is therefore
more sensitive to the dielectric contrast than Blaze.

#figure(
  plot("epsilon-bar"),
  caption: [
    Solve time against rod permittivity for TM (left) and TE (right). Blaze is
    flat across the sweep; MPB is not.
  ],
) <fig_epsilon_bar>
