#import "../figures.typ": plot

= Dielectric Contrast <sec_dielectric_contrast>

The dielectric-contrast sweep covers rod permittivities from 2 to 13. Blaze times vary little over these sampled configurations; MPB times vary more. This does not establish contrast-independent cost outside the measured range.

#figure(
  plot("epsilon-bar"),
  caption: [
    Solve time against rod permittivity for TM (left) and TE (right). Blaze is
    flat across the sweep; MPB is not.
  ],
) <fig_epsilon_bar>
