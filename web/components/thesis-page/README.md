# Thesis article assets

The article follows `master-thesis/thesisV3.pdf`, the second edition revised in
September 2026. The source repository is a sibling of `blaze2d` on the development
machine. The PDF copied into `public/reports/masters-thesis-second-edition.pdf`
has 92 pages. It is intentionally distinct from Blaze's technical report.

The SVGs in `public/figures/thesis/second-edition/` are unmodified source exports:

| Site asset | Source within master-thesis | Thesis figure | PDF page |
| --- | --- | --- | --- |
| `hero_scaling.svg` | `assets/figures/second_edition/hero_scaling.svg` | 44 | 61 |
| `ladder_big.svg` | `assets/figures/second_edition/ladder_big.svg` | 47 | 63 |
| `order_ladder.svg` | `assets/figures/second_edition/order_ladder.svg` | 48 | 64 |
| `thesis_crystal_ladder.svg` | `assets/figures/second_edition/thesis_crystal_ladder.svg` | 43 | 59 |
| `dirac_tm_full_stack.svg` | `assets/figures/second_edition/dirac_tm_full_stack.svg` | 61 | 82 |
| `dirac_metrics.svg` | `assets/figures/second_edition/dirac_metrics.svg` | 52 | 68 |
| `bandwidth_vs_angle.svg` | `assets/figures/second_edition/bandwidth_vs_angle.svg` | 53 | 69 |
| `te-hamiltonian.svg` | `assets/figures/equation_boxes/custom/eq_te_full_hamiltonian_v3.svg` | Appendix A | |
| `tm-hamiltonian.svg` | `assets/figures/equation_boxes/custom/eq_tm_full_hamiltonian_v3.svg` | Appendix A | |

Results and captions are grounded in `sectionsV3/validations.typ`; equations in
`sectionsV3/photonic_moire_crystals.typ`, `sectionsV3/envelope_approximation.typ`
and the two Hamiltonian poster sources. The solver motivation is supported by
`sectionsV3/blaze_2d.typ` and the author's description of the missing MPB operator
outputs. MPB's public interface does expose eigenvectors and group velocities.

Keep frequency error `f = omega a / (2 pi c)` distinct from eigenvalue error
`lambda = (2 pi f)^2`. The `eta^3.8` convergence belongs to the family with
second-layer amplitude proportional to `eta^2`. It is not an error bound for
fixed materials. The exact-frame validation is a different model from the
finite-order registry-grid Hamiltonian used for the honeycomb angle scan.

Regenerate the original static hero from the site's brand tokens with:

```sh
node scripts/render-thesis-hero.mjs
```

It shows a triangular Bravais bilayer at 6 degrees, 64 lattice constants across.
The homepage's honeycomb animation remains a separate asset. No image generator
or rasterization is needed for either the lattice geometry or the source plots.

`lib/moire.ts` computes geometric beat vectors from the reciprocal-lattice
mismatch. The explorer reduces twists to the nearest equivalent alignment under
60-degree triangular or 90-degree square symmetry. At alignment the beat period
is infinite; it must not be displayed as a finite coincidence cell. A bounded
canvas draws sites only on input or resize; SVG draws the annotations in matching
pixel coordinates, so resizing the viewport never stretches the lattice.

The explorer reserves less than 90% of the viewport height in the article. An
explicit button expands its child to the viewport width and centers it below the
sticky header, while the navigation columns fade out and become inert. Scrolling the module out of view collapses it;
returning does not expand it again. The reserved slot prevents scroll jumps.
Navigation state is restored on collapse and unmount; reduced motion skips transitions.
The hero matches the text width without cropping and uses a CSS mask to blend
into the page.

The browser tests in `tests/thesis.spec.ts` cover reciprocal duality, endpoints,
interaction, viewport fit, navigation restoration, modal focus and zoom, source loading, accessibility and narrow
viewports. Run against a local preview with `BLAZE_TEST_URL` as described in the
website README.
