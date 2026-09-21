// Blaze2D technical report.
//
// Print form of web/content/blaze.mdx. The prose is the post's, with figure
// references converted to cross-references, inline citations resolved against
// ../../references.bib, and the page's note boxes rendered as callouts.
//
// Charts are exported from the site's own React components; regenerate them
// with `make figures REPORT=blaze` before building. Build with `make blaze`.

#import "/typst/lib/report.typ": report

#show: report.with(
  title: "Blaze2D: A High-Performance Solver for Photonic Crystals",
  subtitle: [Achieving order-of-magnitude speedups through a mixed-precision LOBPCG algorithm and cache-aware architecture.],
  kicker: "Technical Report",
  affiliation: [
    Blaze2D \
    A 2D Maxwell Solver in Rust
  ],
  author: "Rene-Marcel Lehner",
  date: "January 8, 2026",
  project: "Blaze2D Technical Report",
  keywords: [photonic crystal, plane-wave expansion, LOBPCG, mixed precision,
             MPB, benchmarks, Rust],
  logo: "/typst/assets/images/blaze.svg",
  marks: ("/typst/assets/images/github.svg", "/typst/assets/images/pypi.svg"),
  links: (
    (label: "Repository", text: "github.com/RnLe/blaze2d",
     url: "https://github.com/RnLe/blaze2d"),
    (label: "Package", text: "pypi.org/project/blaze2d",
     url: "https://pypi.org/project/blaze2d/"),
    (label: "Source", text: "rnle.github.io/blaze2d/blaze",
     url: "https://rnle.github.io/blaze2d/blaze/", on-cover: false),
  ),
  abstract: [
    Photonic band structure calculations commonly rely on the Plane Wave
    Expansion (PWE) method, and MIT Photonic Bands
    (#link("https://mpb.readthedocs.io/en/latest/")[MPB]) has long served as a
    widely used reference implementation for it @mpb @sakoda2005. MPB is trusted
    for its accuracy and is therefore a natural baseline for performance
    comparisons.

    Blaze uses LOBPCG and mixed-precision arithmetic @knyazev2001 @woo2023.
    This report presents historical measurements. The archived datasets do not
    consistently record hardware identifiers or solver revisions. Series 4 and 5
    use MPB tolerance `1e-7` and Blaze tolerance `1e-4`; those comparisons are
    not at equal tolerance. Reported ratios apply to the recorded experiments.

  ],
)

#include "sections/01-single-core.typ"
#include "sections/02-multi-core.typ"
#include "sections/03-memory.typ"
#include "sections/04-resolution-geometry.typ"
#include "sections/05-accuracy.typ"
#include "sections/06-deviation.typ"
#include "sections/07-parallel-scaling.typ"
#include "sections/08-resolution-scaling.typ"
#include "sections/09-iterations.typ"
#include "sections/10-dielectric-contrast.typ"
#include "sections/11-band-count.typ"
#include "sections/12-conclusion.typ"

#bibliography(
  "/typst/references.bib",
  title: "References",
  style: "american-physics-society",
)
