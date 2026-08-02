// Exported chart figures for this report.
//
// Typst resolves a relative `image()` path against the file the literal is
// written in, so these helpers live beside the figures rather than in the
// shared template or in the individual sections.
//
// Regenerate the SVGs with `make figures REPORT=blaze`.

#import "/typst/lib/report.typ": web-figure, web-figure-row

/// One exported chart, as a rounded black card.
#let plot(name, width: 100%) = web-figure(
  image("figures/" + name + ".svg", width: width),
)

/// Several exported charts side by side inside one numbered figure.
#let plot-row(..names) = web-figure-row(
  names.pos().map(name => image("figures/" + name + ".svg", width: 100%)),
)
