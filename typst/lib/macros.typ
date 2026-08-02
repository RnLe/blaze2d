// Reusable content blocks shared by the reports.
//
// `equation-box` and `placeholder-box` come from the master thesis
// (../master-thesis/macros.typ); `callout-box` generalises the note boxes from
// its preface so a report can carry the highlighted asides that the website
// posts use.

#import "/typst/lib/theme.typ": (
  col-sky-blue, col-steel-blue, col-light-blue, col-stark-orange,
)

#let ub(x) = $upright(bold(#x))$

// Highlighted box for key equations and statements.
#let equation-box(
  body,
  stroke-color: col-steel-blue,
  fill-color: col-light-blue,
  title: none,
  title-size: 9pt,
  title-weight: "bold",
  title-color: col-steel-blue,
  inner-pad: 0pt,
) = {
  align(center)[
    #box(
      width: 100%,
      stroke: 1.2pt + stroke-color,
      fill: fill-color.lighten(92%),
      radius: 0pt,
      inset: 12pt,
      {
        if title != none {
          place(top + left, dx: -8pt, dy: -8pt,
            text(size: title-size, weight: title-weight, fill: title-color, title))
        }
        v(inner-pad)
        body
        v(inner-pad)
      },
    )
  ]
}

// Placeholder for a figure that does not exist yet.
#let placeholder-box(body, width: 100%) = rect(
  width: width,
  stroke: 2.5pt + col-steel-blue,
  fill: col-light-blue.lighten(70%),
  radius: 4pt,
  inset: 12pt,
  [
    #set align(center)
    #set text(fill: col-steel-blue, weight: "bold", size: 11pt)
    #body
  ],
)

// Tinted aside with a coloured rule down one side.
//
// This is the print form of the bordered note boxes the website posts use for
// benchmark configuration and scope caveats.
#let callout-box(
  body,
  title: none,
  color: col-steel-blue,
  stroke-pos: "left",
  icon: none,
) = block(
  width: 100%,
  fill: color.lighten(90%),
  stroke: if stroke-pos == "left" { (left: 4pt + color) } else { (right: 4pt + color) },
  inset: (x: 1.2em, y: 1em),
  radius: if stroke-pos == "left" { (right: 4pt) } else { (left: 4pt) },
  [
    #if icon != none {
      place(top + right, dx: 0em, dy: -0.2em, image(icon, height: 2.2em))
    }
    #if title != none {
      text(weight: "bold", fill: color.darken(20%), size: 1.1em)[#title]
      v(0.5em)
    }
    #body
  ],
)

// Inline icons, sized and baseline-corrected to sit inside a line of text.
#let _icon(path, baseline: 0.15em) = box(baseline: baseline, image(path, height: 1em))

#let icon-github = _icon("/typst/assets/images/github.svg")
#let icon-python = _icon("/typst/assets/images/python.svg")
#let icon-pypi   = _icon("/typst/assets/images/pypi.svg")
#let icon-rust   = _icon("/typst/assets/images/rust.svg")
#let icon-blaze  = _icon("/typst/assets/images/blaze.svg", baseline: 0.2em)
