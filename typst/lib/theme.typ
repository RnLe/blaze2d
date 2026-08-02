// Shared visual identity for the Typst reports.
//
// Lifted from the master thesis (../master-thesis/thesisV2.typ) so a report and
// the thesis chapter on the same subject read as one body of work: the same
// palette, the same table treatment, the same chapter banners and fading
// margin stripes.

// ── Colour palette (master-thesis/assets/guides/StyleGuide.md) ────────────
#let col-sky-blue     = rgb("#4E9AE1")
#let col-gentle-brown = rgb("#AB8954")
#let col-stark-orange = rgb("#EBA538")
#let col-dusty-orange = rgb("#E3B064")
#let col-steel-blue   = rgb("#4D7B9E")
#let col-dusty-brown  = rgb("#857255")
#let col-dark-steel   = rgb("#4F5F6B")
#let col-light-brown  = rgb("#E3D5BF")
#let col-light-blue   = rgb("#A5C6DF")
#let col-sage-green   = rgb("#6B8F71")

// Chapter accents, cycled by heading number. Six entries rather than the
// thesis's five: a report maps one web section to one chapter, so the sequence
// runs longer and a shorter cycle would repeat a colour on adjacent chapters.
#let ch-palette = (
  col-steel-blue,
  col-sage-green,
  col-dusty-brown,
  col-dark-steel,
  col-dusty-orange,
  col-gentle-brown,
)

// ── Heading decoration geometry ───────────────────────────────────────────
// Knob names and values match the thesis so the two documents line up when
// placed side by side.
#let _sw   = 2pt      // stripe width
#let _sg   = 2pt      // gap between lanes
#let _sm   = 0.2cm    // distance from the text edge to the lane-3 centre
#let _sf   = 3cm      // stripe fade-out length
#let _vdy  = 5pt      // vertical shift applied to every stripe
#let _udy  = -2pt     // vertical shift for heading underlines
#let _l1dy = -23.6pt  // extra shift for the chapter lane, which starts at the banner
#let _bsw  = 2pt      // banner left-stroke width
#let _bdx  = 2.5pt    // horizontal nudge of the banner's left edge

#let _page-h        = 29.7cm   // A4
#let _margin-b      = 2.5cm    // Typst's default bottom margin
#let _content-bottom = _page-h - _margin-b

// Left-edge offset for lane n (lane 1 = outermost / chapter, lane 3 = nearest the text).
#let _lane-dx(n) = {
  let center = _sm + (3 - n) * (_sw + _sg)
  -(center + _sw / 2)
}

// The accent for the chapter currently being typeset.
#let ch-color() = context {
  let idx = counter(heading).get().at(0, default: 1)
  ch-palette.at(calc.rem(idx - 1, ch-palette.len()))
}

#let _ch-col() = {
  let idx = counter(heading).get().at(0, default: 1)
  ch-palette.at(calc.rem(idx - 1, ch-palette.len()))
}

// One fading stripe in lane n, clamped so it never bleeds into the bottom margin.
#let _stripe(col, n, pos, dy: 0pt) = {
  let y-start = pos.y + dy + _vdy
  let avail   = _content-bottom - y-start
  let h       = calc.min(_sf, calc.max(0pt, avail))
  if h > 0pt {
    place(
      dx: _lane-dx(n),
      dy: dy + _vdy,
      block(clip: true,
        rect(width: _sw, height: h,
             fill: gradient.linear(col, white, angle: 90deg))
      )
    )
  }
}

// Rule running from the innermost stripe across the width of the heading.
#let _heading-underline(col, n, heading-width, dy: 0pt) = {
  let start-x = _lane-dx(n)
  place(
    dx: start-x,
    dy: dy + _udy,
    rect(width: heading-width - start-x, height: _sw, fill: col)
  )
}

#let _banner-outset = _sm + 2 * (_sw + _sg) + _sw / 2 + 1.5pt - _bdx

// The tinted, left-ruled block that marks a chapter opening.
#let chapter-banner(body, col) = block(
  outset: (left: _banner-outset),
  inset: (left: _banner-outset + 0.3em, top: 0.35em, bottom: 0.35em, right: 0.3em),
  fill: col.lighten(85%),
  stroke: (left: _bsw + col),
  above: 2.4em,
  below: 1.2em,
)[#body]

// ── The theme itself ──────────────────────────────────────────────────────
// Applied with `#show: theme` after the front matter, so the cover and the
// table of contents keep their own undecorated headings.
#let theme(body) = {
  set par(justify: true)
  set text(size: 10pt)

  // Gutter-based table styling: no strokes, a dark header row, and data rows
  // alternating warm and cool (StyleGuide.md).
  set table(
    stroke: none,
    gutter: 0.18em,
    inset: (x: 0.8em, y: 0.55em),
    fill: (x, y) => {
      if y == 0 { col-dark-steel }
      else if calc.odd(y) { col-light-brown }
      else { col-light-blue }
    },
  )
  show table.cell.where(y: 0): it => {
    set text(fill: white, weight: "bold", size: 11pt)
    it
  }

  // Captions sit a step below body text throughout the thesis.
  show figure.caption: set text(size: 9pt)

  // Level 1 - chapter: banner plus the outermost fading stripe.
  show heading.where(level: 1): it => context {
    let col = _ch-col()
    let pos = here().position()
    set align(left)
    chapter-banner(it, col)
    _stripe(col, 1, pos, dy: _l1dy)
  }

  // Level 2 - section: lane-2 stripe and an underline, in a lighter tint.
  show heading.where(level: 2): it => context {
    let col = _ch-col()
    let pos = here().position()
    let sz = measure(it)
    it
    let back = -sz.height
    _stripe(col.lighten(40%), 2, pos, dy: back)
    _heading-underline(col.lighten(40%), 2, sz.width, dy: back + sz.height - _sw)
  }

  // Level 3 - subsection: lane-3 stripe, lighter again.
  show heading.where(level: 3): it => context {
    let col = _ch-col()
    let pos = here().position()
    let sz = measure(it)
    it
    let back = -sz.height
    _stripe(col.lighten(65%), 3, pos, dy: back)
    _heading-underline(col.lighten(65%), 3, sz.width, dy: back + sz.height - _sw)
  }

  body
}
