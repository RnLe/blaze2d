// Report template shared by every post converted from the website.
//
// Front matter follows the master thesis (../master-thesis/thesisV2.typ): a
// cover with rules above and below the title, an imprint page, roman-numbered
// preliminaries, a table of contents and an abstract, then arabic numbering
// from the body onwards.
//
// Usage:
//   #import "/typst/lib/report.typ": report, web-figure
//   #show: report.with(title: "...", kicker: "Technical Report", ...)

#import "/typst/lib/theme.typ": theme, col-dark-steel

// The `title` element, captured before the `report` parameter of the same name
// shadows it inside the function body.
#let _doc-title = title

// ── Figures exported from the website ────────────────────────────────────
//
// These take already-constructed content rather than a path, because Typst
// resolves a relative `image()` path against the file the literal is written
// in. A path passed through this module would resolve against lib/, not
// against the report. Each report therefore builds its own one-line helper
// around `image()` and hands the result here.
//
// The chart SVGs carry their own #000000 background and 16px of padding,
// because that is the page they are designed against. All that is added here
// is the rounded clip that turns a plot into a card on a white page.
#let web-figure(body, radius: 6pt) = block(radius: radius, clip: true, body)

// Several exported figures in one row inside a single numbered figure,
// matching the TM/TE pairs the website lays out in a flex row.
#let web-figure-row(items, gutter: 0.9em, radius: 6pt) = grid(
  columns: items.map(_ => 1fr),
  column-gutter: gutter,
  ..items.map(item => web-figure(item, radius: radius)),
)

#let _rule = align(center + horizon)[#line(length: 100%)]

// One row of the imprint table.
#let _imprint-row(label, value) = ([#emph(label):], value)

#let report(
  title: none,
  subtitle: none,
  kicker: "Technical Report",
  author: none,
  date: none,
  keywords: none,
  project: none,
  affiliation: none,
  logo: none,
  marks: (),
  links: (),
  abstract: none,
  abstract-title: "Abstract",
  body,
) = {
  set document(title: title, author: if author == none { () } else { (author,) })
  show _doc-title: set align(center + horizon)

  // ── Cover ───────────────────────────────────────────────────────────────
  grid(
    columns: (1fr, 1fr),
    if logo != none { image(logo, width: 22%) } else { [] },
    align(right + horizon)[
      #for mark in marks [
        #box(baseline: 0.15em, image(mark, height: 1.6em)) #h(0.8em)
      ]
    ],
  )

  block(above: 4em)
  align(center)[
    #text(if affiliation != none { affiliation } else { [] }, size: 15pt)
  ]

  block(above: -2em)
  align(center + horizon)[
    #text(kicker, size: 15pt, font: "Latin Modern Roman Caps")
  ]
  _rule
  block(below: 2.0em)
  _doc-title(title)
  block(below: 2.0em)
  _rule
  block(below: 1.6em)

  if subtitle != none {
    align(center)[#text(size: 11pt, style: "italic")[#subtitle]]
  }
  block(below: 3.0em)

  grid(
    columns: (1fr, 1fr),
    align(left)[
      _Author:_ \
      #author
    ],
    align(right)[
      #for entry in links.filter(e => e.at("on-cover", default: true)) [
        #emph[#entry.label:] \
        #link(entry.url)[#entry.text] \
      ]
    ],
  )

  align(center + bottom)[
    Published \
    #date
  ]

  pagebreak()

  // ── Imprint ─────────────────────────────────────────────────────────────
  page(margin: (x: 4cm, y: 3cm))[
    #set text(size: 9pt)
    #align(bottom)[
      #text("Imprint", size: 18pt, weight: "bold") \
      #block(below: 2em)
      #table(
        columns: 2,
        align: (top + left, top + left),
        stroke: 0pt,
        gutter: 0pt,
        fill: none,
        .._imprint-row("Project", if project != none { project } else { kicker }),
        .._imprint-row("Title", title),
        .._imprint-row("Author", author),
        .._imprint-row("Date", date),
        ..if keywords != none { _imprint-row("Keywords", keywords) } else { () },
      )
      #block(below: 2em)
      #for entry in links [
        #emph(entry.label): #link(entry.url)[#entry.url] \
      ]
    ]
  ]

  pagebreak()

  // ── Preliminaries: roman numbering ──────────────────────────────────────
  set page(numbering: "i", number-align: top + right)

  outline(
    title: [
      #text("Contents", size: 30pt)
      #block(above: 20pt)
    ],
    depth: 2,
  )

  if abstract != none {
    pagebreak()
    set par(justify: false)
    set align(center)
    align(center)[#text(abstract-title, size: 16pt, weight: "bold")]
    v(1em)
    abstract
  }

  // ── Body: arabic numbering, decorated headings ──────────────────────────
  pagebreak()
  set heading(numbering: "1.1.1.1")
  set page(numbering: "1", number-align: bottom + center)
  counter(page).update(1)

  show: theme
  body
}
