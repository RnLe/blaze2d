#!/usr/bin/env python3
"""Build the extended OpenAI Sans faces used to render the exported chart SVGs.

Two problems are solved here.

1. Typst reads TTF/OTF but not WOFF2, and the website ships OpenAI Sans only as
   WOFF2. The faces are unwrapped to plain TTF.

2. OpenAI Sans has no Greek and no mathematical operators, yet the chart axes
   are labelled with them ("omega a / 2 pi c", the Gamma point, epsilon sweeps,
   "proportional to N^1.09"). In a browser those characters quietly fall through
   the font-family list. Typst renders SVG through resvg, which resolves *one*
   font per <text> element rather than per glyph: a single uncovered character
   drags the entire label onto a fallback face, so "proportional to N^1.09"
   comes out in a serif inside an otherwise sans chart. Adding a fallback
   <tspan> does not help, because the fallback still happens at element level.

   The fix is to make the primary face cover everything the labels need. Glyphs
   the base lacks are borrowed from donor faces, scaled to the base's em square,
   and merged in. Roboto is the first donor because it is already in the site's
   own fallback chain, so those glyphs are what a Linux or Android visitor
   actually sees; DejaVu Sans supplies the mathematical operators Roboto lacks.

   The result is written as a distinct family, "OpenAI Sans Extended", so it is
   never confused with the shipped face. The exporter emits it ahead of plain
   "OpenAI Sans" in the font-family list, which means the SVGs still render
   exactly like the page in a browser that has neither.

Usage:
    /home/renlephy/miniforge3/bin/python build-fonts.py
"""

from __future__ import annotations

import glob
import io
import sys
from pathlib import Path

from fontTools.merge import Merger
from fontTools.subset import Subsetter
from fontTools.ttLib import TTFont
from fontTools.ttLib.scaleUpem import scale_upem

HERE = Path(__file__).resolve().parent
TYPST_ROOT = HERE.parent.parent
SRC = TYPST_ROOT.parent / "web" / "public" / "fonts"
DST = TYPST_ROOT / "assets" / "fonts"

BASE_FAMILY = "OpenAI Sans"
EXTENDED_FAMILY = "OpenAI Sans Extended"

# Ranges the chart labels draw from beyond OpenAI Sans's own repertoire.
# Deliberately wider than today's figures need, so a future report that uses
# another symbol does not require rebuilding the fonts.
EXTRA_RANGES = [
    (0x0370, 0x03FF),   # Greek and Coptic       (Gamma, epsilon, pi, omega, ...)
    (0x1F00, 0x1FFF),   # Greek Extended
    (0x2070, 0x209F),   # super- and subscripts
    (0x2190, 0x21FF),   # arrows
    (0x2200, 0x22FF),   # mathematical operators (proportional to, infinity, ...)
    (0x2A00, 0x2AFF),   # supplemental mathematical operators
]

# Donor faces per base weight, in priority order. Roboto matches the site's own
# fallback chain; DejaVu Sans is the catch-all for symbols Roboto lacks.
DONORS = {
    "Regular": ["Roboto-Regular.ttf", "DejaVuSans.ttf"],
    "Medium": ["Roboto-Medium.ttf", "DejaVuSans.ttf"],
    "SemiBold": ["Roboto-Medium.ttf", "DejaVuSans.ttf"],
    "Bold": ["Roboto-Bold.ttf", "DejaVuSans-Bold.ttf"],
}

# Tables kept from a donor: outlines, metrics, character map and the bookkeeping
# a valid font needs. Anything else (layout, math, hinting metadata, vendor
# tables) is dropped so the merge stays predictable.
DONOR_KEEP_TABLES = {
    "head", "hhea", "maxp", "OS/2", "hmtx", "cmap", "loca", "glyf", "name", "post",
}

FONT_SEARCH_ROOTS = ["/usr/share/fonts", "/usr/local/share/fonts", str(Path.home() / ".fonts")]


def find_font(filename: str) -> Path | None:
    for root in FONT_SEARCH_ROOTS:
        matches = glob.glob(f"{root}/**/{filename}", recursive=True)
        if matches:
            return Path(sorted(matches)[0])
    return None


def coverage(font: TTFont) -> set[int]:
    covered: set[int] = set()
    for table in font["cmap"].tables:
        covered.update(table.cmap.keys())
    return covered


def wanted_codepoints() -> set[int]:
    return {cp for lo, hi in EXTRA_RANGES for cp in range(lo, hi + 1)}


def compile_to_buffer(font: TTFont) -> io.BytesIO:
    """Serialise a font so the merger, which only opens files, can read it."""
    buffer = io.BytesIO()
    font.save(buffer)
    buffer.seek(0)
    return buffer


def subset_donor(path: Path, codepoints: set[int], upem: int):
    """Cut a donor down to the requested codepoints and match the base em square.

    Returns (buffer, added_codepoints), or None when the donor has nothing to
    contribute.
    """
    donor = TTFont(str(path))
    available = codepoints & coverage(donor)
    if not available:
        return None

    subsetter = Subsetter()
    subsetter.options.layout_features = []
    subsetter.options.hinting = False
    subsetter.options.notdef_outline = False
    subsetter.options.name_IDs = []
    subsetter.populate(unicodes=available)
    subsetter.subset(donor)

    # Donors carry tables the merger has no rules for (DejaVu ships MATH, for
    # instance). Only outlines, metrics and the character map are wanted, so
    # everything else is dropped rather than enumerated table by table.
    for tag in set(donor.keys()) - DONOR_KEEP_TABLES:
        del donor[tag]

    if donor["head"].unitsPerEm != upem:
        scale_upem(donor, upem)

    added = available & coverage(donor)
    return compile_to_buffer(donor), added


def rename_family(font: TTFont) -> None:
    """Re-badge every family-bearing name record without disturbing the style."""
    spaced = (BASE_FAMILY, EXTENDED_FAMILY)
    tight = (BASE_FAMILY.replace(" ", ""), EXTENDED_FAMILY.replace(" ", ""))

    for record in font["name"].names:
        try:
            value = record.toUnicode()
        except UnicodeDecodeError:
            continue
        if spaced[0] in value:
            record.string = value.replace(*spaced)
        elif tight[0] in value:
            record.string = value.replace(*tight)


def build_face(src: Path, targets: set[int]) -> tuple[Path, int, list[str]]:
    base = TTFont(str(src))
    base.flavor = None                                  # drop the woff2 wrapper
    upem = base["head"].unitsPerEm
    missing = targets - coverage(base)

    style = src.stem.split("-", 1)[1] if "-" in src.stem else "Regular"

    buffers = [compile_to_buffer(base)]
    used: list[str] = []

    for donor_name in DONORS.get(style, []):
        if not missing:
            break
        donor_path = find_font(donor_name)
        if donor_path is None:
            continue
        piece = subset_donor(donor_path, missing, upem)
        if piece is None:
            continue
        buffer, added = piece
        missing -= added
        buffers.append(buffer)
        used.append(f"{donor_path.name} (+{len(added)})")

    # The base is listed first, so it wins every cmap conflict and none of its
    # own glyphs can be displaced by a donor.
    merged = Merger().merge(buffers) if len(buffers) > 1 else TTFont(buffers[0])
    rename_family(merged)

    out = DST / f"{src.stem}-Extended.ttf"
    merged.save(str(out))
    return out, len(coverage(merged)), used


def main() -> int:
    if not SRC.is_dir():
        print(f"error: source font directory not found: {SRC}", file=sys.stderr)
        return 1

    sources = sorted(SRC.glob("*.woff2"))
    if not sources:
        print(f"error: no .woff2 files in {SRC}", file=sys.stderr)
        return 1

    DST.mkdir(parents=True, exist_ok=True)
    for stale in DST.glob("*.ttf"):
        stale.unlink()

    targets = wanted_codepoints()
    for src in sources:
        out, covered, used = build_face(src, targets)
        donors = ", ".join(used) if used else "no donors needed"
        print(f"  {src.name:28s} -> {out.name:34s} {covered:5d} codepoints  [{donors}]")

    print(f"\n{len(sources)} face(s) built as '{EXTENDED_FAMILY}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
