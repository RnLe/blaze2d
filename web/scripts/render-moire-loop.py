#!/usr/bin/env python3
"""Renders the moiré bilayer loop used by the Master's Thesis card.

Run by hand, not from `prepare:site`: it needs Python with numpy, matplotlib
and Pillow, and its output is committed.

    python3 web/scripts/render-moire-loop.py

The animation that shipped on the portfolio site had no generator under version
control anywhere -- the commit that added it says only that it was "rendered
from the thesis scripts". This reproduces it from the geometry in
master-thesis/assets/python/moire_hex_bilayer.py so that it can be re-rendered
whenever the palette, the framing or the resolution needs to change.

One honeycomb layer is fixed; the other turns through a full 60 degrees over
the loop. A honeycomb is six-fold symmetric about a hexagon centre, so after 60
degrees the turning layer lands exactly on itself and the last frame meets the
first -- the rotation reads as endless rather than as a sweep that jumps back.
The moiré period it beats against the fixed layer is L = a / (2 sin(theta/2)),
which is what appears to move.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# ---- Output -----------------------------------------------------------------
OUT = Path(__file__).resolve().parent.parent / "public" / "banners" / "blaze_thesis_moire.webp"

# 16:9, to sit in the card's picture frame without cropping.
#
# A full turn costs frames, and frames are the whole file: at the speed the
# earlier eight-degree sweep turned at, sixty degrees would run for half a
# minute and weigh several megabytes. This holds the loop near a quarter of a
# minute by playing slowly rather than by adding frames -- the dots move under
# two pixels between frames, so eleven a second still reads as smooth.
WIDTH, HEIGHT = 576, 324
FRAMES = 180
FRAME_MS = 95

# Flat colour on a flat background, so this compresses as graphics, not as
# photography: quantising to a shared palette and writing lossless is roughly a
# third the size of lossy WebP, which spends its bits on ringing around every
# dot edge. One palette for the whole loop also keeps the colours from crawling
# between frames and gives the encoder something to difference against.
#
# The palette is built from the four colours that are actually drawn rather than
# measured off the frames. Median cut divides by how many pixels sit where, and
# the antialiased rims outnumber the dot centres by enough that it spends its
# slots on rim blends and slides the fill off the brand colour: at twelve
# colours it moved every orange dot to #b66a35, with not one pixel of #eb7929
# left in the frame.
#
# One step between each pair is enough: the rims are a pixel wide, so a finer
# ramp buys nothing the eye can find at this size and costs half the file again.
BLEND_STEPS = 1

# ---- Palette: the Blaze2D mark ----------------------------------------------
BLAZE_BLUE = "#317cb8"
BLAZE_ORANGE = "#eb7929"
BACKGROUND = "#000000"
DOT_EDGE = "#ffffff"
# Scaled to the view: the dots keep the same size relative to the lattice, so
# widening the frame shows more of the pattern rather than the same pattern
# drawn larger. Area goes as the square of the linear scale, the edge does not.
DOT_AREA = 11.5  # matplotlib marker area, points^2
DOT_EDGE_WIDTH = 0.45

# ---- Lattice ----------------------------------------------------------------
A = 1.0
A1 = A * np.array([1.0, 0.0])
A2 = A * np.array([0.5, np.sqrt(3.0) / 2.0])
BASIS_B = (A1 + A2) / 3.0  # honeycomb's second atom

# Half-width of the view, in lattice constants. The moiré is a beat between the
# two layers, so its period is far longer than either lattice: the frame has to
# hold several beats before the superlattice reads as a pattern rather than as
# pairs of dots. Only the turning layer changes between frames, but its dots are
# everywhere, so the dot count sets the file size almost on its own.
VIEW_X = 17.0
VIEW_Y = VIEW_X * HEIGHT / WIDTH

# A honeycomb maps onto itself under a sixth of a turn about a hexagon centre,
# so this is the shortest sweep that closes. The centre is two thirds along the
# cell diagonal; rotating about anything else -- an atom, for instance -- is
# only three-fold and would need twice the frames to come back.
FULL_TURN = 60.0  # degrees
HEX_CENTRE = 2.0 * (A1 + A2) / 3.0


def cells_to_cover(radius: float) -> int:
    """Cells along each axis needed to fill a disc of `radius` lattice constants.

    A rotation can bring any point within that radius into the frame, so the
    lattice has to be built over the disc rather than the rectangle. The axes
    are sixty degrees apart, not square: a point at (x, y) sits at
    m = 2y/sqrt(3) and n = x - y/sqrt(3), and both run up to 2/sqrt(3) times the
    radius around the circle. Reading the count off the diagonal instead is
    short by a hair's breadth here and by more the further the view opens out.
    """
    return int(np.ceil(2.0 / np.sqrt(3.0) * radius / A)) + 1


def honeycomb(reach: int) -> np.ndarray:
    """Honeycomb atoms within `reach` cells, centred on a hexagon centre."""
    index = np.arange(-reach, reach + 1)
    n, m = (axis.ravel() for axis in np.meshgrid(index, index))
    cells = np.outer(n, A1) + np.outer(m, A2)
    return np.vstack([cells, cells + BASIS_B]) - HEX_CENTRE


def rotated(points: np.ndarray, radians: float) -> np.ndarray:
    cos, sin = np.cos(radians), np.sin(radians)
    return points @ np.array([[cos, sin], [-sin, cos]])


def visible(points: np.ndarray) -> np.ndarray:
    """Drops what falls outside the frame, so each frame draws far fewer dots."""
    margin = 1.5 * A
    inside = (np.abs(points[:, 0]) <= VIEW_X + margin) & (np.abs(points[:, 1]) <= VIEW_Y + margin)
    return points[inside]


def brand_palette() -> Image.Image:
    """The drawing's own colours, plus the blends its antialiasing produces.

    Everything in a frame is one of four colours or an edge between two of them,
    so naming them outright keeps the fill exactly on the brand colour however
    many rim pixels a zoom level happens to produce. Orange and blue never meet
    without white between them, so only their blends towards black and towards
    white are worth a slot.
    """
    black, white = np.zeros(3), np.full(3, 255.0)
    orange = np.array([int(BLAZE_ORANGE[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    blue = np.array([int(BLAZE_BLUE[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)
    stops = [black, white, orange, blue]
    for near, far in ((orange, black), (blue, black), (orange, white), (blue, white)):
        stops += [near + (far - near) * step / (BLEND_STEPS + 1) for step in range(1, BLEND_STEPS + 1)]
    table = np.clip(np.array(stops), 0, 255).astype(np.uint8).ravel().tolist()
    palette = Image.new("P", (1, 1))
    palette.putpalette(table + [0] * (768 - len(table)))
    return palette


def render() -> None:
    lattice = honeycomb(cells_to_cover(np.hypot(VIEW_X, VIEW_Y) + 1.5 * A))

    # The figure carries the background, not the axes: `axis("off")` turns the
    # axes frame off, and with it the patch its facecolor would have painted.
    figure = plt.figure(figsize=(WIDTH / 100, HEIGHT / 100), dpi=100, facecolor=BACKGROUND)
    axes = figure.add_axes((0, 0, 1, 1))
    fixed = visible(lattice)
    frames = []
    for step in range(FRAMES):
        # Exclusive of a full turn: the frame that would repeat the first one is
        # the loop point itself, so drawing it would stutter.
        turn = np.radians(FULL_TURN * step / FRAMES)
        axes.clear()
        for shown, colour in ((fixed, BLAZE_BLUE), (visible(rotated(lattice, turn)), BLAZE_ORANGE)):
            axes.scatter(shown[:, 0], shown[:, 1], s=DOT_AREA, c=colour,
                         edgecolors=DOT_EDGE, linewidths=DOT_EDGE_WIDTH, zorder=2)
        axes.set_xlim(-VIEW_X, VIEW_X)
        axes.set_ylim(-VIEW_Y, VIEW_Y)
        axes.set_aspect("equal")
        axes.axis("off")
        figure.canvas.draw()
        frames.append(Image.frombuffer(
            "RGBA", figure.canvas.get_width_height(), figure.canvas.buffer_rgba(), "raw", "RGBA", 0, 1
        ).convert("RGB"))
    plt.close(figure)

    palette = brand_palette()
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE).convert("RGB")
              for frame in frames]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(OUT, format="WEBP", save_all=True, append_images=frames[1:],
                   duration=FRAME_MS, loop=0, lossless=True, method=6, minimize_size=True)
    print(f"{OUT.relative_to(OUT.parents[3])}: {len(frames)} frames, "
          f"{WIDTH}x{HEIGHT}, {OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    render()
